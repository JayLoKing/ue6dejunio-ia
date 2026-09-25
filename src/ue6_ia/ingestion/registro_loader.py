"""Carga de las notas por criterio desde el registro trimestral.

QUE DEVUELVE
------------
Una fila por (estudiante, area, trimestre), y en cada una las **listas** de notas
que el docente cargo en cada dimension. No los promedios: el promedio es
justamente lo que hay que dejar de tirar, porque 90/90/20 y 67/67/66 dan el mismo
numero y no son lo mismo.

DE DONDE SALE CADA DIMENSION
----------------------------
No todas viven en el mismo lugar, y eso cambio entre gestiones:

- `knowing` y `doing` (SABER y HACER) tienen sus criterios en la hoja del area,
  siempre.
- `being` y `deciding` (SER y DECIDIR) se califican **una vez por trimestre para
  todo el curso**, en la hoja 'EVAL SER Y DECIDIR'. La hoja del area solo copia el
  numero ya resuelto. Por eso el mismo par de listas acompaña a las nueve areas de
  un estudiante en ese trimestre.
- En 2026 SER vuelve a la hoja del area con criterios propios. Si estan ahi, se
  leen de ahi; si no, de la hoja aparte.

VOCABULARIO
-----------
La planilla dice SER / SABER / HACER / DECIDIR y el modelo dice being / knowing /
doing / deciding, que son los valores exactos que la BDD acepta en
`evaluation_criteria.dimension`. La traduccion vive aca, en el unico lugar que
conoce las dos puntas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import AppConfig
from .bloques import aplanar_cabecera, columnas_de_criterios, normalizar
from .discovery import CursoFolder
from .excel_reader import find_file, list_sheets, read_sheet_grid

logger = logging.getLogger(__name__)

# Como se llama cada dimension en la planilla y como en el modelo. El orden es el
# del `total_score` generado en la BDD.
DIMENSION_POR_COLUMNA: dict[str, tuple[str, ...]] = {
    "being": ("SER",),
    "knowing": ("SABER",),
    "doing": ("HACER",),
    # 2023 y 2025 la llaman DECIDIR; 2026, AUTOEVALUACION a secas. Es el mismo
    # casillero de 5 puntos.
    "deciding": ("DECIDIR", "AUTOEVALUACION"),
}

# Un alias no puede quedarse con la autoevaluacion de otra dimension. En 2023 hay
# dos columnas: 'AUTOEVALUACION - SER 5' y 'AUTOEVALUACION - DECIDIR 5', y el
# alias de `deciding` matchearia la primera —la de SER— dejando la nota de Ser
# medida contra el tope de Decidir. Ninguna fila falla y las dos leen mal.
_DIMENSIONES_AJENAS: dict[str, tuple[str, ...]] = {
    "deciding": ("SER", "SABER", "HACER"),
}

# Las que la hoja del area siempre califica por criterio.
DIMENSIONES_DE_AREA = ("knowing", "doing")

# Las que pueden venir de la hoja de curso.
DIMENSIONES_DE_CURSO = ("being", "deciding")

HOJA_SER_DECIDIR = "EVAL SER Y DECIDIR"

# Cuantas filas de encabezado se miran hacia arriba desde la primera de alumnos.
# Alcanzan para el nombre del bloque, los nombres de criterio y 'APELLIDOS Y
# NOMBRE(S)', y dejan afuera los titulos del membrete, que no describen columnas.
_ALTURA_CABECERA = 3


def _a_float(valor: object) -> float | None:
    if valor is None or valor == "":
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _es_entero(valor: object) -> bool:
    return (isinstance(valor, int) and not isinstance(valor, bool)) or (
        isinstance(valor, float) and valor == int(valor)
    )


# Como se anuncia la columna de nombres, ya normalizada.
_ENCABEZADO_NOMBRE = "APELLIDOS"
_COL_NOMBRE_POR_DEFECTO = 1


# Cuantas filas de arriba se miran buscando el encabezado de nombres.
_BANDA_CABECERA = 12


def _columna_de_nombres(grid: list[list]) -> int:
    """Donde dice 'APELLIDOS Y NOMBRE(S)', no la columna 1 por costumbre.

    Se busca sobre la grilla cruda y **antes** de ubicar la primera fila de
    alumnos, no despues: localizar al primer alumno ya necesita saber en que
    columna mirar el nombre. Haciendolo al reves, un template que corriera la
    columna un lugar no encontraba ninguna fila, la hoja entera se perdia, y la
    deteccion por encabezado quedaba de adorno.
    """
    for fila in grid[:_BANDA_CABECERA]:
        for col, celda in enumerate(fila or []):
            if _ENCABEZADO_NOMBRE in normalizar(celda):
                return col
    logger.warning(
        "Sin encabezado de nombres ('%s') en las primeras %d filas; se asume la " "columna %d.",
        _ENCABEZADO_NOMBRE,
        _BANDA_CABECERA,
        _COL_NOMBRE_POR_DEFECTO,
    )
    return _COL_NOMBRE_POR_DEFECTO


def _fila_primer_estudiante(grid: list[list], col_nombre: int) -> int:
    """La primera fila numerada que ademas trae un nombre en esa columna."""
    for fila, celdas in enumerate(grid):
        primera = celdas[0] if celdas else None
        if _es_entero(primera) and _nombre_de(celdas, col_nombre):
            return fila
    return -1


def _nombre_de(fila: list, col_nombre: int) -> str | None:
    valor = fila[col_nombre] if len(fila) > col_nombre else None
    return str(valor).strip() if valor and str(valor).strip() else None


def _cabecera(grid: list[list], fila_est: int) -> list[str]:
    return aplanar_cabecera(grid[max(0, fila_est - _ALTURA_CABECERA) : fila_est])


def _declarados_por_dimension(columnas: dict[str, list[int]]) -> dict[str, int]:
    """Cuantos criterios tiene cada dimension: las mismas columnas que se leen.

    Contando exactamente sobre las que `_notas` recorre, el numerador de
    `progress_pct` nunca puede pasar al denominador — una nota calificada siempre
    sale de una columna contada. Deducirlo del texto del encabezado, en cambio,
    fallaba justo donde el bloque es una sola columna que **trae la nota** y no
    nombra ningun criterio: DECIDIR en 2023. Ahi el divisor daba 0, el numerador
    contaba la nota igual, y el progreso se iba arriba de 100 para clavarse en el
    tope sin que fallara una fila.
    """
    return {dimension: len(cols) for dimension, cols in columnas.items()}


def _columnas_por_dimension(
    cabecera: list[str], dimensiones: tuple[str, ...]
) -> dict[str, list[int]]:
    """Las columnas de criterio de cada dimension.

    Una dimension sin columnas no es un error: puede que la hoja no la califique.
    Quien llama decide si buscarla en otro lado o dejarla vacia.
    """
    columnas: dict[str, list[int]] = {}
    for dimension in dimensiones:
        columnas[dimension] = []
        for nombre_planilla in DIMENSION_POR_COLUMNA[dimension]:
            candidatas = columnas_de_criterios(cabecera, nombre_planilla)
            if not candidatas:
                continue
            if _menciona_otra_dimension(cabecera[candidatas[0]], dimension):
                logger.debug(
                    "Bloque '%s' descartado para %s: su encabezado (%r) nombra otra " "dimension.",
                    nombre_planilla,
                    dimension,
                    cabecera[candidatas[0]],
                )
                continue
            columnas[dimension] = candidatas
            break
    return columnas


def _menciona_otra_dimension(encabezado: str, dimension: str) -> bool:
    """Si el encabezado nombra, como palabra entera, otra dimension.

    Por token y no por subcadena. `normalizar` deja el texto sin espacios, y el
    encabezado aplanado suele traer tambien el nombre del primer criterio: un
    DECIDIR cuyo criterio diga "Observacion de la conducta" queda
    'DECIDIR-10 OBSERVACIONDELACONDUCTA', donde 'SER' aparece adentro de
    ObSERvacion. Buscando la subcadena, el bloque entero se descartaba en silencio.
    Lo mismo con 'asertivo', 'conserva', 'reserva'.

    Lo que si tiene que rechazar es 'AUTOEVALUACION - SER 5', donde SER es una
    palabra propia: ahi la autoevaluacion pertenece a Ser y no a Decidir.
    """
    palabras = {p for p in re.split(r"[^A-Z]+", encabezado) if p}
    return any(ajena in palabras for ajena in _DIMENSIONES_AJENAS.get(dimension, ()))


def _columna_promedio_trimestral(cabecera: list[str]) -> int | None:
    """La columna con el resultado del area en el trimestre.

    Es la fuente de la etiqueta, no una feature. Cierra la fila entera, por eso se
    reconoce por decir TRIMESTRAL ademas de PROMEDIO.
    """
    for col, texto in enumerate(cabecera):
        if "PROMEDIO" in texto and "TRIMESTRAL" in texto:
            return col
    # Sin esta columna no hay etiqueta, y sin etiqueta el area entera desaparece
    # del entrenamiento al filtrar por `solo_etiquetados`. Callarlo dejaba una
    # materia afuera sin una sola linea que lo dijera.
    logger.warning(
        "Sin columna 'PROMEDIO TRIMESTRAL': las filas de esta hoja no se van a "
        "poder etiquetar y quedaran fuera del entrenamiento."
    )
    return None


def _notas(fila: list, columnas: list[int]) -> list[float]:
    """Las notas presentes en esas columnas, en el orden de la planilla.

    Se descartan las celdas vacias en vez de convertirlas en cero: un criterio que
    el docente todavia no califico no es un aplazo.
    """
    valores = []
    for col in columnas:
        valor = _a_float(fila[col]) if col < len(fila) else None
        if valor is not None:
            valores.append(valor)
    return valores


@dataclass(frozen=True)
class NotasDeCurso:
    """SER y DECIDIR de un trimestre, que se califican una vez para todo el curso.

    `planificados` viaja aparte y no como una entrada mas del diccionario: es el
    numero de criterios que el docente definio para esas dos dimensiones, y el
    denominador de `progress_pct` lo necesita. Sin el, el numerador cuenta notas
    de las cuatro dimensiones contra un denominador que solo planifico dos, la
    fraccion pasa de 1 y `min(..., 100)` la deja clavada en 100 para casi todas
    las filas: la feature que existe para distinguir la semana 2 de la 10 deja de
    distinguir, y no falla ni una fila.
    """

    por_estudiante: dict[str, dict[str, list[float]]]
    # Por dimension, no un total: en 2026 SER vive en la hoja del area y DECIDIR
    # sigue viniendo de aca. Sumar el total cuando solo una cayo al curso vuelve a
    # inflar el denominador, que es el mismo error una rama mas alla.
    planificados: dict[str, int]


def _leer_hoja_curso(path: Path, trimestre: int) -> NotasDeCurso:
    """SER y DECIDIR por estudiante, desde la hoja del curso.

    Vacio si la hoja no existe, que es el caso de los templates donde SER vive en
    la hoja del area.
    """
    vacio = NotasDeCurso(por_estudiante={}, planificados=dict.fromkeys(DIMENSIONES_DE_CURSO, 0))
    hojas = {normalizar(h): h for h in list_sheets(path)}
    real = hojas.get(normalizar(HOJA_SER_DECIDIR))
    if real is None:
        # Legitimo en 2026, donde SER vive en la hoja del area. En 2023 y 2025 es
        # la unica fuente de SER y DECIDIR, y sin ella las filas igual se entrenan
        # con medio vector en `None`. Desde aca no se distingue un template del
        # otro, asi que se deja dicho en vez de devolver vacio callado.
        logger.info(
            "Sin hoja '%s' en T%d: SER y DECIDIR tendran que venir de las hojas de "
            "area. Hojas disponibles: %s",
            HOJA_SER_DECIDIR,
            trimestre,
            sorted(hojas.values()),
        )
        return vacio

    grid = read_sheet_grid(path, real)
    col_nombre = _columna_de_nombres(grid)
    fila_est = _fila_primer_estudiante(grid, col_nombre)
    if fila_est < 0:
        logger.warning("'%s' sin filas de estudiante en T%d", HOJA_SER_DECIDIR, trimestre)
        return vacio

    columnas = _columnas_por_dimension(_cabecera(grid, fila_est), DIMENSIONES_DE_CURSO)
    por_estudiante = {}
    for fila in grid[fila_est:]:
        if not fila or not _es_entero(fila[0]):
            continue
        nombre = _nombre_de(fila, col_nombre)
        if nombre is None:
            continue
        por_estudiante[normalizar(nombre)] = {
            dimension: _notas(fila, cols) for dimension, cols in columnas.items()
        }
    return NotasDeCurso(
        por_estudiante=por_estudiante,
        planificados=_declarados_por_dimension(columnas),
    )


def _leer_area(grid: list[list], area: str, del_curso: NotasDeCurso) -> list[dict]:
    """Las filas de una hoja de area, con sus cuatro listas de notas."""
    col_nombre = _columna_de_nombres(grid)
    fila_est = _fila_primer_estudiante(grid, col_nombre)
    if fila_est < 0:
        # Sin una sola fila reconocible el area desaparece del entrenamiento, y
        # callarlo la deja afuera sin mas sintoma que filas de menos.
        logger.warning(
            "Hoja de '%s' sin filas de estudiante reconocibles: se omite.",
            area,
        )
        return []

    cabecera = _cabecera(grid, fila_est)
    columnas = _columnas_por_dimension(cabecera, DIMENSIONES_DE_AREA)
    if not columnas["knowing"] and not columnas["doing"]:
        # Puede ser legitimo —un area tecnica que este docente no dicta viene sin
        # bloques— o puede ser un template que renombro el encabezado de SABER. Se
        # avisa porque desde aca no se distinguen, y callarlo hace que un formato
        # nuevo se lleve la hoja entera y solo se note como filas de menos.
        logger.warning(
            "Hoja de '%s' sin bloques de SABER ni HACER: se omite. Si el docente "
            "dicta esa area, el template cambio los encabezados.",
            area,
        )
        return []

    # SER y DECIDIR: de la propia hoja si las trae (2026), del curso si no.
    columnas.update(_columnas_por_dimension(cabecera, DIMENSIONES_DE_CURSO))
    col_trimestral = _columna_promedio_trimestral(cabecera)
    declarados_del_area = _declarados_por_dimension(columnas)

    filas = []
    sin_pareja = 0
    for fila in grid[fila_est:]:
        if not fila or not _es_entero(fila[0]):
            continue
        nombre = _nombre_de(fila, col_nombre)
        if nombre is None:
            continue

        notas = {d: _notas(fila, columnas[d]) for d in DIMENSION_POR_COLUMNA}
        if del_curso.por_estudiante and normalizar(nombre) not in del_curso.por_estudiante:
            sin_pareja += 1
        # El denominador del progreso se arma fila por fila y sigue a la fuente
        # que realmente aporto las notas. Contar las columnas del area cuando la
        # nota vino del curso mueve la fraccion sin que nada falle, y esa fraccion
        # es la feature que distingue la semana 2 de la 10.
        planificados = dict(declarados_del_area)
        del_estudiante = del_curso.por_estudiante.get(normalizar(nombre), {})
        for dimension in DIMENSIONES_DE_CURSO:
            if notas[dimension]:
                continue
            del_alumno = list(del_estudiante.get(dimension, []))
            if del_alumno:
                # Sustituye la fuente, y con ella el divisor.
                notas[dimension] = del_alumno
                planificados[dimension] = del_curso.planificados.get(dimension, 0)
            elif not columnas[dimension]:
                # Ni el area ni el curso la declararon: no hay criterios que contar.
                planificados[dimension] = 0
            # Si el area SI declaro columnas y el alumno todavia no tiene nota, el
            # divisor se queda con las del area. Ponerlo en cero encogia el
            # denominador, el progreso salia inflado, y volvia a bajar en cuanto
            # llegaba la primera nota: una feature que retrocede sola.

        if not notas["knowing"] and not notas["doing"]:
            # Fila sin una sola nota en el area: el alumno no cursa, o la planilla
            # esta en blanco. No aporta nada y ensuciaria el conteo.
            continue

        filas.append(
            {
                "nombre": nombre,
                "area": area,
                **notas,
                # Cuantos criterios definio el docente, calificados o no. Es lo que
                # permite distinguir tres notas de tres de tres notas de siete.
                "criterios_planificados": sum(planificados.values()),
                # El resultado del area en el trimestre. NO es una feature: es de donde
                # sale la etiqueta, y sale de un trimestre distinto al de las notas.
                "prom_area_trim": _a_float(fila[col_trimestral])
                if col_trimestral is not None and col_trimestral < len(fila)
                else None,
            }
        )

    if sin_pareja:
        # Se cuentan, no se nombran: son menores. `normalizar` arregla acentos y
        # mayusculas pero no un segundo nombre que aparece en una hoja y no en la
        # otra, y esos alumnos quedan sin SER ni DECIDIR sin que nada lo diga.
        logger.warning(
            "En '%s', %d estudiantes no se encontraron en la hoja de curso: se "
            "quedan sin SER ni DECIDIR.",
            area,
            sin_pareja,
        )
    return filas


def load_registro_trimestre(curso: CursoFolder, trimestre: int, cfg: AppConfig) -> pd.DataFrame:
    """Notas por criterio de un curso en un trimestre, una fila por estudiante y area.

    Columnas: gestion, grado, paralelo, trimestre, nombre, area,
              being, knowing, doing, deciding (listas), criterios_planificados.
    """
    patron = cfg["archivos"]["registro_trimestre"].format(trimestre=trimestre)
    path = find_file(curso.path, patron)
    if path is None:
        return pd.DataFrame()

    del_curso = _leer_hoja_curso(path, trimestre)
    hojas = {normalizar(h): h for h in list_sheets(path)}

    todas = []
    faltantes = []
    for area in cfg["areas"]:
        real = hojas.get(normalizar(area["hoja"]))
        if real is None:
            faltantes.append(area["hoja"])
            continue
        todas.extend(_leer_area(read_sheet_grid(path, real), area["nombre"], del_curso))

    if faltantes:
        # Una hoja renombrada se lleva la materia entera del entrenamiento, y el
        # unico sintoma serian filas de menos.
        logger.warning(
            "%d de %d areas configuradas no tienen hoja en %s T%d: %s",
            len(faltantes),
            len(cfg["areas"]),
            curso.label,
            trimestre,
            faltantes,
        )

    df = pd.DataFrame(todas)
    if df.empty:
        logger.warning("%s T%d: sin datos de areas", curso.label, trimestre)
        return df

    df["gestion"] = curso.gestion
    df["grado"] = curso.grado
    df["paralelo"] = curso.paralelo
    df["trimestre"] = trimestre
    logger.info(
        "Registro %s T%d: %d filas (estudiante x area), %d areas, %d notas por criterio",
        curso.label,
        trimestre,
        len(df),
        df["area"].nunique(),
        int(sum(df[d].map(len).sum() for d in DIMENSION_POR_COLUMNA)),
    )
    return df
