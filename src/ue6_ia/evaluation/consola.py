"""Las tablas que los comandos imprimen al terminar, para leerlas y para capturarlas.

POR QUE UN MODULO APARTE
------------------------
`build-dataset` y `train` dejaban su resultado repartido en lineas de log, una por
curso o por pliegue. Para el documento hace falta un bloque final que se lea de un
vistazo y quepa en una captura de 100 columnas. Ese bloque no es diagnostico —es
la salida del comando—, asi que va a la consola y no a `logging`, y vive aca para
que el pipeline y el entrenamiento sigan sin imprimir nada.

Cada tabla se arma primero como datos (`Tabla`) y despues se dibuja: con `rich`
si esta instalado, en texto alineado si no. Los tests miran los datos y el texto
plano; ninguno depende de como pinta `rich`.

Ninguna tabla lleva nombres de estudiantes: se cuentan filas, no personas.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..contract import DIMENSIONES
from .graficos import (
    METADATA,
    REGISTRO_ENTRENAMIENTO,
    REPORTE_VALIDACION,
    candidato_elegido,
    etiqueta_de_clase,
    leer_predicciones,
    matriz_confusion_acumulada,
    metricas_por_clase,
    ordenar_por_severidad,
)
from .metricas import promedios_por_clase, resumen_clasificacion

if TYPE_CHECKING:  # solo para la firma: importar el pipeline arrastra la ingesta
    from ..pipeline import ResumenDataset

logger = logging.getLogger(__name__)

ANCHO_MAXIMO = 100

# Los nombres de las dimensiones tal como los dice la escuela. En el vector van en
# ingles porque son los valores de la BDD; en una captura para el documento, no.
NOMBRE_DIMENSION = {"being": "Ser", "knowing": "Saber", "doing": "Hacer", "deciding": "Decidir"}


@dataclass
class Tabla:
    """Una tabla como datos: titulo, encabezados, filas y que columnas son numeros."""

    titulo: str
    encabezados: list[str]
    filas: list[list[str]] = field(default_factory=list)
    numericas: set[int] = field(default_factory=set)
    # Indices de fila antes de los cuales va una linea separadora.
    secciones: set[int] = field(default_factory=set)
    nota: str | None = None


def _num(valor: float | None, decimales: int = 3) -> str:
    return "n/d" if valor is None else f"{valor:.{decimales}f}"


def _pct(parte: int, total: int) -> str:
    return "n/d" if not total else f"{100 * parte / total:.1f}%"


# --------------------------------------------------------------------------
# build-dataset
# --------------------------------------------------------------------------


def tablas_resumen_dataset(resumen: ResumenDataset) -> list[Tabla]:
    """El resumen del procesamiento, en el orden en que ocurrio."""
    fuentes = Tabla(
        "Fuentes leídas",
        ["Concepto", "Cantidad"],
        [
            ["Gestiones procesadas", ", ".join(str(g) for g in resumen.gestiones)],
            ["Carpetas de curso encontradas", str(resumen.carpetas_encontradas)],
            ["Copias duplicadas descartadas", str(resumen.duplicados_descartados)],
            ["Cursos procesados", str(resumen.cursos_procesados)],
            ["Cursos que aportaron filas", str(resumen.cursos_con_filas)],
            ["Registros trimestrales leídos", str(resumen.trimestres_leidos)],
        ],
        numericas={1},
    )

    limpieza = Tabla("Limpieza y etiquetado", ["Concepto", "Cantidad"], numericas={1})
    limpieza.filas.append(
        ["Filas de features (estudiante x materia x trimestre)", str(resumen.filas_features)]
    )
    # Las cuatro dimensiones siempre, aunque alguna no haya descartado nada: el
    # cero dice que se revisaron sus notas y estaban todas en rango.
    for dimension in DIMENSIONES:
        n = resumen.notas_fuera_de_rango.get(dimension, 0)
        limpieza.filas.append(
            [f"Notas fuera de rango descartadas: {NOMBRE_DIMENSION[dimension]}", str(n)]
        )
    limpieza.filas.append(
        ["Filas con etiqueta del trimestre siguiente", str(resumen.filas_con_etiqueta)]
    )
    for trimestre, n in sorted(resumen.filas_sin_etiqueta_por_trimestre.items()):
        motivo = "no hay trimestre siguiente" if trimestre == 3 else "sin el trimestre siguiente"
        limpieza.filas.append([f"Filas sin etiqueta, trimestre {trimestre} ({motivo})", str(n)])
    limpieza.filas.append(
        ["Filas descartadas por no tener etiqueta", str(resumen.filas_descartadas_sin_etiqueta)]
    )

    final = Tabla(
        "Dataset final",
        ["Concepto", "Cantidad"],
        [
            ["Filas", str(resumen.filas_finales)],
            ["Estudiantes distintos", str(resumen.estudiantes)],
            ["Materias", str(resumen.materias)],
        ],
        numericas={1},
    )

    total = resumen.filas_finales
    por_gestion = Tabla(
        "Filas por gestión y trimestre",
        ["Corte", "Filas", "%"],
        [[f"Gestión {g}", str(n), _pct(n, total)] for g, n in resumen.filas_por_gestion.items()],
        numericas={1, 2},
    )
    por_gestion.secciones.add(len(por_gestion.filas))
    por_gestion.filas.extend(
        [f"Trimestre {t}", str(n), _pct(n, total)] for t, n in resumen.filas_por_trimestre.items()
    )

    clases = Tabla(
        "Distribución de clases",
        ["Clase", "Filas", "%"],
        [
            [
                etiqueta_de_clase(c),
                str(resumen.distribucion_clases[c]),
                _pct(resumen.distribucion_clases[c], total),
            ]
            for c in ordenar_por_severidad(list(resumen.distribucion_clases))
        ],
        numericas={1, 2},
        nota="La clase es el riesgo de la misma materia en el trimestre siguiente.",
    )
    return [fuentes, limpieza, final, por_gestion, clases]


def lineas_resumen_dataset(resumen: ResumenDataset) -> list[str]:
    """El resumen en texto plano, el mismo que se imprime sin `rich`."""
    return _lineas_planas(tablas_resumen_dataset(resumen))


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def tabla_pliegues(modelo: str, bloque: dict) -> Tabla:
    """Una fila por pliegue: con cuanto se entreno, hasta donde llego y que midio."""
    pliegues = bloque.get("pliegues") or []
    particiones = bloque.get("particiones") or [{} for _ in pliegues]
    tabla = Tabla(
        f"Validación cruzada agrupada por estudiante: {modelo}",
        [
            "Pliegue",
            "Filas e/p/v",
            "Estud. e/p/v",
            "Iter.",
            "Accuracy",
            "Base",
            "Macro F1",
            "Rec. crítico",
        ],
        numericas={3, 4, 5, 6, 7},
        nota="e = entrenamiento, p = parada temprana, v = validación (estudiantes no vistos). "
        "Iter. = iteraciones entrenadas (entre paréntesis, la de mejor pérdida de validación).",
    )
    for i, (medidas, particion) in enumerate(zip(pliegues, particiones, strict=False), start=1):
        iteraciones = particion.get("iteraciones") or {}
        entrenadas = iteraciones.get("iteraciones_entrenadas")
        mejor = iteraciones.get("iteracion_mejor_validacion")
        if entrenadas is None:
            texto_iter = "n/d"
        elif mejor is None or mejor == entrenadas:
            texto_iter = str(entrenadas)
        else:
            texto_iter = f"{entrenadas} ({mejor})"
        tabla.filas.append(
            [
                f"P{i}",
                _tripleta(particion, "filas"),
                _tripleta(particion, "estudiantes"),
                texto_iter,
                _num(medidas.get("accuracy")),
                _num(medidas.get("linea_base")),
                _num(medidas.get("macro_f1")),
                _num(medidas.get("recall_riesgo_critico")),
            ]
        )

    resumen = bloque.get("resumen") or {}
    tabla.secciones.add(len(tabla.filas))
    for nombre, clave in (("Media", "media"), ("Desvío", "desvio")):
        tabla.filas.append(
            [
                "",
                "",
                "",
                nombre,
                *(_num((resumen.get(m) or {}).get(clave)) for m in _METRICAS_PLIEGUE),
            ]
        )
    # Un pliegue que no trajo casos de la clase no midio: la media es de los que si.
    parciales = [
        f"{m} sobre {r['pliegues']} de {len(pliegues)} pliegues"
        for m in _METRICAS_PLIEGUE
        if (r := resumen.get(m)) and r.get("pliegues", len(pliegues)) < len(pliegues)
    ]
    if parciales:
        tabla.nota += " Media y desvío de " + "; ".join(parciales) + " (los demás no midieron)."
    return tabla


_METRICAS_PLIEGUE = ("accuracy", "linea_base", "macro_f1", "recall_riesgo_critico")


def _tripleta(particion: dict, que: str) -> str:
    valores = [particion.get(f"{que}_{lado}") for lado in ("entrenamiento", "parada", "validacion")]
    if all(v is None for v in valores):
        return "n/d"
    return "/".join("-" if v is None else str(v) for v in valores)


def tabla_seleccion(comparacion: dict, seleccion: dict) -> Tabla:
    """Por que gano el que gano: cada candidato con su macro F1, recall y estado.

    Lee la `seleccion` que guardo el entrenamiento, que es la misma regla que
    eligio; no la vuelve a aplicar.
    """
    adentro = {c["candidato"] for c in seleccion.get("dentro_de_la_banda") or []}
    motivos = {c["candidato"]: c for c in seleccion.get("excluidos") or []}
    mejor = seleccion.get("mejor_macro_f1") or {}
    banda = seleccion.get("banda") or {}
    tabla = Tabla(
        "Selección del modelo",
        ["Candidato", "Macro F1", "Rec. crítico", "Estado"],
        numericas={1, 2},
    )
    for clave, bloque in comparacion.items():
        resumen = bloque.get("resumen") or {}
        macro = resumen.get("macro_f1") or {}
        recall = resumen.get("recall_riesgo_critico") or {}
        if clave == seleccion.get("ganador"):
            estado = "Ganador"
        elif clave in adentro:
            estado = "En la banda"
        elif clave in motivos:
            # Sin uno de los dos macro F1 no hay distancia que mostrar: inventarla
            # con un cero se leeria como medida.
            tope = mejor.get("valor")
            propio = motivos[clave].get("macro_f1")
            if tope is None or propio is None:
                estado = "Fuera de la banda (n/d)"
            else:
                estado = f"Fuera de la banda (-{tope - propio:.3f})"
        else:
            estado = "n/d"
        texto_recall = _num(recall.get("media"))
        if recall.get("pliegues") is not None and recall.get("pliegues") < len(
            bloque.get("pliegues") or []
        ):
            texto_recall += f" ({recall['pliegues']}/{len(bloque['pliegues'])})"
        tabla.filas.append(
            [
                clave,
                f"{_num(macro.get('media'))} ± {_num(macro.get('desvio'))}",
                texto_recall,
                estado,
            ]
        )
    tabla.nota = (
        f"Mejor macro F1: {_num(mejor.get('valor'))} ({mejor.get('candidato', 'n/d')}). "
        f"Banda de empate: {_num(banda.get('ancho'))}, el mayor desvío de macro F1 "
        f"({banda.get('candidato', 'n/d')}). Dentro de la banda decide el mayor recall medio "
        f"de Riesgo crítico. Ganador: {seleccion.get('ganador', 'n/d')} — "
        f"{seleccion.get('criterio', '')}. Rec. crítico (k/n): medido en k de n pliegues."
    )
    return tabla


def tabla_particion_final(modelo_final: dict) -> Tabla:
    """Con cuantas filas y estudiantes se ajusto el modelo que se guarda."""
    particion = modelo_final.get("particion") or {}
    iteraciones = modelo_final.get("iteraciones") or {}
    tabla = Tabla(
        f"Modelo final: {modelo_final.get('modelo', '?')}",
        ["Conjunto", "Filas", "% filas", "Estudiantes", "% estud."],
        numericas={1, 2, 3, 4},
    )
    for lado, nombre in (("entrenamiento", "Entrenamiento"), ("parada", "Parada temprana")):
        filas = particion.get(f"filas_{lado}")
        # Un modelo sin parada temprana (Random Forest) no tiene ese conjunto: una
        # fila en cero se leeria como una parada vacia.
        if filas is None or (lado == "parada" and not filas):
            continue
        tabla.filas.append(
            [
                nombre,
                str(filas),
                _num(particion.get(f"pct_filas_{lado}"), 1) + "%",
                str(particion.get(f"estudiantes_{lado}")),
                _num(particion.get(f"pct_estudiantes_{lado}"), 1) + "%",
            ]
        )
    entrenadas = iteraciones.get("iteraciones_entrenadas")
    if entrenadas is not None:
        tabla.nota = (
            f"Iteraciones entrenadas: {entrenadas}; mejor iteración de validación: "
            f"{iteraciones.get('iteracion_mejor_validacion', 'n/d')}; árboles del modelo: "
            f"{iteraciones.get('arboles_del_modelo', 'n/d')}."
        )
    else:
        tabla.nota = f"Árboles del modelo: {iteraciones.get('arboles_del_modelo', 'n/d')}."
    pesos = modelo_final.get("pesos_clases")
    if pesos:
        tabla.nota += " Pesos por clase (balanceada): " + ", ".join(
            f"{etiqueta_de_clase(c)} {pesos[c]:.2f}" for c in ordenar_por_severidad(list(pesos))
        )
        tabla.nota += "."
    return tabla


def tabla_por_clase(medidas: dict, clases: list[str], origen: str) -> Tabla:
    """Precision, recall, F1 y soporte de cada clase, con los dos promedios abajo."""
    tabla = Tabla(
        f"Reporte por clase fuera de muestra ({origen})",
        ["Clase", "Precisión", "Recall", "F1", "Soporte"],
        numericas={1, 2, 3, 4},
    )
    for clase in ordenar_por_severidad(clases):
        d = medidas["por_clase"][clase]
        tabla.filas.append(
            [
                etiqueta_de_clase(clase),
                _num(d["precision"]),
                _num(d["recall"]),
                _num(d["f1"]),
                str(d["soporte"]),
            ]
        )
    promedios = promedios_por_clase(medidas["por_clase"])
    tabla.secciones.add(len(tabla.filas))
    tabla.filas.append(["Accuracy", "", "", _num(medidas["accuracy"]), str(medidas["n"])])
    for clave, nombre in (("macro", "Promedio macro"), ("ponderado", "Promedio ponderado")):
        p = promedios[clave]
        tabla.filas.append(
            [nombre, _num(p["precision"]), _num(p["recall"]), _num(p["f1"]), str(p["soporte"])]
        )
    tabla.nota = f"Línea base (clase más frecuente): {_num(medidas['linea_base'])}"
    return tabla


def tablas_de_entrenamiento(modelo_dir: Path) -> list[Tabla]:
    """Las tablas del entrenamiento, leidas de lo que `train` dejo en disco.

    Se leen de los archivos y no de la memoria del entrenamiento para que la misma
    salida se pueda volver a imprimir —en Windows, sin TensorFlow— con
    `cli metricas`, y para que la captura muestre exactamente lo que se guardo.
    """
    reporte = _leer(modelo_dir / REPORTE_VALIDACION)
    metadata = _leer(modelo_dir / METADATA)
    registro = _leer(modelo_dir / REGISTRO_ENTRENAMIENTO, obligatorio=False)
    if reporte is None or metadata is None:
        raise FileNotFoundError(
            f"Faltan los reportes del entrenamiento en {modelo_dir}. Corre primero: train"
        )

    tablas = [tabla_pliegues(nombre, bloque) for nombre, bloque in reporte.items()]
    if metadata.get("seleccion"):
        tablas.append(tabla_seleccion(reporte, metadata["seleccion"]))
    if registro and registro.get("modelo_final"):
        tablas.append(tabla_particion_final(registro["modelo_final"]))

    elegido = candidato_elegido(metadata)
    bloque = reporte.get(elegido) or next(iter(reporte.values()), {})
    clases = list(bloque.get("clases") or [])
    predicciones = leer_predicciones(modelo_dir, elegido) if elegido else None
    if predicciones is not None and clases:
        medidas = resumen_clasificacion(
            predicciones["real"].tolist(), predicciones["predicha"].tolist(), clases
        )
        origen = f"{elegido}, predicciones de los pliegues"
    elif clases:
        # Sin el parquet, la suma de las matrices de los pliegues dice lo mismo:
        # cada celda cuenta las mismas predicciones.
        medidas = metricas_por_clase(matriz_confusion_acumulada(bloque), clases)
        origen = f"{elegido}, suma de las matrices de los pliegues"
    else:
        return tablas
    tablas.append(tabla_por_clase(medidas, clases, origen))
    return tablas


def _leer(path: Path, obligatorio: bool = True) -> dict | None:
    if not path.exists():
        if not obligatorio:
            logger.warning("Falta %s: se omite su tabla. Reentrena para generarlo.", path.name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------


def imprimir(tablas: list[Tabla], titulo: str | None = None) -> None:
    """Imprime las tablas con `rich` si esta, y si no en texto alineado."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ModuleNotFoundError:
        if titulo:
            print(f"\n{titulo}\n{'=' * len(titulo)}")  # es la salida del comando, no un rastro
        print("\n".join(_lineas_planas(tablas)))  # es la salida del comando, no un rastro
        return

    consola = Console(width=min(Console().width, ANCHO_MAXIMO))
    if titulo:
        consola.rule(f"[bold]{titulo}")
    for tabla in tablas:
        dibujo = Table(
            title=tabla.titulo,
            title_justify="left",
            title_style="bold",
            box=box.SIMPLE_HEAD,
            caption=tabla.nota,
            caption_justify="left",
        )
        for k, encabezado in enumerate(tabla.encabezados):
            dibujo.add_column(encabezado, justify="right" if k in tabla.numericas else "left")
        for k, fila in enumerate(tabla.filas):
            if k in tabla.secciones and k:
                dibujo.add_section()
            dibujo.add_row(*fila)
        consola.print(dibujo)
        consola.print()


def _lineas_planas(tablas: list[Tabla]) -> list[str]:
    lineas: list[str] = []
    for tabla in tablas:
        anchos = [
            max([len(tabla.encabezados[k]), *(len(f[k]) for f in tabla.filas)])
            for k in range(len(tabla.encabezados))
        ]

        def renglon(celdas: list[str], tabla: Tabla = tabla, anchos: list[int] = anchos) -> str:
            return "  ".join(
                c.rjust(anchos[k]) if k in tabla.numericas else c.ljust(anchos[k])
                for k, c in enumerate(celdas)
            ).rstrip()

        lineas.append("")
        lineas.append(tabla.titulo)
        lineas.append(renglon(tabla.encabezados))
        lineas.append("-" * min(sum(anchos) + 2 * (len(anchos) - 1), ANCHO_MAXIMO))
        for k, fila in enumerate(tabla.filas):
            if k in tabla.secciones and k:
                lineas.append("-" * min(sum(anchos) + 2 * (len(anchos) - 1), ANCHO_MAXIMO))
            lineas.append(renglon(fila))
        if tabla.nota:
            lineas.extend(textwrap.wrap(tabla.nota, ANCHO_MAXIMO))
    return lineas
