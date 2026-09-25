"""Contrato de features: cinco variables semanticas afuera, treinta columnas adentro.

QUE VE QUIEN LLAMA
------------------
Cinco entradas semanticas, por estudiante y por materia en un trimestre:

    being, knowing, doing, deciding   -> las notas de esa dimension (1 o muchas)
    attendance_pct                    -> porcentaje de asistencia del trimestre

Mas un sexto dato que no es una nota sino el contexto que las hace legibles:
`criterios_planificados`, de donde sale `progress_pct`. Un promedio de 40 en la
semana dos y uno de 40 en la semana diez significan cosas opuestas, y contar
notas no distingue tres de tres de tres de siete.

Las cuatro dimensiones son las de la RM 0001/2026 (Art. 29) y las que la BDD
hace cumplir en `evaluation_criteria.dimension`. Cuanto vale cada una depende de
la gestion: ver `Escala`. La autoevaluacion no es una dimension: es el
instrumento con que se califica Decidir, y por eso no aparece aca.

QUE VE EL MODELO
----------------
Un arbol de decision parte sobre un escalar: no existe forma de que una lista de
largo variable sea una feature. Pasar `nota_1, nota_2, ...` tampoco sirve, porque
fija N en tiempo de entrenamiento — el criterio que el docente agregue despues
cae fuera — y porque la posicion no significa nada: el `nota_3` de Matematicas y
el de Lenguaje son criterios distintos.

Lo que si conserva la informacion que el promedio tira son estadisticos por
dimension. Se calculan sobre cuantas notas haya, asi que **un criterio nuevo
cambia los VALORES de las features, nunca su cantidad**.

ORDEN CRONOLOGICO
-----------------
`trend` es la ultima nota menos la primera, de modo que las listas tienen que
llegar **ordenadas por fecha**. Sin esa garantia la feature es ruido.

FUENTE UNICA
------------
Esta es la unica construccion del vector. La usan `preprocessing/features.py` al
armar la matriz de entrenamiento, `serving/api.py` al responder una prediccion, y
`training/train.py` y `evaluation/evaluate.py` para saber que columnas esperar.

No es una formalidad: calcularlo de dos formas distintas le daria al modelo, en
produccion, features que nunca vio, y se degradaria **en silencio** — sin un solo
error, solo prediciendo peor. Hubo un tiempo en que convivio con otra definicion
de cinco columnas escalares en `preprocessing/features.py`; ya no existe.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# Nota de aprobacion del Ministerio, expresada sobre 100 para que signifique lo
# mismo en las cuatro dimensiones.
#
# El valor lo declara `config.yaml` (`nota_aprobacion: 51`) y este modulo lo lee de
# ahi: tenerlo escrito tambien aca eran dos fuentes para el mismo numero, y la
# feature `below` habria seguido contando contra 51 aunque el Ministerio lo
# cambiara en el YAML. La constante queda como respaldo para el uso sin config.
_NOTA_APROBACION_POR_DEFECTO = 51.0


@lru_cache(maxsize=1)
def nota_aprobacion_pct() -> float:
    """El umbral vigente, leido cuando se usa y no al importar.

    Leerlo a nivel de modulo hacia que importar este contrato —logica pura,
    testeable sin nada— abriera `config.yaml` y reconfigurara el logging de paso.

    Cacheado: `_estadisticos` lo llama una vez por dimension por fila, y con una
    config rota eso eran cuatro re-lecturas y cuatro warnings por fila —cientos de
    lineas iguales tapando la unica que importaba—, porque `lru_cache` no cachea
    excepciones y el fallback volvia a intentarlo cada vez.
    """
    try:
        from .config import get_config

        return float(get_config().nota_aprobacion)
    except Exception as e:  # noqa: BLE001 - sin config utilizable manda la RM vigente
        logger.warning(
            "Sin config utilizable (%s): la feature `below` cuenta contra %.1f, "
            "aunque el YAML diga otra cosa.",
            e,
            _NOTA_APROBACION_POR_DEFECTO,
        )
        return _NOTA_APROBACION_POR_DEFECTO


# Los nombres exactos que la BDD acepta en `evaluation_criteria.dimension`, en el
# orden del `total_score` generado. Cuanto vale cada uno NO vive aca: depende de la
# gestion, y `Escala` es el unico lugar que lo dice.
DIMENSIONES: tuple[str, ...] = ("being", "knowing", "doing", "deciding")

# Margen para comparar la suma de una escala contra 100: los pesos podrian no ser
# enteros y un `!= 100.0` sobre flotantes rechazaria un reparto valido.
_TOLERANCIA_SUMA = 1e-9


@dataclass(frozen=True)
class Escala:
    """La nota maxima que puede tener cada dimension, tal como el loader la lee.

    **No es el reparto de 100 puntos de la RM: es el tope del bloque.** La
    diferencia importa. El loader lee las columnas de criterio de SER, SABER,
    HACER y DECIDIR, y la autoevaluacion vive en una columna aparte que no lee.
    Poniendo aca el tope con la autoevaluacion ya sumada, un SER perfecto de 2023
    —10 sobre un bloque de 10— se normalizaba contra 15 y salia 67 por ciento. Sin
    error, sin fila rechazada: todas las notas de esa dimension corridas hacia
    abajo, que es la forma de mentir que este contrato existe para evitar.

    Por eso los topes historicos no suman 100: lo que falta es exactamente la
    autoevaluacion que no se lee. El detalle por año esta en
    `ESCALAS_POR_GESTION`, que es la unica fuente — no lo repitas en otro
    docstring, porque la copia envejece, y la que envejecio decia que 2024 usaba
    Saber 35.
    """

    being: float
    knowing: float
    doing: float
    deciding: float

    def __post_init__(self) -> None:
        if self.total() - 100.0 > _TOLERANCIA_SUMA:
            # Puede sumar menos de 100 —lo que falta es la autoevaluacion, que se
            # califica aparte y no se lee— pero nunca mas: eso seria un tope
            # inventado, y normalizar contra el achataria todas las notas.
            raise ValueError(f"Los topes no pueden pasar de 100, suman {self.total()}")
        for dimension in DIMENSIONES:
            if self.tope_de(dimension) <= 0:
                # Sin esto un tope en cero pasa las dos guardas de `_a_porcentaje`
                # (una nota 0 no es negativa ni supera el tope) y revienta recien
                # al dividir, con un ZeroDivisionError que no dice cual dimension.
                raise ValueError(
                    f"La escala da {self.tope_de(dimension)} puntos a {dimension}: "
                    "una dimension que no vale nada no es una dimension"
                )

    def total(self) -> float:
        return self.being + self.knowing + self.doing + self.deciding

    def tope_de(self, dimension: str) -> float:
        return getattr(self, dimension)


# La del sistema y la RM 0001/2026: es la escala a la que se lleva todo lo demas.
#
# Sale de `config.yaml:ponderacion`, que declaraba los cuatro pesos y no la leia
# nadie: editar `saber: 45` ahi no cambiaba una sola feature. La autoevaluacion del
# yaml es el casillero que la BDD llama `deciding`.
_ESCALA_POR_DEFECTO = Escala(being=10.0, knowing=45.0, doing=40.0, deciding=5.0)


@lru_cache(maxsize=1)
def escala_vigente() -> Escala:
    """La escala del sistema, leida cuando se usa y no al importar.

    Igual que `nota_aprobacion_pct`, y por el mismo motivo: resolverla a nivel de
    modulo hacia que importar este contrato —logica pura, testeable sin nada—
    abriera `config.yaml` y reconfigurara el logging del proceso que lo importo.
    Ademas congelaba la escala en el default de `expandir`, que asi no se puede
    inyectar en un test.
    """
    try:
        from .config import get_config

        p = get_config()["ponderacion"]
        return Escala(
            being=float(p["ser"]),
            knowing=float(p["saber"]),
            doing=float(p["hacer"]),
            deciding=float(p["autoevaluacion"]),
        )
    except Exception as e:  # noqa: BLE001 - sin config utilizable manda la RM vigente
        logger.warning(
            "Sin ponderacion utilizable en config.yaml (%s): se usa la de la RM "
            "0001/2026 (10/45/40/5).",
            e,
        )
        return _ESCALA_POR_DEFECTO


# Lo que decia el encabezado de cada registro historico.
#
# Saber y Hacer se leen literales del encabezado y no admiten discusion. Ser y
# Decidir si, porque la autoevaluacion figura como columna aparte y hay que
# decidir a que dimension se le suma:
#
# Son los topes **de los bloques que el loader lee**, no el reparto de 100 de la
# RM. La autoevaluacion esta en columnas aparte que no se leen, asi que su puntaje
# no entra aca: sumarlo achataba las notas de SER y DECIDIR contra un tope que
# nadie podia alcanzar.
#
#   2023: 'SER - 10', 'SABER - 35', 'HACER - 35', 'DECIDIR - 10'. Suman 90; los
#     10 que faltan son 'AUTOEVALUACION - SER 5' y 'AUTOEVALUACION - DECIDIR 5'.
#   2024 y 2025: 'SER - 5 Puntos' y 'DECIDIR - 5 Puntos' en la hoja EVAL, mas
#     'SABER - 45' y 'HACER - 40'. Suman 95; los 5 que faltan son la columna
#     'AUTOEVALUACION - SER Y DECIDIR'.
#
# 2024 lleva la escala nueva, no la de 2023, aunque sea el año del medio. Se
# midio en la carpeta que **tiene datos** (`...Montaño Nogales++`, 3304 notas):
# dice 'SABER - 45', 'HACER - 40' y una sola 'AUTOEVALUACION - SER Y DECIDIR'. La
# carpeta vacia del mismo curso conserva el template viejo con 'SABER - 35', y
# leer la ponderacion de ahi hacia rebotar notas legitimas de 37 sobre un tope de
# 35. La leccion esta anotada: la escala se mide donde estan las notas.
# Desde esta gestion el colegio carga en el sistema y rige la ponderacion vigente.
_PRIMERA_GESTION_DEL_SISTEMA = 2026

ESCALAS_POR_GESTION: dict[int, Escala] = {
    2023: Escala(being=10.0, knowing=35.0, doing=35.0, deciding=10.0),
    2024: Escala(being=5.0, knowing=45.0, doing=40.0, deciding=5.0),
    2025: Escala(being=5.0, knowing=45.0, doing=40.0, deciding=5.0),
    # 2026 no esta aca: es la vigente, y resolverla al construir el diccionario
    # volveria a leer la config al importar. La resuelve `escala_de`.
}


def escala_de(gestion: int | None) -> Escala:
    """La escala de esa gestion, o la vigente si no se conoce.

    Sin gestion declarada es el caso del sistema, y la vigente es la respuesta
    correcta. Una gestion que si viene pero no esta mapeada es otra cosa: sale una
    planilla historica normalizada contra los topes de hoy, sin que falle una sola
    fila. Por eso se avisa.
    """
    if gestion is None:
        return escala_vigente()
    escala = ESCALAS_POR_GESTION.get(gestion)
    if escala is None and gestion >= _PRIMERA_GESTION_DEL_SISTEMA:
        return escala_vigente()
    if escala is None:
        logger.warning(
            "Gestion %s sin escala mapeada: se normaliza contra la vigente y las "
            "notas de esa planilla van a leerse corridas. Agregar su ponderacion "
            "a ESCALAS_POR_GESTION.",
            gestion,
        )
        return escala_vigente()
    return escala


# Los siete estadisticos por dimension. El orden fija el de FEATURE_COLS.
_ESTADISTICOS = ("mean", "min", "max", "std", "count", "below", "trend")

FEATURE_COLS: list[str] = [f"{d}_{e}" for d in DIMENSIONES for e in _ESTADISTICOS] + [
    "attendance_pct",
    "progress_pct",
]


@dataclass
class ObservacionMateria:
    """Lo que se sabe de un estudiante en una materia, hoy.

    Las cuatro listas llevan las notas ya registradas por el docente, en su
    escala original (0..tope de la dimension) y ordenadas por fecha.

    `criterios_planificados` es cuantos criterios definio el docente para la
    materia en ese trimestre. Sin ese numero no hay forma de distinguir tres
    notas de tres de tres notas de siete, que significan cosas opuestas.
    """

    being: list[float] = field(default_factory=list)
    knowing: list[float] = field(default_factory=list)
    doing: list[float] = field(default_factory=list)
    deciding: list[float] = field(default_factory=list)
    attendance_pct: float | None = None
    criterios_planificados: int | None = None

    def notas_de(self, dimension: str) -> list[float]:
        return getattr(self, dimension)


def puede_predecir(obs: ObservacionMateria) -> bool:
    """Si ya hay al menos una nota en cada una de las cuatro dimensiones.

    Es la regla que pidio el docente: antes de eso el estudiante no tiene
    todavia una materia que describir, y una prediccion sobre dos dimensiones
    diria mas sobre lo que falta cargar que sobre el estudiante.
    """
    return all(obs.notas_de(d) for d in DIMENSIONES)


def _a_porcentaje(notas: list[float], dim: str, tope: float) -> list[float]:
    """Cada nota como porcentaje del tope que regia cuando se puso.

    Un 5 de Being es la mitad de la dimension y un 5 de Knowing es un noveno. Y un
    35 de Saber es la nota perfecta de 2023 pero un 78 por ciento hoy. El
    porcentaje es lo unico que significa lo mismo entre dimensiones y entre
    gestiones, y es lo que deja entrenar con las dos cosas juntas.
    """
    for n in notas:
        if n < 0:
            raise ValueError(f"{dim}: nota negativa ({n})")
        if n > tope:
            raise ValueError(f"{dim}: nota {n} sobre el tope {tope}")
    return [n / tope * 100.0 for n in notas]


def _estadisticos(pct: list[float]) -> dict[str, float | None]:
    """Los siete numeros que resumen una dimension sin importar cuantas notas trae."""
    aprobacion = nota_aprobacion_pct()
    if not pct:
        # Dimension sin calificar: falta el dato, y `count` es el unico que se
        # sabe de verdad. Cero en los demas seria afirmar notas malas.
        return dict.fromkeys(("mean", "min", "max", "std", "below", "trend")) | {"count": 0.0}
    return {
        "mean": statistics.fmean(pct),
        "min": min(pct),
        "max": max(pct),
        # Poblacional, no muestral: una sola nota tiene dispersion cero, no
        # indefinida. `stdev` explotaria con n=1.
        "std": statistics.pstdev(pct),
        # Contadores, pero en punto flotante como todo lo demas. El tipo de cada
        # columna tiene que ser el mismo al entrenar y al predecir: si una queda
        # entera porque en el entrenamiento nunca le faltó un valor, el SavedModel
        # la fija como int64 y en inferencia rechaza la firma con un
        # "Could not find matching concrete function" que no explica nada.
        "count": float(len(pct)),
        "below": float(sum(1 for p in pct if p < aprobacion)),
        # None con una sola nota, no cero. La dispersion de un punto SI es cero,
        # pero su pendiente no existe, y un cero ahi se lee como "se midio y no
        # cambio" — la misma confusion entre faltante y medido que dejo una
        # dimension entera en cero durante tres gestiones.
        "trend": pct[-1] - pct[0] if len(pct) > 1 else None,
    }


def _progreso(obs: ObservacionMateria) -> float | None:
    """Cuanto del trimestre ya ocurrio, no cuantas notas hay.

    Un promedio de 40 en la semana 2 y uno de 40 en la semana 10 significan
    cosas opuestas, y sin esto el modelo no puede distinguirlos: describiria el
    presente en vez de anticipar el final.
    """
    planificados = obs.criterios_planificados
    # `!= planificados` atrapa NaN, que es lo que deja un merge sin pareja y que
    # `<= 0` no ve: comparar NaN siempre da False y la division seguiria adelante.
    if planificados is None or planificados != planificados or planificados <= 0:
        return None
    calificados = sum(len(obs.notas_de(d)) for d in DIMENSIONES)
    return min(calificados / planificados * 100.0, 100.0)


def filtrar_en_rango(
    obs: ObservacionMateria, escala: Escala
) -> tuple[ObservacionMateria, list[tuple[str, float]]]:
    """La observacion sin las notas imposibles, y la lista de las que se fueron.

    Vive aca y no en la ingesta a proposito. Descartar una nota cambia `count`,
    `mean`, `std`, `below` y `trend`: es una transformacion del vector, y un vector
    transformado fuera del contrato es la misma fuga que tener dos contratos. El
    entrenamiento filtra con esta funcion y la inferencia valida con `expandir`,
    pero el criterio de que entra y que no esta escrito una sola vez.

    Se descarta en vez de recortar: recortar inventa exactamente el tope, que es
    una nota que nadie puso.
    """
    limpias: dict[str, list[float]] = {}
    descartadas: list[tuple[str, float]] = []
    for dimension in DIMENSIONES:
        tope = escala.tope_de(dimension)
        adentro = []
        for nota in obs.notas_de(dimension):
            if 0 <= nota <= tope:
                adentro.append(float(nota))
            else:
                descartadas.append((dimension, float(nota)))
        limpias[dimension] = adentro

    return (
        ObservacionMateria(
            **limpias,
            attendance_pct=obs.attendance_pct,
            criterios_planificados=obs.criterios_planificados,
        ),
        descartadas,
    )


def expandir(obs: ObservacionMateria, escala: Escala | None = None) -> dict[str, float | None]:
    """La observacion como el vector que entra al modelo.

    `escala` es la ponderacion que regia cuando se pusieron esas notas: por
    defecto la del sistema, y `escala_de(gestion)` para una planilla historica.

    Devuelve siempre las mismas claves, en el mismo orden, con `None` donde el
    dato falta. `None` y cero no son lo mismo: TF-DF trata el faltante como tal,
    y un cero seria una nota mala inventada.
    """
    escala = escala or escala_vigente()
    vector: dict[str, float | None] = {}
    for dim in DIMENSIONES:
        stats = _estadisticos(_a_porcentaje(obs.notas_de(dim), dim, escala.tope_de(dim)))
        for nombre in _ESTADISTICOS:
            vector[f"{dim}_{nombre}"] = stats[nombre]
    vector["attendance_pct"] = obs.attendance_pct
    vector["progress_pct"] = _progreso(obs)
    return vector
