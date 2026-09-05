"""Contrato de features: cinco variables afuera, treinta columnas adentro.

QUE VE QUIEN LLAMA
------------------
Cinco entradas semanticas, por estudiante y por materia en un trimestre:

    being, knowing, doing, deciding   -> las notas de esa dimension (1 o muchas)
    attendance_pct                    -> porcentaje de asistencia del trimestre

Las cuatro dimensiones son las de la RM 0001/2026 (Art. 29) y las que la BDD
hace cumplir en `evaluation_criteria.dimension`: Being 10, Knowing 45, Doing 40,
Deciding 5. Suman 100. La autoevaluacion no es una dimension: es el instrumento
con que se califica Decidir, y por eso no aparece aca.

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
Esta es la unica funcion que construye el vector: la usan el pipeline de
entrenamiento y `serving/api.py`. Calcularlo de dos formas distintas le daria al
modelo, en produccion, features que nunca vio — y se degradaria en silencio, sin
un solo error, solo prediciendo peor.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# Nota de aprobacion del Ministerio, expresada sobre 100 para que signifique lo
# mismo en las cuatro dimensiones.
NOTA_APROBACION_PCT = 51.0


@dataclass(frozen=True)
class Dimension:
    nombre: str
    tope: float


# El orden es el de la RM y el del `total_score` generado en la BDD.
DIMENSIONES: tuple[Dimension, ...] = (
    Dimension("being", 10.0),
    Dimension("knowing", 45.0),
    Dimension("doing", 40.0),
    Dimension("deciding", 5.0),
)

# Los siete estadisticos por dimension. El orden fija el de FEATURE_COLS.
_ESTADISTICOS = ("mean", "min", "max", "std", "count", "below", "trend")

FEATURE_COLS: list[str] = [
    f"{d.nombre}_{e}" for d in DIMENSIONES for e in _ESTADISTICOS
] + ["attendance_pct", "progress_pct"]


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

    def notas_de(self, dimension: Dimension) -> list[float]:
        return getattr(self, dimension.nombre)


def puede_predecir(obs: ObservacionMateria) -> bool:
    """Si ya hay al menos una nota en cada una de las cuatro dimensiones.

    Es la regla que pidio el docente: antes de eso el estudiante no tiene
    todavia una materia que describir, y una prediccion sobre dos dimensiones
    diria mas sobre lo que falta cargar que sobre el estudiante.
    """
    return all(obs.notas_de(d) for d in DIMENSIONES)


def _a_porcentaje(notas: list[float], dim: Dimension) -> list[float]:
    """Cada nota como porcentaje de su propio tope.

    Un 5 de Being es la mitad de la dimension y un 5 de Knowing es un noveno.
    Sin normalizar, el modelo los ve como el mismo numero.
    """
    for n in notas:
        if n < 0:
            raise ValueError(f"{dim.nombre}: nota negativa ({n})")
        if n > dim.tope:
            raise ValueError(f"{dim.nombre}: nota {n} sobre el tope {dim.tope}")
    return [n / dim.tope * 100.0 for n in notas]


def _estadisticos(pct: list[float]) -> dict[str, float | None]:
    """Los siete numeros que resumen una dimension sin importar cuantas notas trae."""
    if not pct:
        # Dimension sin calificar: falta el dato, y `count` es el unico que se
        # sabe de verdad. Cero en los demas seria afirmar notas malas.
        return dict.fromkeys(("mean", "min", "max", "std", "below", "trend")) | {"count": 0}
    return {
        "mean": statistics.fmean(pct),
        "min": min(pct),
        "max": max(pct),
        # Poblacional, no muestral: una sola nota tiene dispersion cero, no
        # indefinida. `stdev` explotaria con n=1.
        "std": statistics.pstdev(pct),
        "count": len(pct),
        "below": sum(1 for p in pct if p < NOTA_APROBACION_PCT),
        "trend": pct[-1] - pct[0],
    }


def _progreso(obs: ObservacionMateria) -> float | None:
    """Cuanto del trimestre ya ocurrio, no cuantas notas hay.

    Un promedio de 40 en la semana 2 y uno de 40 en la semana 10 significan
    cosas opuestas, y sin esto el modelo no puede distinguirlos: describiria el
    presente en vez de anticipar el final.
    """
    planificados = obs.criterios_planificados
    if not planificados or planificados <= 0:
        return None
    calificados = sum(len(obs.notas_de(d)) for d in DIMENSIONES)
    return min(calificados / planificados * 100.0, 100.0)


def expandir(obs: ObservacionMateria) -> dict[str, float | None]:
    """La observacion como el vector que entra al modelo.

    Devuelve siempre las mismas claves, en el mismo orden, con `None` donde el
    dato falta. `None` y cero no son lo mismo: TF-DF trata el faltante como tal,
    y un cero seria una nota mala inventada.
    """
    vector: dict[str, float | None] = {}
    for dim in DIMENSIONES:
        stats = _estadisticos(_a_porcentaje(obs.notas_de(dim), dim))
        for nombre in _ESTADISTICOS:
            vector[f"{dim.nombre}_{nombre}"] = stats[nombre]
    vector["attendance_pct"] = obs.attendance_pct
    vector["progress_pct"] = _progreso(obs)
    return vector
