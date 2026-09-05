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

FUENTE UNICA — TODAVIA NO
-------------------------
La intencion es que esta sea la unica construccion del vector, usada por el
pipeline de entrenamiento y por `serving/api.py`: calcularlo de dos formas
distintas le daria al modelo, en produccion, features que nunca vio, y se
degradaria en silencio, sin un solo error, solo prediciendo peor.

**Hoy no lo es.** `training/train.py`, `serving/api.py` y `evaluation/evaluate.py`
siguen importando `FEATURE_COLS` de `preprocessing/features.py`, que define otras
cinco columnas escalares y promedia sobre las areas. Nadie importa este modulo
todavia. Conectarlo es el paso siguiente, junto con `registro_loader`; hasta
entonces conviven dos definiciones y esta nota es la que evita creer lo contrario.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Nota de aprobacion del Ministerio, expresada sobre 100 para que signifique lo
# mismo en las cuatro dimensiones.
NOTA_APROBACION_PCT = 51.0


# Los nombres exactos que la BDD acepta en `evaluation_criteria.dimension`, en el
# orden del `total_score` generado. Cuanto vale cada uno NO vive aca: depende de la
# gestion, y `Escala` es el unico lugar que lo dice.
DIMENSIONES: tuple[str, ...] = ("being", "knowing", "doing", "deciding")

# Margen para comparar la suma de una escala contra 100: los pesos podrian no ser
# enteros y un `!= 100.0` sobre flotantes rechazaria un reparto valido.
_TOLERANCIA_SUMA = 1e-9


@dataclass(frozen=True)
class Escala:
    """Cuanto vale cada dimension en una gestion, ya resuelta la autoevaluacion.

    Los encabezados crudos de los registros no suman 100 por si solos: dejan la
    autoevaluacion en columnas aparte (2023-2024 reparten Ser 10 / Saber 35 /
    Hacer 35 / Decidir 10 mas dos columnas de 5; 2025 reparte 5 / 45 / 40 / 5 mas
    una de 5). Lo que se guarda aca es el reparto **despues** de sumarlas, que es
    lo que hace comparable una nota de 2023 con una de hoy — ver
    `ESCALAS_POR_GESTION` para como se asigno cada una.

    Un 35 de Saber era la nota perfecta en 2023 y es un 78 por ciento hoy. Sin
    esto, adaptar el historico al sistema 2026 lo deforma en silencio: ninguna
    fila falla, todas mienten un poco.
    """

    being: float
    knowing: float
    doing: float
    deciding: float

    def __post_init__(self) -> None:
        if abs(self.total() - 100.0) > _TOLERANCIA_SUMA:
            raise ValueError(f"La escala debe sumar 100, suma {self.total()}")
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
ESCALA_VIGENTE = Escala(being=10.0, knowing=45.0, doing=40.0, deciding=5.0)

# Lo que decia el encabezado de cada registro historico.
#
# Saber y Hacer se leen literales del encabezado y no admiten discusion. Ser y
# Decidir si, porque la autoevaluacion figura como columna aparte y hay que
# decidir a que dimension se le suma:
#
#   2023-2024: 'SER - 10', 'DECIDIR - 10', y DOS columnas de autoevaluacion que
#     dicen a quien pertenecen — 'AUTOEVALUACION - SER 5' y
#     'AUTOEVALUACION - DECIDIR 5'. Se suma cada una a la suya: 15 y 15.
#   2025: 'SER - 5 Puntos' y 'DECIDIR - 5 Puntos' en la hoja EVAL, y una sola
#     'AUTOEVALUACION - SER Y D' que cubre las dos sin decir como se reparte.
#     Los 5 puntos van a Decidir, que es donde el sistema 2026 absorbe la
#     autoevaluacion. **Es una decision de mapeo, no una lectura**: si el reparto
#     real fuera otro, cambia aca y solo aca.
ESCALAS_POR_GESTION: dict[int, Escala] = {
    2023: Escala(being=15.0, knowing=35.0, doing=35.0, deciding=15.0),
    2024: Escala(being=15.0, knowing=35.0, doing=35.0, deciding=15.0),
    2025: Escala(being=5.0, knowing=45.0, doing=40.0, deciding=10.0),
    2026: ESCALA_VIGENTE,
}


def escala_de(gestion: int | None) -> Escala:
    """La escala de esa gestion, o la vigente si no se conoce.

    Sin gestion declarada es el caso del sistema, y la vigente es la respuesta
    correcta. Una gestion que si viene pero no esta mapeada es otra cosa: sale una
    planilla historica normalizada contra los topes de hoy, sin que falle una sola
    fila. Por eso se avisa.
    """
    if gestion is None:
        return ESCALA_VIGENTE
    escala = ESCALAS_POR_GESTION.get(gestion)
    if escala is None:
        logger.warning(
            "Gestion %s sin escala mapeada: se normaliza contra la vigente y las "
            "notas de esa planilla van a leerse corridas. Agregar su ponderacion "
            "a ESCALAS_POR_GESTION.",
            gestion,
        )
        return ESCALA_VIGENTE
    return escala


# Los siete estadisticos por dimension. El orden fija el de FEATURE_COLS.
_ESTADISTICOS = ("mean", "min", "max", "std", "count", "below", "trend")

FEATURE_COLS: list[str] = [
    f"{d}_{e}" for d in DIMENSIONES for e in _ESTADISTICOS
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
    if planificados is None or planificados <= 0:
        return None
    calificados = sum(len(obs.notas_de(d)) for d in DIMENSIONES)
    return min(calificados / planificados * 100.0, 100.0)


def expandir(
    obs: ObservacionMateria, escala: Escala = ESCALA_VIGENTE
) -> dict[str, float | None]:
    """La observacion como el vector que entra al modelo.

    `escala` es la ponderacion que regia cuando se pusieron esas notas: por
    defecto la del sistema, y `escala_de(gestion)` para una planilla historica.

    Devuelve siempre las mismas claves, en el mismo orden, con `None` donde el
    dato falta. `None` y cero no son lo mismo: TF-DF trata el faltante como tal,
    y un cero seria una nota mala inventada.
    """
    vector: dict[str, float | None] = {}
    for dim in DIMENSIONES:
        stats = _estadisticos(
            _a_porcentaje(obs.notas_de(dim), dim, escala.tope_de(dim))
        )
        for nombre in _ESTADISTICOS:
            vector[f"{dim}_{nombre}"] = stats[nombre]
    vector["attendance_pct"] = obs.attendance_pct
    vector["progress_pct"] = _progreso(obs)
    return vector
