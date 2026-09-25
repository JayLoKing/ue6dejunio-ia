"""Como se mide un clasificador de riesgo academico, y por que asi.

El accuracy solo no sirve aca. Las clases estan desbalanceadas —`RiesgoCritico`
es el 6.6% de las filas— asi que un modelo que jamas prediga riesgo puede sacar
un numero alto y no servir para nada. Y el objetivo del sistema no es acertar en
promedio: es **avisarle al docente antes de que el estudiante repruebe**.

De ahi el orden de importancia:

1. `recall_riesgo_critico` — de todos los que estaban en riesgo, a cuantos los
   detecto. Un falso negativo es un chico que reprueba sin que nadie avisara; un
   falso positivo cuesta una revision de mas. Los dos errores no valen igual.
2. `macro_f1` — promedia las clases dandoles el mismo peso, de modo que la clase
   rara no desaparece detras de las dos mayoritarias.
3. `accuracy`, siempre contra `linea_base_mayoritaria`. Sin esa referencia un 84%
   puede ser excelente o peor que responder siempre lo mismo.
4. La matriz de confusion, porque muestra hacia donde se equivoca: confundir
   `EnRiesgo` con `RiesgoCritico` no es lo mismo que confundirlo con
   `Sobresaliente`.

Nada de error cuadratico medio: eso mide cuanto se aleja un numero predicho, y
aca no se predice un numero sino una de cuatro categorias.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter

logger = logging.getLogger(__name__)

RIESGO_CRITICO = "RiesgoCritico"


def linea_base_mayoritaria(y: list[str]) -> float | None:
    """Que sacaria un modelo que siempre responde la clase mas frecuente.

    Es el piso contra el que se lee cualquier accuracy. Sin el, el numero no dice
    si el modelo aprendio algo o solo aprendio a repetir.
    """
    if not y:
        return None
    return Counter(y).most_common(1)[0][1] / len(y)


def _por_clase(y_true: list[str], y_pred: list[str], clase: str) -> dict:
    verdaderos = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p == clase)
    reales = sum(1 for t in y_true if t == clase)
    predichos = sum(1 for p in y_pred if p == clase)

    # Sin ejemplos de la clase el recall no es 1.0 ni 0.0: no se sabe. Devolver un
    # numero ahi inventaria una medicion que el pliegue no permite hacer.
    recall = verdaderos / reales if reales else None
    precision = verdaderos / predichos if predichos else None

    if reales == 0:
        # El pliegue no trajo ni un caso de esta clase, asi que su F1 no existe.
        # Ponerle 0.0 seria peor que no medirlo: ese cero entra al macro F1, y el
        # macro F1 elige el modelo — quedaria castigado por un reparto de pliegues
        # que no eligio. Los falsos positivos que haya cometido igual se ven en la
        # precision de las otras clases.
        f1 = None
    elif precision is None:
        # Hubo casos y no predijo ninguno: eso SI se midio, y es un fracaso total.
        f1 = 0.0
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1, "soporte": reales}


def resumen_clasificacion(y_true: list[str], y_pred: list[str], clases: list[str]) -> dict:
    """Las metricas de una tanda de predicciones, sin promediar nada de antemano."""
    aciertos = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    por_clase = {c: _por_clase(y_true, y_pred, c) for c in clases}

    f1s = [d["f1"] for d in por_clase.values() if d["f1"] is not None]
    indice = {c: i for i, c in enumerate(clases)}
    matriz = [[0] * len(clases) for _ in clases]
    desconocidas: set[str] = set()
    for t, p in zip(y_true, y_pred, strict=True):
        if t in indice and p in indice:
            matriz[indice[t]][indice[p]] += 1
        else:
            desconocidas.update(e for e in (t, p) if e not in indice)

    if desconocidas:
        # El accuracy de arriba cuenta todos los pares; la matriz solo cuenta
        # los que tienen sus dos etiquetas entre las clases declaradas. Con una
        # etiqueta desconocida los dos numeros dejan de hablar de la misma
        # poblacion, y `graficos.expandir_matriz` vuelve a medir *desde la
        # matriz*: el documento termina citando dos accuracy distintos de la
        # misma corrida, los dos rotulados fuera de muestra. Tampoco cierra
        # `sum(matriz) == n`, y `n` viaja en este mismo dict.
        #
        # No se puede decidir aca cual es la buena — una clase nueva en los
        # datos y un error de tipeo en la configuracion llegan iguales — pero
        # descartarla sin decirlo es lo unico que seguro esta mal. Son nombres
        # de clase, no de estudiantes: se pueden registrar.
        logger.warning(
            "Etiquetas fuera de las clases declaradas, descartadas de la matriz "
            "de confusion pero contadas en el accuracy: %s. Las clases son %s",
            ", ".join(sorted(desconocidas)),
            clases,
        )

    return {
        "accuracy": aciertos / len(y_true) if y_true else None,
        "macro_f1": statistics.fmean(f1s) if f1s else None,
        "recall_riesgo_critico": por_clase.get(RIESGO_CRITICO, {}).get("recall"),
        "linea_base": linea_base_mayoritaria(y_true),
        "por_clase": por_clase,
        "matriz_confusion": matriz,
        "n": len(y_true),
    }


def resumir_pliegues(pliegues: list[dict]) -> dict:
    """Media y desvio de cada metrica a lo largo de los pliegues.

    Con 34 estudiantes un solo corte es una anecdota: cae distinto segun a quien
    le toque quedar afuera. El desvio es la parte honesta del resultado — dice
    cuanto se mueve el numero cuando cambia el reparto.

    Los `None` se descartan en vez de contarse como cero: un pliegue que no tuvo
    ni un caso de `RiesgoCritico` no midio mal, no midio nada.
    """
    nombres = {k for p in pliegues for k, v in p.items() if isinstance(v, int | float)}
    nombres |= {k for p in pliegues for k in p if p[k] is None}

    resumen = {}
    for nombre in sorted(nombres):
        valores = [
            p[nombre] for p in pliegues
            if isinstance(p.get(nombre), int | float)
        ]
        resumen[nombre] = {
            "media": statistics.fmean(valores) if valores else None,
            "desvio": statistics.pstdev(valores) if valores else None,
            "pliegues": len(valores),
        }
    return resumen
