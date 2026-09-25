"""Reporte del modelo ya entrenado sobre un conjunto de datos.

QUE MIDE Y QUE NO
-----------------
Esto describe como se comporta el modelo **sobre las filas que se le pasen**. Si
son las mismas con las que se entreno, el numero que sale no es una estimacion de
nada: mide cuanto memorizo. La medicion honesta —validacion cruzada agrupada por
estudiante— la hace `training/train.py` y queda en `metadata.json`.

Sirve para mirar la matriz de confusion y la importancia de variables, que es la
ventaja de los arboles: se puede decir que dimension pesa en la decision.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import AppConfig, get_config
from ..preprocessing.features import TARGET_COL
from ..training.train import (
    CLASES_ORDENADAS,
    MODEL_SUBDIR,
    REPORTE_EN_MUESTRA,
    cargar_modelo,
    features_presentes,
    predecir,
)
from .metricas import resumen_clasificacion

logger = logging.getLogger(__name__)


def reporte_clasificacion(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> dict:
    """Carga el modelo guardado y reporta metricas sobre `dataset`."""
    cfg = cfg or get_config()

    modelo = cargar_modelo(cfg.models_dir / MODEL_SUBDIR)
    # La misma funcion que usa el entrenamiento, con su aviso incluido: filtrar
    # aca por separado seria reportar un accuracy como si el contrato estuviera
    # completo cuando le falta la mitad.
    cols = features_presentes(dataset)
    pred = predecir(modelo, dataset[cols + [TARGET_COL]])
    y_true = dataset[TARGET_COL].tolist()

    medidas = resumen_clasificacion(y_true, pred, CLASES_ORDENADAS)
    logger.info(
        "Sobre %d filas: accuracy %.3f (linea base %.3f), macro F1 %s",
        medidas["n"],
        medidas["accuracy"],
        medidas["linea_base"],
        "n/d" if medidas["macro_f1"] is None else f"{medidas['macro_f1']:.3f}",
    )
    logger.info(
        "Matriz de confusion (orden %s):\n%s", CLASES_ORDENADAS, medidas["matriz_confusion"]
    )
    for clase, detalle in medidas["por_clase"].items():
        logger.info(
            "  %-14s soporte=%3d recall=%s",
            clase,
            detalle["soporte"],
            "n/d" if detalle["recall"] is None else f"{detalle['recall']:.3f}",
        )

    # Estas metricas NO son una estimacion de como le ira con alguien nuevo, y el
    # reporte lo dice para que nadie las cite como si lo fueran.
    return {
        "advertencia": (
            "Metricas sobre las filas provistas. Si son las de entrenamiento, "
            "miden memorizacion. La estimacion honesta esta en metadata.json, "
            "bajo validacion_agrupada."
        ),
        "clases": CLASES_ORDENADAS,
        **medidas,
    }


def guardar_reporte(reporte: dict, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    out = cfg.models_dir / MODEL_SUBDIR / REPORTE_EN_MUESTRA
    out.write_text(json.dumps(reporte, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
