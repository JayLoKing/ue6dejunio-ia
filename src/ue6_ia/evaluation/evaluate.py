"""Metricas y reporte del modelo: matriz de confusion, F1 por clase e
importancia de variables (ventaja de los arboles: son interpretables)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import AppConfig, get_config
from ..preprocessing.features import FEATURE_COLS, TARGET_COL
from ..training.train import MODEL_SUBDIR

logger = logging.getLogger(__name__)


def reporte_clasificacion(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> dict:
    """Carga el modelo guardado y reporta metricas sobre `dataset`."""
    cfg = cfg or get_config()
    import tensorflow as tf
    import tensorflow_decision_forests as tfdf  # noqa: F401  (registra ops del modelo)
    from sklearn.metrics import classification_report, confusion_matrix

    model_dir = cfg.models_dir / MODEL_SUBDIR
    model = tf.keras.models.load_model(str(model_dir))

    cols = [c for c in FEATURE_COLS if c in dataset.columns]
    ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset[cols + [TARGET_COL]], label=TARGET_COL)

    proba = model.predict(ds)
    clases = list(model.make_inspector().label_classes())
    pred = [clases[i] for i in proba.argmax(axis=1)]
    y_true = dataset[TARGET_COL].tolist()

    rep = classification_report(y_true, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=clases)
    logger.info("Matriz de confusion (orden %s):\n%s", clases, cm)
    logger.info("Reporte:\n%s", classification_report(y_true, pred, zero_division=0))

    # Importancia de variables
    try:
        inspector = model.make_inspector()
        importancias = inspector.variable_importances()
        logger.info("Importancia de variables: %s", importancias)
    except Exception as e:  # noqa: BLE001
        logger.debug("No se pudo extraer importancia: %s", e)

    return {"classification_report": rep, "labels": clases, "confusion_matrix": cm.tolist()}


def guardar_reporte(reporte: dict, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    import json

    out = cfg.models_dir / MODEL_SUBDIR / "reporte_evaluacion.json"
    out.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
