"""Entrenamiento del modelo de riesgo academico con TF Decision Forests.

Modelo principal: Gradient Boosted Trees (o Random Forest, segun config).
Coincide con el doc ("motor de ML basado en Arboles de Decision"), es ideal
para datos tabulares pequeños y es interpretable (importancia de variables).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import AppConfig, get_config
from ..preprocessing.features import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

MODEL_SUBDIR = "tfdf_riesgo"


def _split(dataset: pd.DataFrame, cfg: AppConfig):
    from sklearn.model_selection import train_test_split  # opcional; ver nota abajo

    semilla = cfg["entrenamiento"]["semilla"]
    test_size = cfg["entrenamiento"]["test_size"]
    return train_test_split(
        dataset,
        test_size=test_size,
        random_state=semilla,
        stratify=dataset[TARGET_COL],
    )


def entrenar(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> Path:
    """Entrena y guarda el modelo. Devuelve la ruta del SavedModel."""
    cfg = cfg or get_config()
    import tensorflow_decision_forests as tfdf

    logger.info("Entrenamiento en CPU (TF Decision Forests no usa GPU).")

    cols = [c for c in FEATURE_COLS if c in dataset.columns]
    data = dataset[cols + [TARGET_COL]].copy()

    # TF-DF maneja NaN nativamente; aun asi documentamos cuantos hay.
    logger.info("Filas: %d | features: %s", len(data), cols)
    logger.info("Distribucion de clases:\n%s", data[TARGET_COL].value_counts())

    train_df, test_df = _split(data, cfg)

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=TARGET_COL)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=TARGET_COL)

    modelo_tipo = cfg["entrenamiento"]["modelo"]
    if modelo_tipo == "random_forest":
        model = tfdf.keras.RandomForestModel(verbose=1)
    else:
        model = tfdf.keras.GradientBoostedTreesModel(verbose=1)

    model.fit(train_ds)
    model.compile(metrics=["accuracy"])
    evaluacion = model.evaluate(test_ds, return_dict=True)
    logger.info("Evaluacion holdout: %s", evaluacion)

    out_dir = cfg.models_dir / MODEL_SUBDIR
    model.save(str(out_dir))

    # Metadatos para inferencia y trazabilidad
    meta = {
        "modelo": modelo_tipo,
        "features": cols,
        "target": TARGET_COL,
        "metricas_holdout": {k: float(v) for k, v in evaluacion.items()},
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Modelo guardado en %s", out_dir)
    return out_dir
