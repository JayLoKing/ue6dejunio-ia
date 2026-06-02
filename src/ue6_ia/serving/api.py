"""Servicio REST de inferencia (FastAPI).

La API Spring Boot (ue6dejunio-api) llama a este servicio para clasificar el
riesgo de un estudiante. Devuelve la categoria y la probabilidad, que el
backend persiste en la tabla RiskPredictions.

Levantar:  uvicorn ue6_ia.serving.api:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..config import get_config
from ..preprocessing.features import FEATURE_COLS
from ..training.train import MODEL_SUBDIR

logger = logging.getLogger(__name__)
app = FastAPI(title="UE6 - Riesgo Academico", version="0.1.0")


class FeaturesEntrada(BaseModel):
    """Variables de entrada (doc HU-019). Exactamente lo que envia la API
    Spring Boot por estudiante: dimensiones + porcentaje de asistencia."""

    ser: float | None = Field(None, ge=0, le=100)
    saber: float | None = Field(None, ge=0, le=100)
    hacer: float | None = Field(None, ge=0, le=100)
    auto: float | None = Field(None, ge=0, le=100)
    attendance_pct: float | None = Field(None, ge=0, le=100)


class Prediccion(BaseModel):
    risk_level: str
    probability_score: float
    probabilidades: dict[str, float]


@lru_cache(maxsize=1)
def _cargar_modelo():
    import tensorflow as tf
    import tensorflow_decision_forests as tfdf  # noqa: F401

    cfg = get_config()
    model_dir = cfg.models_dir / MODEL_SUBDIR
    if not model_dir.exists():
        raise RuntimeError(f"Modelo no encontrado en {model_dir}. Entrena primero.")
    model = tf.keras.models.load_model(str(model_dir))
    clases = list(model.make_inspector().label_classes())
    return model, clases


def _verificar_token(authorization: str = Header(default="")) -> None:
    cfg = get_config()
    esperado = f"Bearer {cfg.env.api_token}"
    if authorization != esperado:
        raise HTTPException(status_code=401, detail="Token invalido")


@app.get("/health")
def health() -> dict:
    try:
        _, clases = _cargar_modelo()
        return {"status": "ok", "clases": clases}
    except Exception as e:  # noqa: BLE001
        return {"status": "sin_modelo", "detalle": str(e)}


@app.post("/predict", response_model=Prediccion)
def predict(entrada: FeaturesEntrada, _: None = Depends(_verificar_token)) -> Prediccion:
    import tensorflow_decision_forests as tfdf

    model, clases = _cargar_modelo()
    fila = {c: getattr(entrada, c, None) for c in FEATURE_COLS}
    df = pd.DataFrame([fila])
    ds = tfdf.keras.pd_dataframe_to_tf_dataset(df)
    proba = model.predict(ds)[0]
    idx = int(proba.argmax())
    return Prediccion(
        risk_level=clases[idx],
        probability_score=float(proba[idx]),
        probabilidades={c: float(p) for c, p in zip(clases, proba, strict=False)},
    )
