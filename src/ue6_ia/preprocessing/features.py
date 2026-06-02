"""Construccion de la matriz de features para el modelo.

CONTRATO DE FEATURES (debe coincidir EXACTAMENTE con lo que la API Spring Boot
envia en inferencia): por cada estudiante el sistema manda 5 valores:

    ser, saber, hacer, auto, attendance_pct

Esos mismos 5 son los que se usan para entrenar (con data de gestiones
pasadas). Cualquier otra variable derivada NO se usa, porque el sistema no la
provee en tiempo de prediccion.

Granularidad: una fila por (estudiante, gestion, trimestre). Las dimensiones se
promedian sobre todas las areas del trimestre (un estudiante = un set de
ser/saber/hacer/auto). El target sale del centralizador de años pasados.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import AppConfig
from ..labeling import aplicar_override_situacion, clasificar_riesgo
from .cleaning import coaccionar_rango, limpiar_df_nombres

logger = logging.getLogger(__name__)

# Columnas crudas que entrega registro_loader
_DIM_RAW = ["score_ser", "score_saber", "score_hacer", "score_auto"]

# Contrato con el sistema: estas son las unicas variables de entrada del modelo.
FEATURE_COLS = ["ser", "saber", "hacer", "auto", "attendance_pct"]
TARGET_COL = "risk_level"

_CLAVES = ["gestion", "grado", "paralelo", "trimestre", "nombre_key"]


def construir_features(
    df_registro: pd.DataFrame,
    df_centralizador: pd.DataFrame,
    df_asistencia: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Combina registro + asistencia + centralizador en la matriz final."""
    if df_registro.empty:
        logger.warning("Registro vacio: no se pueden construir features")
        return pd.DataFrame()

    reg = limpiar_df_nombres(df_registro)
    reg = coaccionar_rango(reg, _DIM_RAW, 0, 100)

    # --- promedio de cada dimension por estudiante-trimestre (sobre areas) ---
    agg = (
        reg.groupby(_CLAVES)
        .agg(
            ser=("score_ser", "mean"),
            saber=("score_saber", "mean"),
            hacer=("score_hacer", "mean"),
            auto=("score_auto", "mean"),
        )
        .reset_index()
    )

    # --- asistencia (opcional; NaN hasta confirmar layout) ---
    if not df_asistencia.empty:
        asis = limpiar_df_nombres(df_asistencia)
        agg = agg.merge(
            asis[[*_CLAVES, "attendance_pct"]],
            on=_CLAVES,
            how="left",
        )
    else:
        agg["attendance_pct"] = pd.NA

    # --- target desde el centralizador de años pasados ---
    agg["risk_level"] = _etiquetar(agg, reg, df_centralizador, cfg)

    logger.info(
        "Features: %d filas (estudiante-trimestre), %d etiquetadas",
        len(agg),
        agg["risk_level"].notna().sum(),
    )
    return agg


def _etiquetar(
    feats: pd.DataFrame,
    reg: pd.DataFrame,
    df_centralizador: pd.DataFrame,
    cfg: AppConfig,
) -> pd.Series:
    """Asigna risk_level por (estudiante, trimestre) usando el promedio del
    centralizador (fuente independiente). Si no hay centralizador, usa como
    respaldo el promedio de areas del propio trimestre."""
    umbrales = cfg["riesgo"]

    # respaldo: promedio de las areas del trimestre por estudiante
    respaldo = (
        reg.groupby(_CLAVES)["prom_area_trim"].mean().rename("prom_resp")
        if "prom_area_trim" in reg.columns
        else pd.Series(dtype=float)
    )
    feats = feats.merge(respaldo, on=_CLAVES, how="left") if not respaldo.empty else feats

    if df_centralizador.empty:
        base = feats["prom_resp"] if "prom_resp" in feats.columns else None
        if base is None:
            return pd.Series([None] * len(feats), index=feats.index)
        return base.map(lambda p: clasificar_riesgo(p, umbrales))

    cen = limpiar_df_nombres(df_centralizador)
    cols_trim = {1: "prom_t1", 2: "prom_t2", 3: "prom_t3"}
    cen_idx = cen.set_index(["gestion", "grado", "paralelo", "nombre_key"])

    etiquetas = []
    for _, row in feats.iterrows():
        key = (row["gestion"], row["grado"], row["paralelo"], row["nombre_key"])
        prom, situacion = None, None
        if key in cen_idx.index:
            crow = cen_idx.loc[key]
            if isinstance(crow, pd.DataFrame):
                crow = crow.iloc[0]
            prom = crow.get(cols_trim[int(row["trimestre"])])
            situacion = crow.get("situacion")
        if prom is None:
            prom = row.get("prom_resp")
        clase = clasificar_riesgo(prom, umbrales)
        etiquetas.append(aplicar_override_situacion(clase, situacion))
    return pd.Series(etiquetas, index=feats.index)
