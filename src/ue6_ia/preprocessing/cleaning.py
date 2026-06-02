"""Limpieza y normalizacion de datos crudos.

El punto critico es el NOMBRE del estudiante: es la llave para cruzar
calificaciones, asistencia y centralizador, y viene con tildes, espacios
dobles, mayus/minus inconsistentes y enie. Se normaliza a una forma canonica.
"""

from __future__ import annotations

import re

import pandas as pd
from unidecode import unidecode

_ESPACIOS = re.compile(r"\s+")


def normalizar_nombre(nombre: str) -> str:
    """Forma canonica de un nombre para hacer join entre archivos.

    'ARGANDOÑA  Vargas ' -> 'ARGANDONA VARGAS'
    """
    if nombre is None:
        return ""
    s = unidecode(str(nombre)).upper().strip()
    s = _ESPACIOS.sub(" ", s)
    return s


def limpiar_df_nombres(df: pd.DataFrame, col: str = "nombre") -> pd.DataFrame:
    """Añade columna `nombre_key` normalizada y descarta filas sin nombre."""
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out["nombre_key"] = out[col].map(normalizar_nombre)
    out = out[out["nombre_key"].str.len() > 0]
    return out


def coaccionar_rango(df: pd.DataFrame, cols: list[str], lo: float, hi: float) -> pd.DataFrame:
    """Pone NaN a valores fuera de [lo, hi] (notas corruptas / cabeceras coladas)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out.loc[(out[c] < lo) | (out[c] > hi), c] = pd.NA
    return out
