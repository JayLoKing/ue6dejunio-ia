"""Orquestacion: de carpetas Excel a la matriz de features lista para entrenar.

Recorre las carpetas de curso de las gestiones indicadas, carga registros,
asistencia y centralizador de cada trimestre, construye features y concatena
todo en un unico DataFrame que se guarda en data/processed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import AppConfig, get_config
from .ingestion.asistencia_loader import load_asistencia_trimestre
from .ingestion.centralizador_loader import load_centralizador
from .ingestion.discovery import discover_cursos
from .ingestion.registro_loader import load_registro_trimestre
from .preprocessing.features import construir_features

logger = logging.getLogger(__name__)


def construir_dataset(cfg: AppConfig | None = None, solo_etiquetados: bool = True) -> pd.DataFrame:
    """Construye la matriz de features de todas las gestiones de entrenamiento.

    Args:
        solo_etiquetados: si True, descarta filas sin risk_level (para entrenar).
    """
    cfg = cfg or get_config()
    gestiones = set(cfg["entrenamiento"]["gestiones_entrenamiento"])

    # Entrenamiento = SOLO gestiones pasadas. El 2026 se centraliza en el
    # sistema y se predice via API (no se procesa aqui).
    cursos = discover_cursos(cfg.dir_pasados)
    cursos = [c for c in cursos if c.gestion in gestiones]

    # Dedup: la carpeta tiene copias del mismo curso (- copia, ++, etc.).
    # Se conserva la primera por (gestion, grado, paralelo) y se avisa.
    vistos: set[str] = set()
    unicos = []
    for c in cursos:
        if c.label in vistos:
            logger.warning("Curso duplicado ignorado: %s (%s)", c.label, c.path.name)
            continue
        vistos.add(c.label)
        unicos.append(c)
    cursos = unicos
    logger.info("Cursos a procesar (gestiones %s): %d", sorted(gestiones), len(cursos))

    partes = []
    for curso in cursos:
        cen = load_centralizador(curso, cfg)
        for trimestre in (1, 2, 3):
            reg = load_registro_trimestre(curso, trimestre, cfg)
            if reg.empty:
                continue
            asis = load_asistencia_trimestre(curso, trimestre, cfg)
            feats = construir_features(reg, cen, asis, cfg)
            if not feats.empty:
                partes.append(feats)

    if not partes:
        logger.error("No se genero ninguna fila. Revisa rutas/layout en config.yaml")
        return pd.DataFrame()

    dataset = pd.concat(partes, ignore_index=True)
    if solo_etiquetados:
        dataset = dataset[dataset["risk_level"].notna()].reset_index(drop=True)

    logger.info("Dataset final: %d filas", len(dataset))
    return dataset


def guardar_dataset(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "dataset_entrenamiento.parquet"
    dataset.to_parquet(out, index=False)
    logger.info("Dataset guardado en %s", out)
    return out
