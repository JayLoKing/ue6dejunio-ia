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
from .ingestion.discovery import CursoFolder, discover_cursos
from .ingestion.registro_loader import DIMENSION_POR_COLUMNA, load_registro_trimestre
from .preprocessing.features import TARGET_COL, construir_features, etiquetar

logger = logging.getLogger(__name__)

# El nombre lo sabe este modulo y lo importa quien lo necesite: escrito dos veces,
# cambiarlo en uno dejaba al CLI diciendo "falta el dataset" recien construido.
NOMBRE_DATASET = "dataset_entrenamiento.parquet"


def _notas_cargadas(curso: CursoFolder, cfg: AppConfig) -> int:
    """Cuantas notas por criterio trae el primer trimestre de esa carpeta."""
    try:
        df = load_registro_trimestre(curso, 1, cfg)
    except Exception as e:  # noqa: BLE001 - una copia rota no puede voltear el pipeline
        logger.warning("No se pudo leer %s: %s", curso.path.name, e)
        return 0
    if df.empty:
        return 0
    return int(sum(df[d].map(len).sum() for d in DIMENSION_POR_COLUMNA))


def _elegir_entre_duplicados(
    cursos: list[CursoFolder], cfg: AppConfig
) -> list[CursoFolder]:
    """De cada curso repetido, la carpeta que realmente tiene notas cargadas.

    La carpeta trae copias del mismo curso ('- copia', '++', 'Feli ...'), y
    quedarse con la primera es quedarse con la que el orden alfabetico puso
    delante. Para 5A-2024 esa era una **plantilla en blanco** —celdas de criterio
    vacias y promedios que son formulas devolviendo cadena vacia— mientras la
    descartada tenia 1512 notas. El pipeline entrenaba con un formulario vacio y
    lo unico que lo decia era una linea de log que nadie leia.
    """
    por_label: dict[str, list[CursoFolder]] = {}
    for curso in cursos:
        por_label.setdefault(curso.label, []).append(curso)

    elegidos = []
    for label, candidatos in sorted(por_label.items()):
        if len(candidatos) == 1:
            elegidos.append(candidatos[0])
            continue
        puntajes = [(_notas_cargadas(c, cfg), c) for c in candidatos]
        mejor_notas, mejor = max(puntajes, key=lambda par: par[0])
        for notas, candidato in puntajes:
            if candidato is not mejor:
                # El nombre de la carpeta lleva el de la docente; a INFO va el
                # dato, no la persona.
                logger.info("Duplicado de %s descartado (%d notas)", label, notas)
                logger.debug("  carpeta descartada: %s", candidato.path.name)
        if mejor_notas == 0:
            logger.warning(
                "Ninguna de las %d carpetas de %s tiene notas cargadas",
                len(candidatos), label,
            )
        logger.info("%s: %d notas cargadas", label, mejor_notas)
        logger.debug("  carpeta elegida: %s", mejor.path.name)
        elegidos.append(mejor)
    return elegidos


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

    cursos = _elegir_entre_duplicados(cursos, cfg)
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

    # La etiqueta se pone recien aca: mira el trimestre siguiente de la misma
    # materia, y para eso los tres trimestres tienen que estar juntos. Puesta por
    # trimestre suelto, ninguna fila encontraba pareja y todas salian sin etiqueta.
    dataset = etiquetar(pd.concat(partes, ignore_index=True), cfg)
    if solo_etiquetados:
        dataset = dataset[dataset[TARGET_COL].notna()].reset_index(drop=True)

    logger.info("Dataset final: %d filas", len(dataset))
    return dataset


def guardar_dataset(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    # Por `AppConfig` y no leyendo el yaml crudo: las entradas ya cuelgan de
    # `data_root` y son sobrescribibles por UE6_*. Dejando la salida en la ruta
    # del yaml, mover el data_root movia las entradas y el parquet con notas de
    # menores se quedaba donde estaba, que puede ser adentro del repo.
    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / NOMBRE_DATASET
    dataset.to_parquet(out, index=False)
    logger.info("Dataset guardado en %s", out)
    return out
