"""Carga de la hoja 'PROMEDIOS POR TRIMESTRE' del centralizador.

Es la fuente de ETIQUETAS (labels) para entrenar: en años pasados contiene
el promedio por trimestre, el promedio final y la situacion final
(p.ej. 'perdio el año'). Esto da el outcome real aprobo/reprobo.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import AppConfig
from .discovery import CursoFolder
from .excel_reader import cell, find_file, read_sheet_grid

logger = logging.getLogger(__name__)


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_centralizador(curso: CursoFolder, cfg: AppConfig) -> pd.DataFrame:
    """Devuelve un DataFrame por estudiante con promedios y situacion final.

    Columnas: gestion, grado, paralelo, docente, nro, nombre,
              prom_t1, prom_t2, prom_t3, promedio_final, situacion
    """
    lay = cfg["layout_centralizador"]
    fpat = cfg["archivos"]["centralizador"]
    path = find_file(curso.path, fpat)
    if path is None:
        return pd.DataFrame()

    try:
        grid = read_sheet_grid(path, lay["hoja"])
    except KeyError as e:
        logger.warning("%s: %s", curso.label, e)
        return pd.DataFrame()

    filas = []
    for r in range(lay["fila_inicio_estudiantes"], len(grid)):
        nombre = cell(grid, r, lay["col_nombre"])
        if not nombre or not str(nombre).strip():
            continue
        filas.append(
            {
                "gestion": curso.gestion,
                "grado": curso.grado,
                "paralelo": curso.paralelo,
                "docente": curso.docente,
                "nro": cell(grid, r, lay["col_nro"]),
                "nombre": str(nombre).strip(),
                "prom_t1": _to_float(cell(grid, r, lay["col_prom_t1"])),
                "prom_t2": _to_float(cell(grid, r, lay["col_prom_t2"])),
                "prom_t3": _to_float(cell(grid, r, lay["col_prom_t3"])),
                "promedio_final": _to_float(cell(grid, r, lay["col_promedio_final"])),
                "situacion": (str(cell(grid, r, lay["col_situacion"]) or "").strip().lower()),
            }
        )
    df = pd.DataFrame(filas)
    logger.info("Centralizador %s: %d estudiantes", curso.label, len(df))
    return df
