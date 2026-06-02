"""Carga de calificaciones por dimension desde el registro trimestral.

Las hojas de area (MATE, LENG, ...) tienen bloques de columnas por dimension.
El numero de criterios varia por area, asi que las columnas de PROMEDIO se
DETECTAN por encabezado (no por indice fijo).

Escala alineada a la RM 0001/2026 (Ser=10, Saber=45, Hacer=40, Auto=5):
  ser  = PROMEDIO SER (-5) + PROMEDIO DECIDIR (-5)   -> /10
  saber= PROMEDIO dentro del bloque SABER            -> /45
  hacer= PROMEDIO dentro del bloque HACER            -> /40
  auto = AUTOEVALUACION                              -> /5

Salida tidy: una fila por (estudiante, area, trimestre).
"""

from __future__ import annotations

import logging
import unicodedata

import pandas as pd

from ..config import AppConfig
from .discovery import CursoFolder
from .excel_reader import find_file, list_sheets, read_sheet_grid

logger = logging.getLogger(__name__)


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _es_int(v) -> bool:
    return (isinstance(v, int) and not isinstance(v, bool)) or (
        isinstance(v, float) and v == int(v)
    )


def _nb(v) -> str:
    """Normaliza encabezado: sin acentos, mayusculas, sin espacios."""
    if v is None:
        return ""
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().upper()
    return "".join(s.split())


def _fila_primer_estudiante(grid: list[list], col_nombre: int) -> int:
    for r, row in enumerate(grid):
        c0 = row[0] if row else None
        nom = row[col_nombre] if len(row) > col_nombre else None
        if _es_int(c0) and nom and str(nom).strip():
            return r
    return -1


def _localizar_columnas(grid: list[list], fila_est: int) -> dict:
    """Detecta indices de columna de cada dimension escaneando encabezados."""
    ser = dec = auto = trim = saber_h = hacer_h = None
    proms: list[int] = []
    for r in range(fila_est):
        for c, v in enumerate(grid[r]):
            n = _nb(v)
            if not n:
                continue
            if "SABER" in n and saber_h is None:
                saber_h = c
            if "HACER" in n and hacer_h is None:
                hacer_h = c
            if "PROMEDIO" in n and "SER" in n and "TRIMESTRAL" not in n and "AUTOEVAL" not in n:
                ser = c
            if "PROMEDIO" in n and "DECIDIR" in n:
                dec = c
            if "AUTOEVAL" in n:
                auto = c
            if "PROMEDIO" in n and "TRIMESTRAL" in n:
                trim = c
            if n == "PROMEDIO":
                proms.append(c)
    saber = next((c for c in proms if saber_h and hacer_h and saber_h < c < hacer_h), None)
    hacer = next((c for c in proms if hacer_h and c > hacer_h and (dec is None or c < dec)), None)
    return {"ser": ser, "dec": dec, "saber": saber, "hacer": hacer, "auto": auto, "trim": trim}


def _load_area(grid: list[list], area_nombre: str) -> list[dict]:
    fila_est = _fila_primer_estudiante(grid, 1)
    if fila_est < 0:
        return []
    col = _localizar_columnas(grid, fila_est)
    # Sin bloques SABER/HACER reconocibles -> area no calificada aqui (tecnica).
    if col["saber"] is None or col["hacer"] is None:
        return []

    def cell(row, c):
        return row[c] if c is not None and c < len(row) else None

    filas = []
    for r in range(fila_est, len(grid)):
        row = grid[r]
        c0 = row[0] if row else None
        nom = row[1] if len(row) > 1 else None
        if not (_es_int(c0) and nom and str(nom).strip()):
            continue
        ser = _to_float(cell(row, col["ser"])) or 0.0
        dec = _to_float(cell(row, col["dec"])) or 0.0
        saber = _to_float(cell(row, col["saber"]))
        hacer = _to_float(cell(row, col["hacer"]))
        auto = _to_float(cell(row, col["auto"]))
        prom = _to_float(cell(row, col["trim"]))
        # fila sin nota real en el area -> ignorar
        if saber is None and hacer is None and prom is None:
            continue
        filas.append(
            {
                "nombre": str(nom).strip(),
                "area": area_nombre,
                "score_ser": ser + dec,   # /10 (RM)
                "score_saber": saber,
                "score_hacer": hacer,
                "score_auto": auto,
                "prom_area_trim": prom,
            }
        )
    return filas


def load_registro_trimestre(
    curso: CursoFolder, trimestre: int, cfg: AppConfig
) -> pd.DataFrame:
    """Dimensiones por (estudiante, area) para un trimestre.

    Columnas: gestion, grado, paralelo, trimestre, nombre, area,
              score_ser, score_saber, score_hacer, score_auto, prom_area_trim
    """
    pat = cfg["archivos"]["registro_trimestre"].format(trimestre=trimestre)
    path = find_file(curso.path, pat)
    if path is None:
        return pd.DataFrame()

    # Match tolerante de nombres de hoja (acentos/espacios varian).
    hojas = {_nb(s): s for s in list_sheets(path)}
    todas = []
    for area in cfg["areas"]:
        real = hojas.get(_nb(area["hoja"]))
        if real is None:
            continue
        grid = read_sheet_grid(path, real)
        todas.extend(_load_area(grid, area["nombre"]))

    df = pd.DataFrame(todas)
    if df.empty:
        logger.warning("%s T%d: sin datos de areas", curso.label, trimestre)
        return df

    df["gestion"] = curso.gestion
    df["grado"] = curso.grado
    df["paralelo"] = curso.paralelo
    df["trimestre"] = trimestre
    logger.info(
        "Registro %s T%d: %d filas (estudiante x area), %d areas",
        curso.label,
        trimestre,
        len(df),
        df["area"].nunique(),
    )
    return df
