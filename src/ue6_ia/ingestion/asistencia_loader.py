"""Calculo del porcentaje de asistencia por trimestre.

Las hojas mensuales (FEBRERO, MARZO, ...) tienen una grilla: filas = estudiantes,
columnas = dias del mes con marcas ('.'=presente, 'F'=falta, 'R'=retraso,
'L'=licencia). Las columnas-resumen del archivo son inconsistentes, asi que se
cuentan las marcas directamente.

attendance_pct (por trimestre) =
    (presentes + retrasos + licencias) / dias_habiles * 100

donde dias_habiles = total de marcas no vacias del estudiante en los meses del
trimestre. Es robusto a meses incompletos (gestion en curso) porque solo cuenta
dias efectivamente registrados.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import AppConfig
from .discovery import CursoFolder
from .excel_reader import find_file, list_sheets, read_sheet_grid

logger = logging.getLogger(__name__)


def _es_int(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return v == int(v)
    return False


def _norm(v) -> str:
    return str(v).strip().upper() if v is not None else ""


def _fila_primer_estudiante(grid: list[list], col_nombre: int) -> int:
    """Primera fila cuyo col0 es entero (numero de lista) y tiene nombre."""
    for r, row in enumerate(grid):
        c0 = row[0] if len(row) > 0 else None
        nombre = row[col_nombre] if len(row) > col_nombre else None
        if _es_int(c0) and nombre and str(nombre).strip():
            return r
    return -1


def _cols_dia(grid: list[list], fila_est: int, primera_col: int) -> list[int]:
    """Columnas de marca diaria: aquellas con la fila de numeros-de-dia
    (justo encima del primer estudiante) conteniendo un entero."""
    for r in range(fila_est - 1, max(fila_est - 5, -1), -1):
        row = grid[r] if r < len(grid) else []
        cols = [c for c in range(primera_col, len(row)) if _es_int(row[c])]
        if len(cols) >= 5:  # un mes tiene ~20 dias; 5 ya confirma que es la fila
            return cols
    # respaldo: heuristica por ancho tipico (no deberia usarse)
    return []


def _contar_mes(grid: list[list], acfg: dict) -> dict[str, dict]:
    """Devuelve {nombre: {'asistio': n, 'habiles': n}} para un mes."""
    col_nombre = acfg["col_nombre"]
    fila_est = _fila_primer_estudiante(grid, col_nombre)
    if fila_est < 0:
        return {}
    cols = _cols_dia(grid, fila_est, acfg["primera_col_dia"])
    if not cols:
        return {}

    presente = {_norm(x) for x in acfg["marca_presente"]}
    retraso = {_norm(x) for x in acfg["marca_retraso"]}
    licencia = {_norm(x) for x in acfg["marca_licencia"]}
    falta = {_norm(x) for x in acfg["marca_falta"]}
    asiste = presente | retraso | licencia

    out: dict[str, dict] = {}
    for r in range(fila_est, len(grid)):
        row = grid[r]
        c0 = row[0] if len(row) > 0 else None
        nombre = row[col_nombre] if len(row) > col_nombre else None
        if not (_es_int(c0) and nombre and str(nombre).strip()):
            continue
        asistio = habiles = 0
        for c in cols:
            marca = _norm(row[c]) if c < len(row) else ""
            if marca in asiste:
                asistio += 1
                habiles += 1
            elif marca in falta:
                habiles += 1
            # marca vacia = no hubo clase / no registrado -> no cuenta
        out[str(nombre).strip()] = {"asistio": asistio, "habiles": habiles}
    return out


def load_asistencia_trimestre(
    curso: CursoFolder, trimestre: int, cfg: AppConfig
) -> pd.DataFrame:
    """Devuelve attendance_pct por estudiante para el trimestre.

    Columnas: gestion, grado, paralelo, trimestre, nombre, attendance_pct
    """
    path = find_file(curso.path, cfg["archivos"]["asistencia"])
    if path is None:
        return pd.DataFrame()

    acfg = cfg["asistencia"]
    meses = [m.upper() for m in acfg["meses_por_trimestre"][trimestre]]
    hojas = {s.upper(): s for s in list_sheets(path)}

    acum: dict[str, dict] = {}
    for mes in meses:
        if mes not in hojas:
            continue
        grid = read_sheet_grid(path, hojas[mes])
        for nombre, d in _contar_mes(grid, acfg).items():
            a = acum.setdefault(nombre, {"asistio": 0, "habiles": 0})
            a["asistio"] += d["asistio"]
            a["habiles"] += d["habiles"]

    if not acum:
        logger.warning("%s T%d: sin datos de asistencia", curso.label, trimestre)
        return pd.DataFrame()

    filas = []
    for nombre, d in acum.items():
        pct = (d["asistio"] / d["habiles"] * 100) if d["habiles"] else None
        filas.append(
            {
                "gestion": curso.gestion,
                "grado": curso.grado,
                "paralelo": curso.paralelo,
                "trimestre": trimestre,
                "nombre": nombre,
                "attendance_pct": pct,
            }
        )
    df = pd.DataFrame(filas)
    logger.info(
        "Asistencia %s T%d: %d estudiantes (%%prom %.1f)",
        curso.label,
        trimestre,
        len(df),
        df["attendance_pct"].mean(skipna=True) if len(df) else 0,
    )
    return df
