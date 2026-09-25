"""Descubrimiento de carpetas de curso y metadatos a partir del nombre.

Ejemplos de nombre de carpeta:
  "5º B Registro Pedag. 2026 Elizabeth Argandoña Vargas"
  "6º A Registro Pedag. 2025 Silvia Felicidad Montaño Nogales"
  "QUINTO DE PRIMARIA 2024"   <- formato libre (se ignora si no matchea)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Busca grado+paralelo y año en CUALQUIER parte del nombre (tolerante a
# prefijos tipo "Feli ..." y a los simbolos º/°). Ej:
#   "5º B Registro Pedag. 2026 Nombre"     -> 5, B, 2026
#   "Feli 4º A Registro 2023 Profa Silvia" -> 4, A, 2023
_GRADO_PAR = re.compile(r"(\d+)\s*[º°o]\s*[\"']?\s*([A-Ca-c])\b", re.UNICODE)
_ANIO = re.compile(r"(20\d{2})")


@dataclass(frozen=True)
class CursoFolder:
    path: Path
    grado: int
    paralelo: str
    gestion: int
    docente: str

    @property
    def label(self) -> str:
        return f"{self.grado}{self.paralelo}-{self.gestion}"


def parse_folder_name(path: Path) -> CursoFolder | None:
    """Extrae metadatos de la carpeta. Devuelve None si no se hallan
    grado+paralelo y año."""
    name = path.name
    mg = _GRADO_PAR.search(name)
    ma = _ANIO.search(name)
    if not mg or not ma:
        logger.debug("Carpeta ignorada (sin grado/paralelo/año): %s", name)
        return None
    docente = name[mg.end() :]
    docente = re.sub(r"(?i)registro|pedag\.?|prof[a]?\.?|lic\.?|20\d{2}", " ", docente)
    return CursoFolder(
        path=path,
        grado=int(mg.group(1)),
        paralelo=mg.group(2).upper(),
        gestion=int(ma.group(1)),
        docente=" ".join(docente.split()).strip(),
    )


def discover_cursos(base_dir: Path) -> list[CursoFolder]:
    """Lista las carpetas de curso validas bajo `base_dir`."""
    if not base_dir.exists():
        logger.error("Directorio no existe: %s", base_dir)
        return []
    cursos = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        curso = parse_folder_name(child)
        if curso:
            cursos.append(curso)
    logger.info("Descubiertos %d cursos en %s", len(cursos), base_dir.name)
    return cursos
