"""Carga de configuracion: config.yaml + variables de entorno (.env).

La precedencia es: variable de entorno > config.yaml > default.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz del repo (este archivo esta en src/ue6_ia/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config" / "config.yaml"


class EnvSettings(BaseSettings):
    """Variables de entorno (prefijo UE6_)."""

    # El .env se ancla a la raiz del repo, igual que config.yaml. Relativo al CWD,
    # correr el CLI desde otra carpeta lo hacia desaparecer sin decir nada: el
    # token quedaba vacio y `UE6_DATA_ROOT` caia al valor del yaml.
    model_config = SettingsConfigDict(
        env_prefix="UE6_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    data_root: str | None = None
    models_dir: str = "./models"
    log_level: str = "INFO"
    # Loopback por defecto: el servicio no va expuesto a internet.
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    # Sin valor por defecto a proposito: un placeholder en el repo es un token
    # publico, y compararlo en tiempo constante no protege nada.
    api_token: str = ""


class AppConfig(BaseModel):
    """Configuracion combinada (yaml + entorno) usada por todo el pipeline."""

    raw: dict
    env: EnvSettings

    # --- accesos de conveniencia ---
    @property
    def data_root(self) -> Path:
        root = self.env.data_root or self.raw["paths"]["data_root"]
        return Path(root)

    @property
    def models_dir(self) -> Path:
        """Donde vive el modelo. Leerla no crea nada: crear es tarea de quien escribe."""
        return REPO_ROOT / self.env.models_dir

    @property
    def processed_dir(self) -> Path:
        """Donde se guarda el dataset, resuelto contra la raiz del repo.

        Contra `REPO_ROOT` y no contra el CWD: el yaml lo declara relativo
        (`./data/processed`), asi que corriendo el CLI desde otra carpeta el
        parquet con notas de menores caia en `<CWD>/data/processed`, fuera del
        repo y fuera del `data/` que el .gitignore protege.

        Tampoco cuelga del `data_root`: ese apunta a los registros del colegio,
        que pueden estar en un disco externo o compartido. Lo derivado se queda
        del lado del proyecto, donde ya hay una regla que impide versionarlo.
        """
        return REPO_ROOT / self.raw["paths"]["processed_dir"]

    @property
    def dir_2026(self) -> Path:
        return self.data_root / self.raw["paths"]["registros_2026"]

    @property
    def dir_pasados(self) -> Path:
        return self.data_root / self.raw["paths"]["registros_pasados"]

    @property
    def nota_aprobacion(self) -> int:
        return int(self.raw["nota_aprobacion"])

    def __getitem__(self, key: str) -> object:
        return self.raw[key]


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Devuelve la configuracion (cacheada). Punto de entrada unico."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"No se encontro config: {CONFIG_FILE}")
    with open(CONFIG_FILE, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    env = EnvSettings()
    # permitir override de models_dir desde yaml si no hay env
    # Contra el default del campo y no contra os.environ: `BaseSettings` lee el
    # `.env` hacia el modelo pero NO lo inyecta en el entorno, asi que preguntarle
    # a os.environ daba falso y el yaml pisaba al `.env`. Justo al reves de la
    # precedencia que declara el docstring de este modulo.
    por_defecto = EnvSettings.model_fields["models_dir"].default
    if env.models_dir == por_defecto and raw.get("paths", {}).get("models_dir"):
        env.models_dir = raw["paths"]["models_dir"]
    _setup_logging(env.log_level)
    return AppConfig(raw=raw, env=env)
