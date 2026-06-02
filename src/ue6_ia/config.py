"""Carga de configuracion: config.yaml + variables de entorno (.env).

La precedencia es: variable de entorno > config.yaml > default.
"""

from __future__ import annotations

import logging
import os
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

    model_config = SettingsConfigDict(env_prefix="UE6_", env_file=".env", extra="ignore")

    data_root: str | None = None
    models_dir: str = "./models"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_token: str = "cambia-este-token"


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
        d = REPO_ROOT / self.env.models_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def dir_2026(self) -> Path:
        return self.data_root / self.raw["paths"]["registros_2026"]

    @property
    def dir_pasados(self) -> Path:
        return self.data_root / self.raw["paths"]["registros_pasados"]

    @property
    def nota_aprobacion(self) -> int:
        return int(self.raw["nota_aprobacion"])

    def __getitem__(self, key: str):
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
    if "UE6_MODELS_DIR" not in os.environ and raw.get("paths", {}).get("models_dir"):
        env.models_dir = raw["paths"]["models_dir"]
    _setup_logging(env.log_level)
    return AppConfig(raw=raw, env=env)
