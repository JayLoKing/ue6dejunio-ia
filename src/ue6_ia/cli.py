"""CLI de pipelines.  Uso:

  python -m ue6_ia.cli build-dataset      # Excel -> data/processed/*.parquet
  python -m ue6_ia.cli train              # entrena y guarda el modelo
  python -m ue6_ia.cli evaluate           # reporte sobre el dataset
  python -m ue6_ia.cli all                # build-dataset + train + evaluate
"""

from __future__ import annotations

import logging

import pandas as pd
import typer

from .config import get_config
from .pipeline import construir_dataset, guardar_dataset

app = typer.Typer(add_completion=False, help="Pipeline de riesgo academico UE6")
logger = logging.getLogger(__name__)


def _ruta_dataset():
    from pathlib import Path

    cfg = get_config()
    return Path(cfg["paths"]["processed_dir"]) / "dataset_entrenamiento.parquet"


@app.command("build-dataset")
def build_dataset() -> None:
    """Lee los Excel y construye la matriz de features."""
    cfg = get_config()
    ds = construir_dataset(cfg)
    if ds.empty:
        raise typer.Exit(code=1)
    guardar_dataset(ds, cfg)
    typer.echo(f"OK: {len(ds)} filas")


@app.command()
def train() -> None:
    """Entrena el modelo TF-DF con el dataset ya construido."""
    from .training.train import entrenar

    path = _ruta_dataset()
    if not path.exists():
        typer.echo("Falta el dataset. Corre primero: build-dataset")
        raise typer.Exit(code=1)
    ds = pd.read_parquet(path)
    out = entrenar(ds)
    typer.echo(f"Modelo guardado en {out}")


@app.command()
def evaluate() -> None:
    """Evalua el modelo guardado sobre el dataset completo."""
    from .evaluation.evaluate import guardar_reporte, reporte_clasificacion

    ds = pd.read_parquet(_ruta_dataset())
    rep = reporte_clasificacion(ds)
    out = guardar_reporte(rep)
    typer.echo(f"Reporte guardado en {out}")


@app.command()
def all() -> None:
    """Ejecuta build-dataset + train + evaluate en secuencia."""
    build_dataset()
    train()
    evaluate()


if __name__ == "__main__":
    app()
