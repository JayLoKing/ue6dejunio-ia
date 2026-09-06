"""CLI de pipelines.  Uso:

  python -m ue6_ia.cli build-dataset      # Excel -> data/processed/*.parquet
  python -m ue6_ia.cli train              # entrena y guarda el modelo
  python -m ue6_ia.cli evaluate           # reporte sobre el dataset
  python -m ue6_ia.cli all                # build-dataset + train + evaluate
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .config import get_config
from .pipeline import NOMBRE_DATASET, construir_dataset, guardar_dataset

app = typer.Typer(add_completion=False, help="Pipeline de riesgo academico UE6")


def _ruta_dataset() -> Path:
    cfg = get_config()
    return cfg.processed_dir / NOMBRE_DATASET


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

    path = _ruta_dataset()
    if not path.exists():
        typer.echo("Falta el dataset. Corre primero: build-dataset")
        raise typer.Exit(code=1)
    ds = pd.read_parquet(path)
    rep = reporte_clasificacion(ds)
    out = guardar_reporte(rep)
    typer.echo(f"Reporte guardado en {out}")


@app.command("all")
def run_all() -> None:
    """Ejecuta build-dataset + train + evaluate en secuencia."""
    build_dataset()
    train()
    evaluate()


@app.command()
def serve() -> None:
    """Levanta el servicio de inferencia que consume la API Spring Boot.

    Toma host y puerto de la configuracion (`UE6_API_HOST`, `UE6_API_PORT`). Sin
    este comando esas dos variables estaban declaradas y no las leia nadie: quien
    pusiera `UE6_API_HOST` con la IP interna creia haber cambiado el bind y seguia
    escuchando en loopback.
    """
    import uvicorn

    cfg = get_config()
    if not cfg.env.api_token:
        typer.echo("Falta UE6_API_TOKEN: el servicio rechazaria todo pedido.")
        raise typer.Exit(code=1)
    typer.echo(f"Sirviendo en http://{cfg.env.api_host}:{cfg.env.api_port}")
    uvicorn.run(
        "ue6_ia.serving.api:app", host=cfg.env.api_host, port=cfg.env.api_port
    )


if __name__ == "__main__":
    app()
