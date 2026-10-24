"""CLI de pipelines.  Uso:

python -m ue6_ia.cli build-dataset      # Excel -> data/processed/*.parquet
python -m ue6_ia.cli train              # entrena y guarda el modelo (--detalle: log de TF-DF)
python -m ue6_ia.cli metricas           # vuelve a imprimir las tablas del entrenamiento
python -m ue6_ia.cli evaluate           # reporte sobre el dataset
python -m ue6_ia.cli figuras            # laminas del modelo para el documento
python -m ue6_ia.cli all                # build-dataset + train + evaluate
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .config import get_config
from .pipeline import NOMBRE_DATASET, construir_dataset_con_resumen, guardar_dataset

app = typer.Typer(add_completion=False, help="Pipeline de riesgo academico UE6")


def _ruta_dataset() -> Path:
    cfg = get_config()
    return cfg.processed_dir / NOMBRE_DATASET


@app.command("build-dataset")
def build_dataset() -> None:
    """Lee los Excel y construye la matriz de features."""
    from .evaluation.consola import imprimir, tablas_resumen_dataset

    cfg = get_config()
    ds, resumen = construir_dataset_con_resumen(cfg)
    if ds.empty:
        raise typer.Exit(code=1)
    guardar_dataset(ds, cfg)
    imprimir(tablas_resumen_dataset(resumen), "Resumen del procesamiento del dataset")
    typer.echo(f"OK: {len(ds)} filas")


@app.command()
def train(
    detalle: bool = typer.Option(
        False,
        "--detalle",
        help="Muestra el registro de TF-DF iteracion por iteracion al entrenar el modelo final.",
    ),
) -> None:
    """Entrena el modelo TF-DF con el dataset ya construido."""
    from .training.train import entrenar

    path = _ruta_dataset()
    if not path.exists():
        typer.echo("Falta el dataset. Corre primero: build-dataset")
        raise typer.Exit(code=1)
    ds = pd.read_parquet(path)
    out = entrenar(ds, detalle=detalle)
    typer.echo(f"Modelo guardado en {out}")
    _imprimir_metricas(out)


def _imprimir_metricas(modelo_dir: Path) -> None:
    from .evaluation.consola import imprimir, tablas_de_entrenamiento

    imprimir(tablas_de_entrenamiento(modelo_dir), "Resultados del entrenamiento")


@app.command()
def metricas() -> None:
    """Imprime las tablas del ultimo entrenamiento, leidas de models/. No usa TensorFlow."""
    from .training.train import MODEL_SUBDIR

    try:
        _imprimir_metricas(get_config().models_dir / MODEL_SUBDIR)
    except FileNotFoundError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e


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


@app.command()
def figuras(
    salida: Path | None = typer.Option(
        None, "--salida", help="Carpeta destino. Por defecto reports/figures/ del repo."
    ),
    resumen: bool = typer.Option(
        False, "--resumen", help="Ademas, imprime en JSON los numeros que cita el documento."
    ),
) -> None:
    """Dibuja las laminas del modelo para el documento oficial.

    Lee los reportes que dejo el entrenamiento; no carga el modelo, asi que corre
    en Windows sin WSL. Necesita matplotlib, que esta en requirements-dev.txt.
    """
    import json

    from .config import REPO_ROOT
    from .evaluation.graficos import (
        ARCHIVOS_OPCIONALES,
        SUBDIR_FIGURAS,
        generar_figuras,
        resumen_para_documentar,
    )
    from .training.train import MODEL_SUBDIR

    cfg = get_config()
    modelo_dir = cfg.models_dir / MODEL_SUBDIR
    destino = salida or (REPO_ROOT / SUBDIR_FIGURAS)

    try:
        escritas = generar_figuras(modelo_dir, destino)
    except FileNotFoundError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e
    except ModuleNotFoundError as e:
        typer.echo(f"Falta matplotlib: pip install -r requirements-dev.txt ({e})")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        # Reporte presente pero sin nada util adentro: truncado, o de una corrida
        # que no llego a validar. Es tan esperable como el archivo faltante, y el
        # mensaje ya explica que hacer — no merece un traceback.
        typer.echo(str(e))
        raise typer.Exit(code=1) from e

    for path in escritas:
        typer.echo(f"  {path}")
    omitidas = sorted(set(ARCHIVOS_OPCIONALES) - {p.name for p in escritas})
    for nombre in omitidas:
        # El motivo exacto va en el aviso del log; aca, lo que hay que hacer.
        typer.echo(f"  omitida: {nombre} (el modelo no trae su entrada; reentrena con train)")
    typer.echo(f"OK: {len(escritas)} figuras en {destino}")

    if resumen:
        typer.echo(json.dumps(resumen_para_documentar(modelo_dir), indent=2, ensure_ascii=False))


@app.command("all")
def run_all() -> None:
    """Ejecuta build-dataset + train + evaluate en secuencia."""
    build_dataset()
    # Llamado directo, sin Typer de por medio: sin el argumento, `detalle`
    # recibiria el `OptionInfo` del default, que es verdadero.
    train(detalle=False)
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
    uvicorn.run("ue6_ia.serving.api:app", host=cfg.env.api_host, port=cfg.env.api_port)


if __name__ == "__main__":
    app()
