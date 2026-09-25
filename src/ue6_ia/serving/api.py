"""Servicio REST de inferencia (FastAPI).

La API Spring Boot (ue6dejunio-api) llama a este servicio para clasificar el
riesgo de un estudiante **en una materia**, y persiste la respuesta en
`risk_predictions`. La direccion es siempre Spring -> FastAPI: aca no hay base de
datos, ni sesion, ni idea de que es un PDC. Es una funcion pura.

Levantar:  uvicorn ue6_ia.serving.api:app --host 127.0.0.1 --port 8001

`127.0.0.1` y no `0.0.0.0`: esto sirve datos academicos de menores y solo lo
llama Spring desde la misma maquina o la red interna. Para esa red, poner la
IP interna concreta, nunca todas las interfaces.

QUE RECIBE
----------
Las cinco variables semanticas del contrato: las notas que el docente ya cargo en
cada dimension, mas el porcentaje de asistencia. Listas, no promedios, porque el
promedio es justo lo que hay que dejar de tirar. `contract.expandir` las convierte
en las treinta columnas que el modelo vio al entrenar — la misma funcion, no una
copia, que es lo unico que evita que el modelo reciba en produccion features que
nunca vio.

QUE DEVUELVE
------------
El estado y **dos probabilidades con nombre**: la de reprobar y la de ser
sobresaliente. Antes devolvia `max(proba)`, la confianza en la clase ganadora, de
modo que un 'Sobresaliente' con 0.95 guardaba 0.95 en la columna que el panel
muestra como riesgo.
"""

from __future__ import annotations

import json
import logging
import secrets
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..config import get_config
from ..contract import (
    FEATURE_COLS,
    ObservacionMateria,
    escala_de,
    expandir,
    puede_predecir,
)
from ..labeling import RIESGO_CRITICO, SOBRESALIENTE
from ..training.train import CLASES_ORDENADAS, MODEL_SUBDIR, cargar_modelo

logger = logging.getLogger(__name__)
app = FastAPI(title="UE6 - Riesgo Academico", version="0.2.0")


class NotasMateria(BaseModel):
    """Lo que el docente lleva cargado de un estudiante en una materia.

    Cada lista trae las notas de esa dimension **ordenadas por fecha**: la feature
    de tendencia es la ultima menos la primera, y sin ese orden es ruido.
    """

    being: list[float] = Field(default_factory=list)
    knowing: list[float] = Field(default_factory=list)
    doing: list[float] = Field(default_factory=list)
    deciding: list[float] = Field(default_factory=list)
    attendance_pct: float | None = Field(None, ge=0, le=100)
    criterios_planificados: int | None = Field(None, ge=0)
    # La ponderacion cambio entre gestiones: un 5 de Being era 33% en 2023 y es
    # 50% hoy. Sin declararla, una planilla vieja entraria normalizada con los
    # topes de hoy y el modelo veria una escala que nunca vio. Ausente significa
    # "la vigente", que es el caso del sistema.
    gestion: int | None = None

    def a_observacion(self) -> ObservacionMateria:
        return ObservacionMateria(
            being=self.being,
            knowing=self.knowing,
            doing=self.doing,
            deciding=self.deciding,
            attendance_pct=self.attendance_pct,
            criterios_planificados=self.criterios_planificados,
        )


class Prediccion(BaseModel):
    """El estado y dos probabilidades que dicen de que son.

    `p_reprueba` es P(RiesgoCritico), y eso **es** reprobar: la categoria cubre
    los promedios de 50 para abajo, y la nota de aprobacion del Ministerio es 51.
    `EnRiesgo` (51 a 66) aprueba raspando, asi que sumarlo aca inflaria la alarma
    y el docente terminaria ignorandola.
    """

    risk_level: str
    p_reprueba: float
    p_sobresaliente: float
    probabilidades: dict[str, float]


class Lote(BaseModel):
    """Varias materias de una vez.

    Una corrida sobre el curso entero son cientos de estudiantes por nueve
    materias; de a una serian miles de viajes HTTP para milisegundos de inferencia.
    """

    # Acotado: un curso entero son cientos de items, no miles. Sin tope, un pedido
    # cualquiera puede pedir una inferencia arbitrariamente grande.
    items: list[NotasMateria] = Field(..., max_length=500)


@lru_cache(maxsize=1)
def _cargar_modelo() -> tuple[Any, list[str]]:
    """El modelo y **las columnas con las que fue entrenado**, una vez por proceso.

    Las columnas se leen de su `metadata.json`, no de `FEATURE_COLS`. Si el dataset
    con el que se entreno no traia alguna del contrato, el entrenamiento avisa y
    sigue con el subconjunto: el modelo quedo esperando esas y solo esas. Armando
    siempre las treinta, el vector deja de ser el que vio — en el mejor caso TF-DF
    rechaza la firma, y en el peor no la rechaza.

    Las clases tampoco salen del modelo: `make_inspector().label_classes()` devuelve
    None con label de texto, y el recargado ni siquiera trae el inspector. El orden
    lo fija `CLASES_ORDENADAS`, que es contra lo que se entreno.
    """
    model_dir = get_config().models_dir / MODEL_SUBDIR
    modelo = cargar_modelo(model_dir)

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"Falta {meta_path}: sin el no se sabe con que columnas se entreno.")
    columnas = json.loads(meta_path.read_text(encoding="utf-8"))["features"]

    faltantes = [c for c in columnas if c not in FEATURE_COLS]
    if faltantes:
        raise RuntimeError(
            f"El modelo espera columnas que el contrato ya no produce: {faltantes}. "
            "Reentrena antes de servir."
        )
    if len(columnas) != len(FEATURE_COLS):
        logger.warning(
            "El modelo se entreno con %d de las %d columnas del contrato: se le "
            "envian solo esas. Reentrenar con el dataset completo.",
            len(columnas),
            len(FEATURE_COLS),
        )
    return modelo, columnas


def _verificar_token(authorization: str = Header(default="")) -> None:
    cfg = get_config()
    token = cfg.env.api_token
    if not token:
        # Sin token configurado el servicio no atiende. La alternativa era un
        # valor por defecto en el repo, que es un secreto publico: compararlo en
        # tiempo constante da la sensacion de proteger y no protege nada, y deja
        # calificaciones de menores detras de una puerta cuya llave esta impresa.
        logger.error("UE6_API_TOKEN sin configurar: el servicio no atiende pedidos")
        raise HTTPException(status_code=503, detail="Servicio sin token configurado")
    # Tiempo constante: comparar con != filtra el token caracter por caracter.
    # Sobre bytes: `compare_digest` con str lanza TypeError si hay algo fuera de
    # ASCII, y Starlette decodifica los headers en latin-1. Un Authorization con
    # basura daba 500 en vez de 401.
    if not secrets.compare_digest(
        authorization.encode("utf-8", "replace"),
        f"Bearer {token}".encode(),
    ):
        raise HTTPException(status_code=401, detail="Token invalido")


def _predecir(observaciones: list[tuple[ObservacionMateria, int | None]]) -> list[Prediccion]:
    import tensorflow_decision_forests as tfdf

    modelo, columnas = _cargar_modelo()
    filas = [expandir(obs, escala_de(gestion)) for obs, gestion in observaciones]
    # float64 explicito, no el que pandas infiera. Un `None` —el `trend` de una
    # dimension con una sola nota— vuelve la columna `object`, TF-DF la lee como
    # texto y el SavedModel rechaza la firma. Con el tipo puesto, `None` es NaN,
    # que es exactamente lo que el modelo vio al entrenar.
    df = pd.DataFrame(filas, columns=columnas).astype("float64")
    proba = modelo.predict(tfdf.keras.pd_dataframe_to_tf_dataset(df), verbose=0)

    predicciones = []
    for valores in proba:
        por_clase = {c: float(p) for c, p in zip(CLASES_ORDENADAS, valores, strict=True)}
        predicciones.append(
            Prediccion(
                risk_level=CLASES_ORDENADAS[int(valores.argmax())],
                p_reprueba=por_clase[RIESGO_CRITICO],
                p_sobresaliente=por_clase[SOBRESALIENTE],
                probabilidades=por_clase,
            )
        )
    return predicciones


def _seguro(
    observaciones: list[tuple[ObservacionMateria, int | None]],
) -> list[Prediccion]:
    """Predice, y convierte una nota imposible en un 422 en vez de un 500.

    `expandir` es estricto a proposito: una nota sobre el tope de su dimension es
    un error de quien llama, no algo que el modelo deba adivinar. Pero sin atrapar
    el `ValueError` eso sale como error interno con traza, cuando en realidad el
    pedido vino mal y el cliente puede arreglarlo.
    """
    try:
        return _predecir(observaciones)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _unica(predicciones: list[Prediccion]) -> Prediccion:
    return predicciones[0]


@app.get("/health")
def health() -> dict:
    try:
        _, columnas = _cargar_modelo()
        return {"status": "ok", "clases": CLASES_ORDENADAS, "features": len(columnas)}
    except Exception as e:  # noqa: BLE001 - health reporta el fallo, no lo propaga
        # El detalle va al log, no a la respuesta: `/health` no pide token, y el
        # error de carga trae la ruta absoluta del modelo o una traza de TF.
        logger.warning("El modelo no se pudo cargar: %s", e)
        return {"status": "sin_modelo"}


@app.post("/predict", response_model=Prediccion)
def predict(notas: NotasMateria, _: None = Depends(_verificar_token)) -> Prediccion:
    if not puede_predecir(notas.a_observacion()):
        # La regla es del docente: sin al menos una nota en cada dimension, la
        # prediccion hablaria de lo que falta cargar y no del estudiante.
        raise HTTPException(
            status_code=422,
            detail="Faltan notas: se predice con al menos una en cada dimension",
        )
    return _unica(_seguro([(notas.a_observacion(), notas.gestion)]))


@app.post("/predict/batch", response_model=list[Prediccion])
def predict_batch(lote: Lote, _: None = Depends(_verificar_token)) -> list[Prediccion]:
    listos = [(n.a_observacion(), n.gestion) for n in lote.items]
    faltantes = [i for i, (obs, _) in enumerate(listos) if not puede_predecir(obs)]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"Items sin nota en alguna dimension: {faltantes}",
        )
    return _seguro(listos)
