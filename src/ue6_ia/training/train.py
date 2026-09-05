"""Entrenamiento y evaluacion del modelo de riesgo academico con TF Decision Forests.

POR QUE LA VALIDACION SE AGRUPA POR ESTUDIANTE
----------------------------------------------
El dataset tiene ~181 filas de ~34 estudiantes: cada chico aparece unas cinco
veces, una por trimestre y gestion. Un `train_test_split` por fila deja al mismo
estudiante de los dos lados del corte, y el modelo lo reconoce en vez de
predecirlo. Asi salio el 0.8378 que figuraba en el `metadata.json` viejo.

`StratifiedGroupKFold` agrupando por estudiante mide lo unico que importa: como
le va con un chico que **nunca vio**. El numero baja, y esa bajada es la verdad
que el corte por fila estaba tapando.

POR QUE VARIOS PLIEGUES Y NO UN HOLDOUT
---------------------------------------
Con 34 estudiantes, quien quede afuera cambia el resultado por completo. Un solo
corte es una anecdota; la media entre pliegues, con su desvio al lado, es una
medicion. El desvio es parte del resultado, no un adorno.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import AppConfig, get_config
from ..evaluation.metricas import resumen_clasificacion, resumir_pliegues
from ..labeling import CLASES
from ..preprocessing.features import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

MODEL_SUBDIR = "tfdf_riesgo"

# El orden de las clases lo fija este modulo, no la libreria.
#
# `make_inspector().label_classes()` devuelve None cuando el label entra como
# texto por `pd_dataframe_to_tf_dataset`: el modelo guarda `num_classes=4` y
# pierde los nombres. Leerlo de ahi es lo que hace `serving/api.py` hoy, y por eso
# revienta con TypeError en cuanto se lo llama.
#
# Codificando el label a entero contra esta lista, el indice `i` de la matriz de
# probabilidades es `CLASES_ORDENADAS[i]` por construccion, y deja de depender de
# un detalle interno que ya cambio una vez.
CLASES_ORDENADAS: list[str] = sorted(CLASES)

# La columna que identifica al estudiante. Es la que agrupa los pliegues.
COL_GRUPO = "nombre_key"

MODELOS = ("gradient_boosted_trees", "random_forest")


def _features_presentes(dataset: pd.DataFrame) -> list[str]:
    """Las features del contrato que el dataset realmente trae.

    Filtrar sin decir nada es la misma degradacion silenciosa que el `or 0.0`: el
    modelo entrena con menos columnas, `metadata.json` guarda el subconjunto como
    si fuera el contrato, y quien sirve arma el vector completo. Nada falla.
    """
    presentes = [c for c in FEATURE_COLS if c in dataset.columns]
    faltantes = [c for c in FEATURE_COLS if c not in dataset.columns]
    if faltantes:
        logger.warning(
            "Features del contrato ausentes en el dataset, se entrena sin ellas: %s",
            faltantes,
        )
    return presentes


def _construir(modelo_tipo: str):
    """Un modelo sin entrenar del tipo pedido."""
    import tensorflow_decision_forests as tfdf

    if modelo_tipo == "random_forest":
        return tfdf.keras.RandomForestModel(verbose=0)
    return tfdf.keras.GradientBoostedTreesModel(verbose=0)


def _codificar(df: pd.DataFrame) -> pd.DataFrame:
    """El label como entero, en el orden que fija `CLASES_ORDENADAS`."""
    codigos = {clase: i for i, clase in enumerate(CLASES_ORDENADAS)}
    codificado = df.copy()
    codificado[TARGET_COL] = codificado[TARGET_COL].map(codigos)
    if codificado[TARGET_COL].isna().any():
        desconocidas = set(df[TARGET_COL]) - set(codigos)
        raise ValueError(f"Etiquetas fuera de CLASES_ORDENADAS: {desconocidas}")
    return codificado


def _ajustar(train_df: pd.DataFrame, modelo_tipo: str):
    import tensorflow_decision_forests as tfdf

    modelo = _construir(modelo_tipo)
    modelo.fit(
        tfdf.keras.pd_dataframe_to_tf_dataset(_codificar(train_df), label=TARGET_COL)
    )
    return modelo


def _predecir(modelo, test_df: pd.DataFrame) -> list[str]:
    import tensorflow_decision_forests as tfdf

    ds = tfdf.keras.pd_dataframe_to_tf_dataset(_codificar(test_df), label=TARGET_COL)
    proba = modelo.predict(ds, verbose=0)
    return [CLASES_ORDENADAS[i] for i in proba.argmax(axis=1)]


def _entrenar_y_predecir(train_df: pd.DataFrame, test_df: pd.DataFrame, modelo_tipo: str):
    """Entrena en `train_df` y devuelve (etiquetas reales, predichas)."""
    modelo = _ajustar(train_df, modelo_tipo)
    return test_df[TARGET_COL].tolist(), _predecir(modelo, test_df)


def validacion_cruzada(
    dataset: pd.DataFrame, modelo_tipo: str, n_pliegues: int = 5, semilla: int = 42
) -> dict:
    """Mide el modelo sobre estudiantes que no vio, pliegue por pliegue."""
    from sklearn.model_selection import StratifiedGroupKFold

    cols = _features_presentes(dataset)
    datos = dataset[cols + [TARGET_COL, COL_GRUPO]].copy()
    clases = CLASES_ORDENADAS

    particion = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    pliegues = []
    for i, (idx_tr, idx_te) in enumerate(
        particion.split(datos, datos[TARGET_COL], groups=datos[COL_GRUPO]), start=1
    ):
        train_df = datos.iloc[idx_tr].drop(columns=[COL_GRUPO])
        test_df = datos.iloc[idx_te].drop(columns=[COL_GRUPO])

        # La garantia que da todo el sentido a esto: ningun estudiante cruza.
        comunes = set(datos.iloc[idx_tr][COL_GRUPO]) & set(datos.iloc[idx_te][COL_GRUPO])
        if comunes:
            raise RuntimeError(f"Pliegue {i}: {len(comunes)} estudiantes en ambos lados")

        y_true, y_pred = _entrenar_y_predecir(train_df, test_df, modelo_tipo)
        medidas = resumen_clasificacion(y_true, y_pred, clases)
        pliegues.append(medidas)
        # La linea base va pegada al accuracy: un 0.84 al lado de una base de 0.79
        # se lee muy distinto que un 0.84 solo, y esta linea es la que mira la gente.
        logger.info(
            "  pliegue %d/%d  n=%3d  acc=%.3f (base %.3f)  macroF1=%s  recallCritico=%s",
            i, n_pliegues, medidas["n"], medidas["accuracy"], medidas["linea_base"],
            _fmt(medidas["macro_f1"]), _fmt(medidas["recall_riesgo_critico"]),
        )

    return {
        "modelo": modelo_tipo,
        "n_pliegues": n_pliegues,
        "resumen": resumir_pliegues(pliegues),
        "pliegues": pliegues,
        "clases": clases,
    }


def _fmt(valor: float | None) -> str:
    return "n/d" if valor is None else f"{valor:.3f}"


def entrenar(dataset: pd.DataFrame, cfg: AppConfig | None = None) -> Path:
    """Compara los modelos con validacion agrupada, entrena el elegido y reporta.

    El modelo final se ajusta sobre TODAS las filas: la validacion cruzada ya dio
    la estimacion honesta de como se comporta con estudiantes nuevos, y guardar el
    de un pliegue seria desperdiciar el resto de los datos.
    """
    cfg = cfg or get_config()
    logger.info("Entrenamiento en CPU (TF Decision Forests no usa GPU).")
    cols = _features_presentes(dataset)
    logger.info("Filas: %d | estudiantes: %d | features: %s",
                len(dataset), dataset[COL_GRUPO].nunique(), cols)
    logger.info("Distribucion de clases:\n%s", dataset[TARGET_COL].value_counts())

    semilla = cfg["entrenamiento"]["semilla"]
    comparacion = {}
    for modelo_tipo in MODELOS:
        logger.info("Validacion cruzada agrupada por estudiante — %s", modelo_tipo)
        comparacion[modelo_tipo] = validacion_cruzada(dataset, modelo_tipo, semilla=semilla)

    elegido = _elegir(comparacion)
    logger.info(
        "Modelo elegido sobre estudiantes no vistos (macro F1, y a igualdad de "
        "macro F1 el que mas casos criticos detecta): %s", elegido
    )

    datos = dataset[cols + [TARGET_COL]].copy()
    modelo = _ajustar(datos, elegido)

    # Control de cordura del mapeo de clases: si el indice de la matriz no
    # correspondiera a CLASES_ORDENADAS, un GBT sobre sus propios datos de
    # entrenamiento no acertaria casi nada.
    acierto_train = sum(
        1 for real, pred in zip(datos[TARGET_COL], _predecir(modelo, datos), strict=True)
        if real == pred
    ) / len(datos)
    logger.info("Acierto sobre los propios datos de entrenamiento: %.3f "
                "(control del mapeo de clases, NO una metrica)", acierto_train)

    out_dir = cfg.models_dir / MODEL_SUBDIR
    modelo.save(str(out_dir))

    meta = {
        "modelo": elegido,
        "features": cols,
        "target": TARGET_COL,
        "n_filas": len(dataset),
        "n_estudiantes": int(dataset[COL_GRUPO].nunique()),
        "distribucion_clases": dataset[TARGET_COL].value_counts().to_dict(),
        # Sin `metricas_holdout`: el numero honesto sale de la validacion agrupada,
        # y guardar ademas uno de un corte por fila invitaria a citar el mas alto.
        "validacion_agrupada": {k: v["resumen"] for k, v in comparacion.items()},
        "acierto_en_entrenamiento": acierto_train,
        "importancia_variables": _importancias(modelo),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "reporte_evaluacion.json").write_text(
        json.dumps(comparacion, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Modelo y reporte guardados en %s", out_dir)
    return out_dir


# Diferencias de macro F1 por debajo de esto se consideran empate. El desvio entre
# pliegues ronda 0.13, asi que separar dos modelos por milesimas seria elegir ruido.
_EMPATE_MACRO_F1 = 0.01


def _elegir(comparacion: dict) -> str:
    """Empata por macro F1 y desempata por a cuantos chicos en riesgo detecta.

    Macro F1 primero y no accuracy: con `RiesgoCritico` en el 6.6% de las filas, el
    accuracy premia al modelo que nunca lo predice.

    Pero macro F1 solo tampoco alcanza. Los dos modelos dan practicamente el mismo
    valor —la diferencia medida fue de 0.001 contra un desvio de 0.13— y elegir por
    esa milesima se quedaba con el que **menos** casos criticos detecta. Cuando la
    diferencia cabe dentro del ruido, decide el objetivo del sistema: avisarle al
    docente antes de que el estudiante repruebe. Un falso negativo es un chico que
    reprueba sin aviso; un falso positivo es una revision de mas.

    El macro F1 sigue actuando de piso, asi que un modelo degenerado que gritara
    `RiesgoCritico` en todas las filas —recall 1.0, precision pesima— no gana.
    """
    def macro(nombre: str) -> float:
        media = comparacion[nombre]["resumen"].get("macro_f1", {}).get("media")
        return media if media is not None else -1.0

    def recall_critico(nombre: str) -> float:
        media = comparacion[nombre]["resumen"].get("recall_riesgo_critico", {}).get("media")
        return media if media is not None else -1.0

    mejor_macro = max(macro(n) for n in comparacion)
    empatados = [n for n in comparacion if mejor_macro - macro(n) <= _EMPATE_MACRO_F1]
    return max(empatados, key=recall_critico)


def _importancias(modelo) -> dict:
    """Que variables pesan en la decision. Es la ventaja de los arboles."""
    try:
        inspector = modelo.make_inspector()
        return {
            clave: [(str(v[0]), float(v[1])) for v in valores[:10]]
            for clave, valores in inspector.variable_importances().items()
        }
    except Exception as e:  # noqa: BLE001 - la importancia es informativa, no un gate
        logger.debug("No se pudo extraer importancia de variables: %s", e)
        return {}
