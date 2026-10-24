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

LA PARADA TEMPRANA TAMBIEN SE AGRUPA
------------------------------------
El GBT decide cuantos arboles conservar mirando una perdida de validacion. Por
defecto TF-DF aparta el 10% de las **filas** del entrenamiento para eso, asi que
el mismo chico quedaba en el ajuste y en la parada, y la curva de validacion salia
optimista. Ahora, dentro de cada pliegue, se aparta ~10% de los **estudiantes** de
la porcion de entrenamiento y se le pasa a `fit(validation_data=...)`. Quedan tres
conjuntos sin un estudiante en comun: ajuste, parada temprana y validacion del
pliegue. El Random Forest no para temprano y entrena con la porcion entera.

PONDERACION DE CLASES
---------------------
`RiesgoCritico` es el 4.6% de las filas. Con ponderacion `balanceada` cada fila
pesa `n / (k * n_clase)`, de modo que las cuatro clases suman lo mismo. Los pesos
se calculan **solo** con las filas con que se ajusta (nunca con la validacion del
pliegue) y viajan en una columna de peso que TF-DF no ve como feature. Cada
combinacion de modelo y ponderacion es un candidato, y `_elegir` decide entre
todos con la misma regla.

POR QUE VARIOS PLIEGUES Y NO UN HOLDOUT
---------------------------------------
Con 34 estudiantes, quien quede afuera cambia el resultado por completo. Un solo
corte es una anecdota; la media entre pliegues, con su desvio al lado, es una
medicion. El desvio es parte del resultado, no un adorno.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any  # el modelo es un tf_keras.Model, que no se importa afuera de las funciones

import pandas as pd

from ..config import AppConfig, get_config
from ..contract import FEATURE_COLS
from ..evaluation.metricas import resumen_clasificacion, resumir_pliegues
from ..labeling import CLASES
from ..preprocessing.features import TARGET_COL

logger = logging.getLogger(__name__)

MODEL_SUBDIR = "tfdf_riesgo"

# Dos reportes distintos y dos archivos distintos. Compartian nombre, y como
# `cli all` corre entrenamiento y despues evaluacion, el segundo pisaba al primero:
# sobrevivia el de memorizacion y se perdia el pliegue por pliegue de la validacion
# agrupada, que es justamente la medicion honesta.
REPORTE_VALIDACION = "reporte_validacion_cruzada.json"
REPORTE_EN_MUESTRA = "reporte_en_muestra.json"

# Lo que hace falta para dibujar el entrenamiento y las curvas de confianza sin
# volver a cargar el modelo: las iteraciones del GBT y las probabilidades de cada
# fila, predichas por el pliegue que no la vio. Sin nombres de estudiantes.
REGISTRO_ENTRENAMIENTO = "registro_entrenamiento.json"
PREDICCIONES_FUERA_DE_PLIEGUE = "predicciones_fuera_de_pliegue.parquet"

# La porcion de estudiantes del entrenamiento que se aparta para la parada temprana.
# El mismo 10% que TF-DF usaba por fila, ahora contado en estudiantes.
FRACCION_PARADA = 0.1

# Los modelos que paran temprano y por eso necesitan conjunto de parada.
MODELOS_CON_PARADA = ("gradient_boosted_trees",)

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

# Por si la config no la declara. La lista real sale de `config.yaml`: tenerla solo
# aca dejaba la clave del yaml escrita y sin leer, que es config muerta invitando a
# editarla para nada.
MODELOS_POR_DEFECTO = ("gradient_boosted_trees", "random_forest")

# Sin la clave en la config se compara solo sin ponderar, que es lo que se hacia
# antes de que existiera: un yaml viejo no cambia de comportamiento en silencio.
PONDERACIONES = ("ninguna", "balanceada")
PONDERACION_POR_DEFECTO = ("ninguna",)

# La columna de peso existe solo dentro de `_ajustar`. No es una feature y no
# llega a `metadata.json`, que es de donde el servicio arma el vector. Sin guiones
# bajos al principio: `pd_dataframe_to_tf_dataset` los recorta del nombre y despues
# no encuentra la columna que le pidieron como peso (KeyError '__peso_clase').
COL_PESO = "peso_de_clase"


def candidatos(
    modelos: Sequence[str], ponderaciones: str | Sequence[str]
) -> list[tuple[str, str, str]]:
    """Los candidatos a comparar: (clave, tipo de modelo, ponderacion).

    La clave es estable porque nombra al candidato en los tres reportes: el tipo
    solo si no se pondera, y con `_ponderado` si la ponderacion es balanceada.
    """
    if isinstance(ponderaciones, str):
        ponderaciones = [ponderaciones]
    desconocidas = [x for x in ponderaciones if x not in PONDERACIONES]
    if desconocidas:
        raise ValueError(
            f"ponderacion_clases desconocida: {desconocidas}. Valores validos: {PONDERACIONES}"
        )
    return [
        (tipo if ponderacion == "ninguna" else f"{tipo}_ponderado", tipo, ponderacion)
        for ponderacion in ponderaciones
        for tipo in modelos
    ]


def pesos_balanceados(etiquetas: Sequence[str]) -> dict[str, float]:
    """El peso de cada clase presente: `n / (k * n_clase)`, con k = clases presentes.

    Se llama con las etiquetas de las filas que se ajustan y con ninguna otra.
    Calcularlo sobre el dataset entero metia la distribucion de la validacion del
    pliegue en el entrenamiento.
    """
    if not etiquetas:
        raise ValueError("No hay filas con que calcular los pesos de clase")
    conteo: dict[str, int] = {}
    for etiqueta in etiquetas:
        conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    n, k = len(etiquetas), len(conteo)
    return {clase: n / (k * cuantas) for clase, cuantas in sorted(conteo.items())}


def columna_de_pesos(etiquetas: Sequence[str], pesos: dict[str, float]) -> list[float]:
    """El peso de cada fila. Una clase sin peso calculado pesa 1: no hubo con que medirla."""
    return [float(pesos.get(etiqueta, 1.0)) for etiqueta in etiquetas]


def features_presentes(dataset: pd.DataFrame) -> list[str]:
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


def _construir(modelo_tipo: str, verbose: int = 0) -> Any:
    """Un modelo sin entrenar del tipo pedido.

    `verbose=2` hace que TF-DF imprima su propio registro, iteracion por
    iteracion. Por defecto va callado: con cinco pliegues por dos modelos son
    miles de lineas.
    """
    import tensorflow_decision_forests as tfdf

    if modelo_tipo == "random_forest":
        return tfdf.keras.RandomForestModel(verbose=verbose)
    return tfdf.keras.GradientBoostedTreesModel(verbose=verbose)


def _codificar(df: pd.DataFrame) -> pd.DataFrame:
    """El label como entero, en el orden que fija `CLASES_ORDENADAS`."""
    codigos = {clase: i for i, clase in enumerate(CLASES_ORDENADAS)}
    codificado = df.copy()
    codificado[TARGET_COL] = codificado[TARGET_COL].map(codigos)
    if codificado[TARGET_COL].isna().any():
        desconocidas = set(df[TARGET_COL]) - set(codigos)
        raise ValueError(f"Etiquetas fuera de CLASES_ORDENADAS: {desconocidas}")
    return codificado


def _a_dataset(df: pd.DataFrame, pesos: dict[str, float] | None) -> Any:
    """El dataset de TF-DF, con la columna de peso si se pondera."""
    import tensorflow_decision_forests as tfdf

    codificado = _codificar(df)
    if pesos is None:
        return tfdf.keras.pd_dataframe_to_tf_dataset(codificado, label=TARGET_COL)
    codificado[COL_PESO] = columna_de_pesos(df[TARGET_COL].tolist(), pesos)
    return tfdf.keras.pd_dataframe_to_tf_dataset(codificado, label=TARGET_COL, weight=COL_PESO)


def _ajustar(
    train_df: pd.DataFrame,
    modelo_tipo: str,
    parada_df: pd.DataFrame | None = None,
    verbose: int = 0,
    pesos: dict[str, float] | None = None,
) -> Any:
    """Entrena el modelo; con `parada_df`, el GBT la usa para la parada temprana.

    Pasandole `validation_data`, el GBT deja de apartar filas por su cuenta
    (`validation_ratio`) y mide la parada contra ese conjunto, que se armo por
    estudiante. Con `pesos`, ajuste y parada llevan los mismos pesos por clase:
    medir la parada sin ponderar elegiria el numero de arboles con otra perdida
    que la que se esta minimizando.
    """
    modelo = _construir(modelo_tipo, verbose=verbose)
    datos = _a_dataset(train_df, pesos)
    if parada_df is not None and len(parada_df):
        modelo.fit(datos, validation_data=_a_dataset(parada_df, pesos))
    else:
        modelo.fit(datos)
    return modelo


def cargar_modelo(model_dir: Path) -> Any:
    """El modelo guardado, abierto con la libreria que lo escribio.

    `tf_keras` y NO `tf.keras`: desde TF 2.16 `tf.keras` es Keras 3, que no abre
    el SavedModel de TF-DF y falla con "File format not supported". Vive aca para
    que esa trampa este escrita una sola vez: la usan el evaluador y el servicio.
    """
    import tensorflow_decision_forests as tfdf  # noqa: F401  (registra las ops)
    import tf_keras

    if not model_dir.exists():
        raise RuntimeError(f"Modelo no encontrado en {model_dir}. Entrena primero.")
    return tf_keras.models.load_model(str(model_dir))


def predecir_probabilidades(modelo: Any, test_df: pd.DataFrame) -> Any:
    """La matriz de probabilidades: fila por observacion, columna `i` = `CLASES_ORDENADAS[i]`."""
    import tensorflow_decision_forests as tfdf

    ds = tfdf.keras.pd_dataframe_to_tf_dataset(_codificar(test_df), label=TARGET_COL)
    return modelo.predict(ds, verbose=0)


def predecir(modelo: Any, test_df: pd.DataFrame) -> list[str]:
    """Las clases predichas, nombradas contra `CLASES_ORDENADAS`."""
    proba = predecir_probabilidades(modelo, test_df)
    return [CLASES_ORDENADAS[i] for i in proba.argmax(axis=1)]


def particion_parada_temprana(
    grupos: Sequence[str],
    etiquetas: Sequence[str],
    fraccion: float = FRACCION_PARADA,
    semilla: int = 42,
) -> tuple[list[int], list[int]]:
    """Las posiciones de ajuste y de parada temprana, separadas por estudiante.

    Se toma el primer corte de un `StratifiedGroupKFold` de `1 / fraccion`
    pliegues: aparta ~10% de los estudiantes y, dentro de lo posible, con las
    cuatro clases representadas. Un `GroupShuffleSplit` tambien separa por
    estudiante, pero con 26 chicos puede dejar la parada sin un solo caso critico.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_grupos = len(set(grupos))
    if n_grupos < 2:
        raise ValueError(
            f"Hacen falta al menos 2 estudiantes para apartar una parada temprana, hay {n_grupos}"
        )
    n_cortes = max(2, min(n_grupos, round(1 / fraccion)))
    corte = StratifiedGroupKFold(n_splits=n_cortes, shuffle=True, random_state=semilla)
    ajuste, parada = next(
        corte.split(list(range(len(grupos))), list(etiquetas), groups=list(grupos))
    )
    return [int(i) for i in ajuste], [int(i) for i in parada]


def describir_particion(
    ajuste: Iterable[str],
    parada: Iterable[str],
    validacion: Iterable[str],
    total_filas: int,
    total_estudiantes: int,
) -> dict:
    """Filas y estudiantes de cada conjunto, con su porcentaje del dataset.

    Recibe la columna de estudiante de cada conjunto y **falla** si alguno cae
    en dos: es la garantia de la que depende que la parada temprana y la
    validacion midan estudiantes no vistos.
    """
    conjuntos = {
        "entrenamiento": list(ajuste),
        "parada": list(parada),
        "validacion": list(validacion),
    }
    estudiantes = {lado: set(valores) for lado, valores in conjuntos.items()}
    lados = list(estudiantes)
    for i, uno in enumerate(lados):
        for otro in lados[i + 1 :]:
            comunes = estudiantes[uno] & estudiantes[otro]
            if comunes:
                raise RuntimeError(
                    f"{len(comunes)} estudiantes a la vez en {uno} y en {otro}: "
                    "la particion tiene que ser por estudiante"
                )

    descripcion: dict[str, float | int] = {}
    for lado, valores in conjuntos.items():
        descripcion[f"filas_{lado}"] = len(valores)
        descripcion[f"estudiantes_{lado}"] = len(estudiantes[lado])
        descripcion[f"pct_filas_{lado}"] = 100 * len(valores) / total_filas if total_filas else None
        descripcion[f"pct_estudiantes_{lado}"] = (
            100 * len(estudiantes[lado]) / total_estudiantes if total_estudiantes else None
        )
    return descripcion


def _medido(valor: Any) -> float | None:
    """Un numero real, o `None` si falta o es NaN. Nunca un cero inventado."""
    if valor is None:
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numero) else numero


def _campo(entrada: Any, nombre: str) -> float | None:
    """Un campo escalar de una entrada del registro, distinguiendo ausente de cero.

    En un protobuf un campo opcional que nadie escribio se lee como `0.0`. Si la
    entrada sabe decir si lo tiene (`HasField`), se le pregunta.
    """
    tiene = getattr(entrada, "HasField", None)
    if tiene is not None:
        try:
            if not tiene(nombre):
                return None
        except ValueError:  # campo sin presencia explicita: se lee directo
            pass
    return _medido(getattr(entrada, nombre, None))


def _primero(entrada: Any, nombre: str) -> float | None:
    """El primer valor de un campo repetido (la metrica secundaria es el accuracy)."""
    valores = list(getattr(entrada, nombre, None) or [])
    return _medido(valores[0]) if valores else None


def registro_desde_entradas(entradas: Iterable[Any]) -> list[dict]:
    """Las entradas del registro del GBT como dicts serializables.

    `training_secondary_metrics[0]` y `validation_secondary_metrics[0]` son el
    accuracy en un clasificador.
    """
    return [
        {
            "iteracion": int(getattr(e, "number_of_trees", 0) or 0),
            "training_loss": _campo(e, "training_loss"),
            "training_accuracy": _primero(e, "training_secondary_metrics"),
            "validation_loss": _campo(e, "validation_loss"),
            "validation_accuracy": _primero(e, "validation_secondary_metrics"),
        }
        for e in entradas
    ]


def iteraciones_del_registro(registro: list[dict] | None) -> dict:
    """Hasta donde entreno, y en que iteracion fue minima la perdida de validacion."""
    if not registro:
        return {"iteraciones_entrenadas": None, "iteracion_mejor_validacion": None}
    con_perdida = [e for e in registro if e.get("validation_loss") is not None]
    mejor = min(con_perdida, key=lambda e: e["validation_loss"]) if con_perdida else None
    return {
        "iteraciones_entrenadas": max(e["iteracion"] for e in registro),
        "iteracion_mejor_validacion": None if mejor is None else mejor["iteracion"],
    }


def registro_de_entrenamiento(modelo: Any) -> list[dict] | None:
    """Las iteraciones del GBT, leidas del modelo ya entrenado.

    El Random Forest no las tiene y devuelve `None`. Si la cabecera especializada
    no esta, se cae a `inspector.training_logs()`, que trae solo la validacion.
    """
    try:
        inspector = modelo.make_inspector()
    except Exception as e:  # noqa: BLE001 - el registro es informativo, no un gate
        logger.debug("Sin inspector para leer el registro: %s", e)
        return None
    try:
        entradas = list(inspector.specialized_header().training_logs.entries)
        if entradas:
            return registro_desde_entradas(entradas)
    except Exception as e:  # noqa: BLE001 - un RF no tiene esta cabecera; no es un error
        logger.debug("Sin training_logs en la cabecera especializada: %s", e)
    try:
        logs = inspector.training_logs() or []
    except Exception as e:  # noqa: BLE001 - idem
        logger.debug("Sin training_logs(): %s", e)
        return None
    registro = [
        {
            "iteracion": int(log.num_trees),
            "training_loss": None,
            "training_accuracy": None,
            "validation_loss": _medido(getattr(log.evaluation, "loss", None)),
            "validation_accuracy": _medido(getattr(log.evaluation, "accuracy", None)),
        }
        for log in logs
    ]
    return registro or None


def registro_oob_desde_logs(logs: Iterable[Any]) -> list[dict]:
    """La evaluacion fuera de bolsa del Random Forest a medida que suma arboles.

    TF-DF la registra en algunos puntos (no en cada arbol). Fuera de bolsa quiere
    decir sobre las filas que el bootstrap de cada arbol no tomo: se muestrean
    **filas**, no estudiantes, asi que el mismo chico puede estar dentro y fuera de
    bolsa a la vez. Es una curva de evolucion, no una medicion sobre estudiantes
    no vistos.
    """
    registro = []
    for log in logs:
        evaluacion = getattr(log, "evaluation", None)
        ejemplos = getattr(evaluacion, "num_examples", None)
        registro.append(
            {
                "arboles": int(log.num_trees),
                "oob_accuracy": _medido(getattr(evaluacion, "accuracy", None)),
                "oob_loss": _medido(getattr(evaluacion, "loss", None)),
                "oob_ejemplos": None if ejemplos is None else int(ejemplos),
            }
        )
    return registro


def registro_oob(modelo: Any) -> list[dict] | None:
    """El registro fuera de bolsa de un Random Forest ya entrenado, si lo tiene."""
    try:
        logs = modelo.make_inspector().training_logs() or []
    except Exception as e:  # noqa: BLE001 - el registro es informativo, no un gate
        logger.debug("Sin training_logs() del Random Forest: %s", e)
        return None
    return registro_oob_desde_logs(logs) or None


def _arboles_del_modelo(modelo: Any) -> int | None:
    """Cuantos arboles quedaron en el modelo. En un GBT multiclase, uno por clase e iteracion."""
    try:
        return int(modelo.make_inspector().num_trees())
    except Exception as e:  # noqa: BLE001 - informativo
        logger.debug("No se pudo contar los arboles: %s", e)
        return None


def tabla_predicciones(
    modelo: str,
    pliegue: int,
    filas: Sequence[int],
    reales: Sequence[str],
    proba: Any,
    clases: Sequence[str],
) -> pd.DataFrame:
    """Las predicciones de un pliegue: real, predicha y una columna por clase.

    La columna `i` de `proba` es `clases[i]`, el mismo supuesto que `predecir`
    usa para el argmax. `fila` es la posicion de la observacion en el dataset:
    un indice opaco, no un nombre.
    """
    import numpy as np

    matriz = np.asarray(proba, dtype=float)
    if matriz.ndim != 2 or matriz.shape[1] != len(clases):
        raise ValueError(
            f"La matriz de probabilidades tiene forma {matriz.shape}, se esperaban "
            f"{len(clases)} clases: {list(clases)}"
        )
    tabla = pd.DataFrame(
        {
            "modelo": modelo,
            "pliegue": pliegue,
            "fila": [int(f) for f in filas],
            "real": list(reales),
            "predicha": [clases[i] for i in matriz.argmax(axis=1)],
        }
    )
    for i, clase in enumerate(clases):
        tabla[clase] = matriz[:, i]
    return tabla


def _separar_parada(
    entrenamiento: pd.DataFrame, modelo_tipo: str, semilla: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajuste y parada temprana dentro de una porcion de entrenamiento."""
    if modelo_tipo not in MODELOS_CON_PARADA:
        return entrenamiento, entrenamiento.iloc[[]]
    ajuste, parada = particion_parada_temprana(
        entrenamiento[COL_GRUPO].tolist(), entrenamiento[TARGET_COL].tolist(), semilla=semilla
    )
    return entrenamiento.iloc[ajuste], entrenamiento.iloc[parada]


def _registros_del_ajuste(modelo: Any, modelo_tipo: str) -> dict:
    """Lo que el modelo deja ver de su entrenamiento: iteraciones del GBT u OOB del RF."""
    registro = registro_de_entrenamiento(modelo) if modelo_tipo in MODELOS_CON_PARADA else None
    oob = registro_oob(modelo) if modelo_tipo == "random_forest" else None
    return {
        "registro": registro,
        "registro_oob": oob,
        "iteraciones": {
            **iteraciones_del_registro(registro),
            "arboles_del_modelo": _arboles_del_modelo(modelo),
        },
    }


def validacion_cruzada(
    dataset: pd.DataFrame,
    modelo_tipo: str,
    n_pliegues: int = 5,
    semilla: int = 42,
    ponderacion: str = "ninguna",
    clave: str | None = None,
) -> dict:
    """Mide el modelo sobre estudiantes que no vio, pliegue por pliegue.

    Devuelve el bloque que va a `reporte_validacion_cruzada.json` (con
    `particiones` agregado) y, bajo las claves privadas `_registros` y
    `_predicciones`, lo que va a los otros dos archivos. `entrenar` las saca antes
    de escribir el JSON.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    clave = clave or modelo_tipo
    cols = features_presentes(dataset)
    datos = dataset[cols + [TARGET_COL, COL_GRUPO]].reset_index(drop=True)
    clases = CLASES_ORDENADAS
    total_estudiantes = int(datos[COL_GRUPO].nunique())

    particion = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    pliegues, particiones, registros, predicciones = [], [], [], []
    for i, (idx_tr, idx_te) in enumerate(
        particion.split(datos, datos[TARGET_COL], groups=datos[COL_GRUPO]), start=1
    ):
        validacion = datos.iloc[idx_te]
        ajuste, parada = _separar_parada(datos.iloc[idx_tr], modelo_tipo, semilla)

        # La garantia que da todo el sentido a esto: ningun estudiante cruza entre
        # ajuste, parada temprana y validacion. `describir_particion` falla si no.
        descripcion = describir_particion(
            ajuste[COL_GRUPO],
            parada[COL_GRUPO],
            validacion[COL_GRUPO],
            len(datos),
            total_estudiantes,
        )

        # Los pesos salen de las filas que se ajustan, y de ninguna otra.
        pesos = (
            pesos_balanceados(ajuste[TARGET_COL].tolist()) if ponderacion == "balanceada" else None
        )
        modelo = _ajustar(
            ajuste.drop(columns=[COL_GRUPO]),
            modelo_tipo,
            parada.drop(columns=[COL_GRUPO]),
            pesos=pesos,
        )
        proba = predecir_probabilidades(modelo, validacion.drop(columns=[COL_GRUPO]))
        y_true = validacion[TARGET_COL].tolist()
        tabla = tabla_predicciones(clave, i, validacion.index, y_true, proba, clases)
        medidas = resumen_clasificacion(y_true, tabla["predicha"].tolist(), clases)

        registros_ajuste = _registros_del_ajuste(modelo, modelo_tipo)
        descripcion["iteraciones"] = registros_ajuste["iteraciones"]
        descripcion["pesos_clases"] = pesos
        pliegues.append(medidas)
        particiones.append(descripcion)
        registros.append(
            {
                "pliegue": i,
                "particion": descripcion,
                "registro": registros_ajuste["registro"],
                "registro_oob": registros_ajuste["registro_oob"],
            }
        )
        predicciones.append(tabla)
        # La linea base va pegada al accuracy: un 0.84 al lado de una base de 0.79
        # se lee muy distinto que un 0.84 solo, y esta linea es la que mira la gente.
        logger.info(
            "  pliegue %d/%d  n=%3d  acc=%.3f (base %.3f)  macroF1=%s  recallCritico=%s  iter=%s",
            i,
            n_pliegues,
            medidas["n"],
            medidas["accuracy"],
            medidas["linea_base"],
            _fmt(medidas["macro_f1"]),
            _fmt(medidas["recall_riesgo_critico"]),
            descripcion["iteraciones"]["iteraciones_entrenadas"] or "n/d",
        )

    return {
        "modelo": clave,
        "tipo_modelo": modelo_tipo,
        "ponderacion_clases": ponderacion,
        "n_pliegues": n_pliegues,
        "resumen": resumir_pliegues(pliegues),
        "pliegues": pliegues,
        "particiones": particiones,
        "clases": clases,
        "_registros": registros,
        "_predicciones": pd.concat(predicciones, ignore_index=True),
    }


def _fmt(valor: float | None) -> str:
    return "n/d" if valor is None else f"{valor:.3f}"


def entrenar(dataset: pd.DataFrame, cfg: AppConfig | None = None, detalle: bool = False) -> Path:
    """Compara los modelos con validacion agrupada, entrena el elegido y reporta.

    El modelo final se ajusta sobre TODAS las filas —menos, si es un GBT, el ~10%
    de estudiantes que se aparta para su parada temprana—: la validacion cruzada
    ya dio la estimacion honesta de como se comporta con estudiantes nuevos, y
    guardar el de un pliegue seria desperdiciar el resto de los datos.

    Args:
        detalle: si True, el modelo final se entrena con `verbose=2` y TF-DF
            imprime su registro iteracion por iteracion.
    """
    cfg = cfg or get_config()
    logger.info("Entrenamiento en CPU (TF Decision Forests no usa GPU).")
    cols = features_presentes(dataset)
    logger.info(
        "Filas: %d | estudiantes: %d | features: %s",
        len(dataset),
        dataset[COL_GRUPO].nunique(),
        cols,
    )
    logger.info("Distribucion de clases:\n%s", dataset[TARGET_COL].value_counts())

    entrenamiento = cfg["entrenamiento"]
    semilla = entrenamiento["semilla"]
    pliegues = int(entrenamiento.get("n_pliegues", 5))
    lista = candidatos(
        entrenamiento.get("modelos", MODELOS_POR_DEFECTO),
        entrenamiento.get("ponderacion_clases", PONDERACION_POR_DEFECTO),
    )
    comparacion = {}
    for clave, modelo_tipo, ponderacion in lista:
        logger.info("Validacion cruzada agrupada por estudiante — %s", clave)
        comparacion[clave] = validacion_cruzada(
            dataset,
            modelo_tipo,
            n_pliegues=pliegues,
            semilla=semilla,
            ponderacion=ponderacion,
            clave=clave,
        )

    seleccion = explicar_seleccion(comparacion)
    elegido = seleccion["ganador"]
    tipo_elegido = comparacion[elegido]["tipo_modelo"]
    ponderacion_elegida = comparacion[elegido]["ponderacion_clases"]
    logger.info("Candidato elegido: %s (%s)", elegido, seleccion["criterio"])

    registros = {nombre: bloque.pop("_registros") for nombre, bloque in comparacion.items()}
    predicciones = pd.concat(
        [bloque.pop("_predicciones") for bloque in comparacion.values()], ignore_index=True
    )

    datos = dataset[cols + [TARGET_COL, COL_GRUPO]].reset_index(drop=True)
    ajuste, parada = _separar_parada(datos, tipo_elegido, semilla)
    particion_final = describir_particion(
        ajuste[COL_GRUPO], parada[COL_GRUPO], [], len(datos), int(datos[COL_GRUPO].nunique())
    )
    ajuste = ajuste.drop(columns=[COL_GRUPO])
    logger.info(
        "Modelo final: %d filas de ajuste, %d de parada temprana (%d estudiantes)",
        len(ajuste),
        len(parada),
        particion_final["estudiantes_parada"],
    )
    pesos_final = (
        pesos_balanceados(ajuste[TARGET_COL].tolist())
        if ponderacion_elegida == "balanceada"
        else None
    )
    modelo = _ajustar(
        ajuste,
        tipo_elegido,
        parada.drop(columns=[COL_GRUPO]),
        verbose=2 if detalle else 0,
        pesos=pesos_final,
    )
    registros_final = _registros_del_ajuste(modelo, tipo_elegido)
    iteraciones_final = registros_final["iteraciones"]

    # Control de cordura del mapeo de clases: si el indice de la matriz no
    # correspondiera a CLASES_ORDENADAS, un GBT sobre sus propios datos de
    # entrenamiento no acertaria casi nada. Se mide sobre las filas con que se
    # ajusto; las de la parada temprana no las vio.
    acierto_train = sum(
        1
        for real, pred in zip(ajuste[TARGET_COL], predecir(modelo, ajuste), strict=True)
        if real == pred
    ) / len(ajuste)
    logger.info(
        "Acierto sobre los propios datos de entrenamiento: %.3f "
        "(control del mapeo de clases, NO una metrica)",
        acierto_train,
    )

    out_dir = cfg.models_dir / MODEL_SUBDIR
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(str(out_dir))

    meta = {
        # El tipo base: es lo que nombra al modelo para quien lo sirve. El
        # candidato (con su ponderacion) va aparte.
        "modelo": tipo_elegido,
        "candidato": elegido,
        "ponderacion_clases": ponderacion_elegida,
        "pesos_clases": pesos_final,
        "seleccion": seleccion,
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
        "particion_modelo_final": particion_final,
        "iteraciones_modelo_final": iteraciones_final,
        "fraccion_parada_temprana": FRACCION_PARADA,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / REPORTE_VALIDACION).write_text(
        json.dumps(comparacion, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    registro = {
        "clases": CLASES_ORDENADAS,
        "modelo_elegido": elegido,
        "fraccion_parada_temprana": FRACCION_PARADA,
        "modelos": {
            nombre: {
                "tipo_modelo": comparacion[nombre]["tipo_modelo"],
                "ponderacion_clases": comparacion[nombre]["ponderacion_clases"],
                "pliegues": pliegues,
            }
            for nombre, pliegues in registros.items()
        },
        "modelo_final": {
            "modelo": elegido,
            "tipo_modelo": tipo_elegido,
            "ponderacion_clases": ponderacion_elegida,
            "pesos_clases": pesos_final,
            "particion": particion_final,
            "iteraciones": iteraciones_final,
            "registro": registros_final["registro"],
            "registro_oob": registros_final["registro_oob"],
        },
    }
    (out_dir / REGISTRO_ENTRENAMIENTO).write_text(
        json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    predicciones.to_parquet(out_dir / PREDICCIONES_FUERA_DE_PLIEGUE, index=False)
    logger.info("Modelo y reporte guardados en %s", out_dir)
    return out_dir


def _elegir(comparacion: dict) -> str:
    """Empata por macro F1 y desempata por a cuantos chicos en riesgo detecta.

    Macro F1 primero y no accuracy: con `RiesgoCritico` en el 6.6% de las filas, el
    accuracy premia al modelo que nunca lo predice.

    Pero macro F1 solo tampoco alcanza. Los dos modelos dan practicamente el mismo
    valor y elegir por esa diferencia se quedaba con el que **menos** casos
    criticos detecta. Cuando la diferencia cabe dentro de lo que la metrica se
    mueve entre pliegues, decide el objetivo del sistema: avisarle al docente antes
    de que el estudiante repruebe. Un falso negativo es un chico que reprueba sin
    aviso; un falso positivo es una revision de mas.

    El macro F1 sigue actuando de piso, asi que un modelo degenerado que gritara
    `RiesgoCritico` en todas las filas —recall 1.0, precision pesima— no gana.
    """

    return explicar_seleccion(comparacion)["ganador"]


REGLA_DE_SELECCION = (
    "Gana el mejor macro F1 medio sobre estudiantes no vistos. Si otros candidatos quedan "
    "a menos de la banda de empate (el mayor desvio de macro F1 entre pliegues de todos los "
    "candidatos), decide el mayor recall medio de RiesgoCritico entre los que quedan adentro."
)


def explicar_seleccion(comparacion: dict) -> dict:
    """La regla de `_elegir` aplicada, con cada paso escrito para el documento.

    `_elegir` devuelve el `ganador` de aca: la regla esta escrita una sola vez, y la
    explicacion no puede contar una eleccion distinta de la que se hizo.
    """

    def macro(nombre: str) -> float:
        media = comparacion[nombre]["resumen"].get("macro_f1", {}).get("media")
        return media if media is not None else -1.0

    def desvio_macro(nombre: str) -> float:
        d = comparacion[nombre]["resumen"].get("macro_f1", {}).get("desvio")
        return d if d is not None else 0.0

    def recall_critico(nombre: str) -> float:
        media = comparacion[nombre]["resumen"].get("recall_riesgo_critico", {}).get("media")
        return media if media is not None else -1.0

    def recall_medido(nombre: str) -> dict:
        r = comparacion[nombre]["resumen"].get("recall_riesgo_critico", {})
        return {"media": r.get("media"), "pliegues": r.get("pliegues")}

    # La banda de empate sale de los datos, no de una constante: es cuanto se
    # mueve el propio macro F1 entre pliegues. Una diferencia mas chica que eso no
    # distingue dos modelos, distingue dos repartos de estudiantes. Con un umbral
    # fijo de 0.01 y un desvio de 0.08, una diferencia de 0.013 se tomaba por real
    # y elegia el modelo que detecta la mitad de los casos criticos.
    mejor = max(comparacion, key=macro)
    mejor_macro = macro(mejor)
    fija_banda = max(comparacion, key=desvio_macro)
    banda = desvio_macro(fija_banda)
    empatados = [n for n in comparacion if mejor_macro - macro(n) <= banda]
    ganador = max(empatados, key=recall_critico)

    if len(empatados) == 1:
        criterio = "unico candidato dentro de la banda: gana por macro F1"
    else:
        criterio = (
            f"mayor recall medio de RiesgoCritico entre los {len(empatados)} candidatos "
            "dentro de la banda de empate"
        )
    return {
        "regla": REGLA_DE_SELECCION,
        "mejor_macro_f1": {"candidato": mejor, "valor": mejor_macro},
        "banda": {"ancho": banda, "candidato": fija_banda},
        "dentro_de_la_banda": [
            {
                "candidato": n,
                "macro_f1": macro(n),
                "recall_riesgo_critico": recall_medido(n),
            }
            for n in empatados
        ],
        "ganador": ganador,
        "criterio": criterio,
        "excluidos": [
            {
                "candidato": n,
                "macro_f1": macro(n),
                "motivo": (
                    f"macro F1 {macro(n):.3f} queda {mejor_macro - macro(n):.3f} por debajo del "
                    f"mejor ({mejor_macro:.3f}), fuera de la banda de empate de {banda:.3f}"
                ),
            }
            for n in comparacion
            if n not in empatados
        ],
    }


def _importancias(modelo: Any) -> dict:
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
