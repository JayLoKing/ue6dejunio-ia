"""De las notas por criterio a la matriz que entra al modelo.

UNA FILA POR ESTUDIANTE, MATERIA Y TRIMESTRE
--------------------------------------------
El modelo es por materia. Promediar las nueve areas, como se hacia antes,
colapsaba todo a una fila por estudiante-trimestre y tiraba la senal que mas
importa: que un chico este bien en Lenguaje y mal en Matematicas.

LA ETIQUETA VIENE DEL TRIMESTRE SIGUIENTE
------------------------------------------
Esto es lo que separa predecir de describir. El promedio trimestral de un area
**es** la suma de sus cuatro dimensiones —la BDD lo hace explicito con
`total_score GENERATED ALWAYS AS (being + knowing + doing + deciding)`—, asi que
etiquetar con el promedio del mismo trimestre del que salen las features es
pedirle al modelo que aprenda una suma y una tabla de umbrales. Da un numero alto
y no sirve para nada.

Etiquetando con el resultado del trimestre **siguiente** en la misma materia, la
pregunta pasa a ser la que el sistema quiere contestar: *con lo que este chico
lleva hasta ahora en esta materia, como va a terminar*. Las filas del ultimo
trimestre no tienen con que etiquetarse y quedan afuera.

EL VECTOR
---------
Lo arma `contract.expandir`, que es la unica funcion que lo construye — la misma
que usa `serving/api.py`. Calcularlo de dos formas distintas le daria al modelo,
en produccion, features que nunca vio.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import AppConfig
from ..contract import (
    FEATURE_COLS,
    Escala,
    ObservacionMateria,
    escala_de,
    expandir,
    filtrar_en_rango,
)
from ..labeling import aplicar_override_situacion, clasificar_riesgo
from .cleaning import limpiar_df_nombres

logger = logging.getLogger(__name__)

TARGET_COL = "risk_level"

# Identifica una fila: un estudiante, en una materia, en un trimestre.
CLAVES = ["gestion", "grado", "paralelo", "trimestre", "nombre_key", "area"]

__all__ = ["FEATURE_COLS", "TARGET_COL", "CLAVES", "construir_features"]


def _observacion(fila: pd.Series) -> ObservacionMateria:
    return ObservacionMateria(
        being=list(fila["being"]),
        knowing=list(fila["knowing"]),
        doing=list(fila["doing"]),
        deciding=list(fila["deciding"]),
        attendance_pct=fila.get("attendance_pct"),
        criterios_planificados=fila.get("criterios_planificados"),
    )


def _vector(fila: pd.Series, escala: Escala, descartadas: list[tuple[str, float]]) -> dict:
    """El vector de una fila, tolerando la suciedad de las planillas viejas.

    El filtro de rango lo hace `contract.filtrar_en_rango`, la misma funcion que
    define que nota es imposible para la API en vivo. Aca se descarta y se cuenta;
    alla se rechaza el pedido. El criterio es uno solo.
    """
    limpia, fuera = filtrar_en_rango(_observacion(fila), escala)
    descartadas.extend(fuera)
    return expandir(limpia, escala)


def etiquetar(dataset: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Agrega la etiqueta: el riesgo de la misma materia en el trimestre siguiente.

    **Va sobre el dataset completo, no sobre un trimestre suelto.** Se busca por
    (gestion, grado, paralelo, estudiante, area) con trimestre + 1, asi que los tres
    trimestres tienen que estar presentes a la vez; llamarla con uno solo no
    encuentra pareja para ninguna fila.

    Las filas del ultimo trimestre quedan sin etiqueta y es correcto: de ellas no
    hay nada que predecir todavia.
    """
    umbrales = cfg["riesgo"]
    siguiente = dataset[[*CLAVES, "prom_area_trim"]].copy()
    siguiente["trimestre"] = siguiente["trimestre"] - 1
    siguiente = siguiente.rename(columns={"prom_area_trim": "prom_siguiente"})
    # Una fila por clave: si la misma (gestion, grado, paralelo, trimestre,
    # estudiante, area) apareciera dos veces, el merge multiplicaria filas y las
    # etiquetas dejarian de alinearse con el dataset.
    siguiente = siguiente.drop_duplicates(subset=CLAVES)

    unido = dataset.merge(siguiente, on=CLAVES, how="left")
    etiquetas = unido["prom_siguiente"].map(lambda p: clasificar_riesgo(p, umbrales))

    if "situacion" in dataset.columns:
        # 'perdio el año' pesa mas que cualquier promedio parcial.
        etiquetas = [
            aplicar_override_situacion(clase, sit)
            for clase, sit in zip(etiquetas, dataset["situacion"], strict=True)
        ]

    salida = dataset.copy()
    salida[TARGET_COL] = list(etiquetas)
    logger.info(
        "Etiquetado: %d de %d filas tienen resultado del trimestre siguiente",
        int(pd.Series(etiquetas).notna().sum()),
        len(salida),
    )
    return salida


def _situacion_final(df_centralizador: pd.DataFrame) -> dict:
    """Como termino el año cada estudiante, si el centralizador lo dice."""
    if df_centralizador.empty or "situacion" not in df_centralizador.columns:
        return {}
    cen = limpiar_df_nombres(df_centralizador)
    return {
        (fila["gestion"], fila["grado"], fila["paralelo"], fila["nombre_key"]): fila.get(
            "situacion"
        )
        for _, fila in cen.iterrows()
    }


def construir_features(
    df_registro: pd.DataFrame,
    df_centralizador: pd.DataFrame,
    df_asistencia: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    """La matriz final: claves, treinta features y lo que hara falta para etiquetar.

    La etiqueta NO se pone aca: la agrega `etiquetar()` cuando estan los tres
    trimestres juntos, porque mira el siguiente.
    """
    if df_registro.empty:
        logger.warning("Registro vacio: no se pueden construir features")
        return pd.DataFrame()

    reg = limpiar_df_nombres(df_registro)

    if not df_asistencia.empty and "attendance_pct" in df_asistencia.columns:
        asis = limpiar_df_nombres(df_asistencia)
        columnas = [c for c in CLAVES if c != "area"]
        reg = reg.merge(
            asis[[*columnas, "attendance_pct"]].drop_duplicates(subset=columnas),
            on=columnas,
            how="left",
        )
    else:
        # Ausente, no cero: nadie midio esa asistencia.
        reg["attendance_pct"] = None

    situaciones = _situacion_final(df_centralizador)
    reg["situacion"] = [
        situaciones.get((f.gestion, f.grado, f.paralelo, f.nombre_key)) for f in reg.itertuples()
    ]

    # Una sola gestion por llamada, y se verifica en vez de confiarse: hoy el
    # pipeline llama por curso, pero nada lo obliga. Un llamador que concatenara
    # gestiones primero normalizaria las notas de 2023 (Saber sobre 35) contra los
    # topes de 2025 (45). Ninguna fila fallaria; todas quedarian corridas, y
    # `filtrar_en_rango` ademas tiraria como imposibles notas legitimas.
    gestiones = reg["gestion"].unique()
    if len(gestiones) != 1:
        raise ValueError(
            f"construir_features espera una sola gestion por llamada, recibio "
            f"{sorted(gestiones)}: cada una tiene su propia ponderacion"
        )
    escala = escala_de(int(gestiones[0]))
    descartadas: list[tuple[str, float]] = []
    vectores = pd.DataFrame(
        [_vector(fila, escala, descartadas) for _, fila in reg.iterrows()],
        index=reg.index,
    ).astype("float64")  # el mismo tipo que arma `serving/api.py`, o el SavedModel
    #                      queda con una firma que la inferencia no puede llamar
    if descartadas:
        por_dimension: dict[str, int] = {}
        for dim, _ in descartadas:
            por_dimension[dim] = por_dimension.get(dim, 0) + 1
        logger.warning(
            "%d notas fuera del rango de su dimension, descartadas: %s. " "Ejemplos: %s",
            len(descartadas),
            por_dimension,
            descartadas[:5],
        )

    # `prom_area_trim` y `situacion` viajan sin ser features: son de donde saldra
    # la etiqueta, y esa se pone recien cuando estan los tres trimestres juntos.
    salida = pd.concat([reg[[*CLAVES, "prom_area_trim", "situacion"]], vectores], axis=1)
    logger.info(
        "Features: %d filas (estudiante x materia x trimestre), %d columnas de entrada",
        len(salida),
        len(FEATURE_COLS),
    )
    return salida
