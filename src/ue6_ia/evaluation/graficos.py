"""Las laminas que documentan al modelo, y de donde sale cada una.

POR QUE ESTE MODULO NO IMPORTA TENSORFLOW
-----------------------------------------
Lee los JSON que el entrenamiento ya dejo en `models/tfdf_riesgo/` y nada mas.
TF-DF no tiene wheel para Windows, asi que el entrenamiento corre en WSL2; si
esto cargara el modelo, las figuras del documento solo se podrian regenerar
desde alli. Leyendo los reportes se regeneran en cualquier maquina con
matplotlib, que es donde se escribe el documento.

El precio es que lo que no quedo guardado no se puede graficar. Por eso el
entrenamiento persiste, ademas de los JSON, `registro_entrenamiento.json` (la
perdida y el accuracy de cada iteracion del GBT, por pliegue y del modelo final)
y `predicciones_fuera_de_pliegue.parquet` (las probabilidades de cada fila,
predichas por el pliegue que no la vio). Un modelo entrenado antes de que
existieran no los trae: sus laminas se omiten avisando, y las demas salen igual.

QUE MUESTRA CADA LAMINA
-----------------------
1. `matriz_confusion.png` — la suma de los cinco pliegues, es decir fuera de
   muestra. Dice **hacia donde** se equivoca: confundir `EnRiesgo` con
   `RiesgoCritico` es un error muy distinto a confundirlo con `Sobresaliente`.
2. `distribucion_clases.png` — el desbalance del dataset. Es la lamina que
   justifica por que el accuracy solo no alcanza.
3. `accuracy_vs_linea_base.png` — pliegue por pliegue, contra lo que sacaria un
   modelo que siempre responde la clase mas frecuente. Sin esa referencia un
   0.63 no se puede leer. Incluye el acierto en entrenamiento como tercera
   marca: la distancia entre las dos es cuanto memorizo.
4. `metricas_por_clase.png` — precision, recall y F1 de cada clase fuera de
   muestra. El recall de `RiesgoCritico` es LA metrica: un falso negativo es un
   chico que reprueba sin que nadie avisara.
5. `dispersion_pliegues.png` — cuanto se mueve cada metrica segun a quien le
   toque quedar afuera. Con 33 estudiantes esto no es un detalle: una media alta
   con desvio del mismo tamano no es un resultado confiable.
6. `importancia_variables.png` — que mira el modelo para decidir. Es la ventaja
   de los arboles sobre una red: se puede responder esa pregunta.
7. `matriz_confusion_normalizada.png` — la misma matriz en porcentaje de cada
   clase real. Con `RiesgoCritico` en 50 filas y las otras en ~350, los conteos
   esconden que la clase chica es la peor detectada.
8. `resultados_entrenamiento.png` — perdida y accuracy contra el numero de
   iteraciones, en entrenamiento y en el conjunto de parada temprana. Necesita
   `registro_entrenamiento.json`.
   Para un Random Forest, la evaluacion fuera de bolsa contra el numero de
   arboles.
9. `curva_precision_confianza.png` y 10. `curva_recall_confianza.png` — que pasa
   con la precision y el recall de cada clase al exigir mas confianza. Necesitan
   `predicciones_fuera_de_pliegue.parquet`.
11. `comparacion_candidatos.png` — macro F1 y recall critico de cada candidato,
   con la banda de empate: por que gano el que gano.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

from .metricas import RIESGO_CRITICO, promedios_por_clase, resumen_clasificacion

logger = logging.getLogger(__name__)

REPORTE_VALIDACION = "reporte_validacion_cruzada.json"
METADATA = "metadata.json"
REGISTRO_ENTRENAMIENTO = "registro_entrenamiento.json"
PREDICCIONES_FUERA_DE_PLIEGUE = "predicciones_fuera_de_pliegue.parquet"

# `.gitignore` ya reserva esta ruta: las figuras son artefactos regenerables y no
# se versionan, igual que el modelo y el dataset.
SUBDIR_FIGURAS = Path("reports") / "figures"

# La importancia por defecto. TF-DF publica varias; esta ordena por que tan
# arriba del arbol aparece la variable, que es la que se lee como "cuanto pesa
# en la decision" sin tener que explicar la mecanica de los splits.
IMPORTANCIA_POR_DEFECTO = "INV_MEAN_MIN_DEPTH"

ARCHIVOS_ESPERADOS = (
    "matriz_confusion.png",
    "distribucion_clases.png",
    "accuracy_vs_linea_base.png",
    "metricas_por_clase.png",
    "dispersion_pliegues.png",
    "importancia_variables.png",
    # Sale de la misma matriz acumulada que la primera, asi que nunca falta.
    "matriz_confusion_normalizada.png",
    # Sale del reporte de validacion, que siempre esta.
    "comparacion_candidatos.png",
)

# Las que dependen de archivos que un modelo viejo no trae. Faltando su entrada se
# omiten con un aviso; no voltean a las demas.
ARCHIVOS_OPCIONALES = (
    "resultados_entrenamiento.png",
    "curva_precision_confianza.png",
    "curva_recall_confianza.png",
)

# Los nombres de las clases como se leen en el documento. El identificador sigue
# siendo el de `labeling.CLASES`; esto es solo lo que se escribe en la lamina.
ETIQUETA_CLASE = {
    "RiesgoCritico": "Riesgo crítico",
    "EnRiesgo": "En riesgo",
    "SinRiesgo": "Sin riesgo",
    "Sobresaliente": "Sobresaliente",
}

# De la mas grave a la mejor. Las tablas y la matriz normalizada se leen en este
# orden, no en el alfabetico que indexa al modelo.
ORDEN_SEVERIDAD = ("RiesgoCritico", "EnRiesgo", "SinRiesgo", "Sobresaliente")

# Cuantos umbrales de confianza recorre cada curva, de 0 a 1 inclusive.
PUNTOS_DE_UMBRAL = 101

# Un color por clase, estable en todas las laminas: la misma clase tiene que ser
# del mismo color en la matriz y en las barras, o los graficos se leen mal al
# ponerlos uno al lado del otro en el documento.
COLOR_POR_CLASE = {
    "RiesgoCritico": "#b2182b",
    "EnRiesgo": "#ef8a62",
    "SinRiesgo": "#67a9cf",
    "Sobresaliente": "#2166ac",
}
COLOR_NEUTRO = "#777777"


# --------------------------------------------------------------------------
# Lectura de los reportes. Todo lo de aqui es puro: entra un dict, sale un dato.
# --------------------------------------------------------------------------


def nombre_de_feature(crudo: str) -> str:
    """El nombre de la variable, sin la contabilidad interna de TF-DF.

    El inspector devuelve `'"knowing_mean" (1; #26)'`: el nombre, su tipo y su
    indice en la tabla interna. Eso es ruido en el eje de un grafico.
    """
    limpio = crudo.split('" (')[0]
    return limpio.strip().strip('"')


def matriz_confusion_acumulada(bloque: dict) -> list[list[int]]:
    """La matriz de los pliegues sumada celda por celda.

    Cada pliegue evalua estudiantes que el modelo no vio, y juntos cubren el
    dataset entero: la suma es la matriz **fuera de muestra**. La de
    `reporte_en_muestra.json` se calcula sobre las filas de entrenamiento y mide
    memorizacion, asi que no es la que va en el documento.
    """
    pliegues = bloque.get("pliegues") or []
    sumados, total = pliegues_sumados(bloque)
    if sumados < total:
        # No se puede recuperar lo que el pliegue no dejo escrito, pero callarlo
        # convierte una cobertura parcial en una que se lee como completa: la
        # lamina sale titulada igual y el documento la cita como el dataset
        # entero fuera de muestra. Misma razon por la que `cajas_de_dispersion`
        # rotula sus pliegues parciales.
        logger.warning(
            "La matriz acumulada suma %d de %d pliegues: los demas no trajeron "
            "matriz_confusion, asi que no cubre el dataset entero",
            sumados,
            total,
        )
    acumulada: list[list[int]] = []
    for pliegue in pliegues:
        matriz = pliegue.get("matriz_confusion")
        if not matriz:
            continue
        if not acumulada:
            acumulada = [list(fila) for fila in matriz]
            continue
        for i, fila in enumerate(matriz):
            for j, n in enumerate(fila):
                acumulada[i][j] += n
    return acumulada


def pliegues_sumados(bloque: dict) -> tuple[int, int]:
    """Cuantos pliegues entraron en la matriz acumulada, y cuantos hay.

    Separado de la suma para que la diferencia se pueda afirmar en un test y
    rotular en la lamina, en vez de quedar como un `continue` que nadie ve.
    """
    pliegues = bloque.get("pliegues") or []
    return sum(1 for p in pliegues if p.get("matriz_confusion")), len(pliegues)


def expandir_matriz(matriz: list[list[int]], clases: list[str]) -> tuple[list[str], list[str]]:
    """Rehace las listas de verdad y prediccion que produjeron la matriz.

    La celda `[i][j]` cuenta los casos cuya clase real es `clases[i]` y que el
    modelo llamo `clases[j]`, asi que la matriz y las listas dicen exactamente lo
    mismo. Volver a las listas permite medir con `metricas.py` en vez de
    reimplementar precision y recall aqui — y con ellas, su regla de que una
    clase sin casos reales no mide cero sino nada.

    Los nombres y la matriz pueden llegar de fuentes distintas: las clases del
    bloque del modelo y la matriz de la suma de los pliegues. Si no cuadran, esto
    falla en vez de indexar igual, porque indexar igual publicaria el recall de
    una clase bajo el nombre de otra.
    """
    if any(len(fila) != len(clases) for fila in matriz) or len(matriz) != len(clases):
        raise ValueError(
            f"La matriz de {len(matriz)} filas no coincide con las {len(clases)} clases "
            f"declaradas: {clases}"
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    for i, fila in enumerate(matriz):
        for j, n in enumerate(fila):
            y_true.extend([clases[i]] * n)
            y_pred.extend([clases[j]] * n)
    return y_true, y_pred


def metricas_por_clase(matriz: list[list[int]], clases: list[str]) -> dict:
    """Precision, recall y F1 de cada clase, derivados de la matriz."""
    y_true, y_pred = expandir_matriz(matriz, clases)
    return resumen_clasificacion(y_true, y_pred, clases)


def serie_por_pliegue(bloque: dict, metrica: str) -> list[float | None]:
    """El valor de una metrica en cada pliegue, en orden y con sus huecos.

    Un `None` se conserva: el pliegue que no trajo un solo caso de la clase no
    midio mal, no midio. Reemplazarlo por cero hundiria la serie y el grafico
    contaria una caida que nunca ocurrio.
    """
    return [p.get(metrica) for p in bloque.get("pliegues") or []]


def puntos_medidos(valores: list[float | None]) -> tuple[list[int], list[float]]:
    """Los indices y valores que se pueden dibujar. Los `None` no entran.

    Separado en su propia funcion porque es una afirmacion sobre el modelo, no
    sobre el dibujo: **un pliegue que no midio no es un pliegue que midio cero**.
    Una barra al ras es indistinguible de un modelo que erro todos los casos, y
    la lamina estaria afirmando una medicion que nunca se hizo. Un cero real si
    entra: ese midio, y midio mal.
    """
    indices = [i for i, v in enumerate(valores) if v is not None]
    return indices, [valores[i] for i in indices]  # type: ignore[misc]


def importancias_ordenadas(
    metadata: dict, metrica: str = IMPORTANCIA_POR_DEFECTO, tope: int = 12
) -> list[tuple[str, float]]:
    """Las variables que mas pesan, de mayor a menor, con el nombre limpio.

    Devuelve vacio cuando la metrica no esta: el entrenamiento se traga la
    excepcion al extraer la importancia porque es informativa, de modo que un
    modelo sin ella es un caso previsto y no puede voltear la generacion entera.
    """
    crudas = (metadata.get("importancia_variables") or {}).get(metrica) or []
    pares = [(nombre_de_feature(nombre), float(valor)) for nombre, valor in crudas]
    pares.sort(key=lambda par: par[1], reverse=True)
    return pares[:tope]


def candidato_elegido(metadata: dict) -> str | None:
    """La clave del candidato que gano: `candidato`, o `modelo` en un metadata viejo.

    Desde que se compara con y sin ponderacion, `modelo` guarda el tipo base (lo
    que entiende quien sirve) y la clave del reporte va en `candidato`.
    """
    return metadata.get("candidato") or metadata.get("modelo")


def resumen_de_candidatos(reporte: dict, metadata: dict) -> dict:
    """Macro F1 y recall critico de cada candidato, el ganador y la banda de empate.

    La banda es `[mejor macro F1 - ancho, mejor macro F1]`. Se toma de la
    `seleccion` que guardo el entrenamiento; si el metadata es anterior, se
    recalcula con la misma regla (el mayor desvio de macro F1).
    """
    ganador = candidato_elegido(metadata)
    filas = []
    for clave, bloque in reporte.items():
        resumen = bloque.get("resumen") or {}
        filas.append(
            {
                "candidato": clave,
                "ganador": clave == ganador,
                "macro_f1": resumen.get("macro_f1") or {},
                "recall_riesgo_critico": resumen.get("recall_riesgo_critico") or {},
            }
        )
    seleccion = metadata.get("seleccion") or {}
    mejor = (seleccion.get("mejor_macro_f1") or {}).get("valor")
    ancho = (seleccion.get("banda") or {}).get("ancho")
    if mejor is None or ancho is None:
        medias = [
            f["macro_f1"].get("media") for f in filas if f["macro_f1"].get("media") is not None
        ]
        desvios = [
            f["macro_f1"].get("desvio") for f in filas if f["macro_f1"].get("desvio") is not None
        ]
        mejor = max(medias) if medias else None
        # Sin ningun desvio medido no hay banda que dibujar: un ancho cero la
        # achicaria a un punto y la lamina afirmaria que no hubo empate.
        ancho = max(desvios) if desvios else None
    banda = None if mejor is None or ancho is None else (mejor - ancho, mejor)
    return {"candidatos": filas, "banda": banda}


def _bloque_del_modelo_elegido(reporte: dict, metadata: dict) -> dict:
    """El bloque del modelo que gano la comparacion.

    El reporte guarda un bloque por modelo candidato. Graficar el que aparece
    primero mostraria un modelo que quiza no es el que quedo sirviendo.
    """
    elegido = candidato_elegido(metadata)
    if elegido and elegido in reporte:
        return reporte[elegido]
    primero = next(iter(reporte.values()), {})
    logger.warning(
        "metadata.json no nombra un modelo presente en el reporte; se grafica %s",
        primero.get("modelo", "?"),
    )
    return primero


def clases_del_bloque(bloque: dict, metadata: dict) -> list[str]:
    """Los nombres de las clases, en el orden en que esta indexada la matriz.

    El reporte los guarda junto al modelo, que es la fuente correcta: el orden de
    la matriz de confusion es ese. Caer al `distribucion_clases` del metadata es
    un ultimo recurso para reportes de una version vieja, y **se avisa**: ese
    orden es alfabetico por construccion y coincide solo porque
    `CLASES_ORDENADAS = sorted(CLASES)`, un detalle que ya podria no valer.

    Sin nombres por ningun lado se falla. Dibujar igual dejaria una lamina
    titulada "por que el accuracy solo no alcanza" sobre un eje en blanco, y el
    documento la citaria como prueba del desbalance.
    """
    del_bloque = bloque.get("clases")
    if del_bloque:
        return list(del_bloque)

    del_metadata = sorted(metadata.get("distribucion_clases") or {})
    if del_metadata:
        logger.warning(
            "El reporte de validacion no declara 'clases'; se usa el orden alfabetico de "
            "distribucion_clases (%s). Verifica que coincida con la matriz de confusion.",
            del_metadata,
        )
        return del_metadata

    raise ValueError(
        "No hay nombres de clases ni en el reporte de validacion ni en metadata.json: "
        "no se puede rotular la matriz de confusion. Reentrena para regenerar los reportes."
    )


def etiqueta_de_clase(clase: str) -> str:
    """El nombre legible de la clase; una clase desconocida queda como vino."""
    return ETIQUETA_CLASE.get(clase, clase)


def ordenar_por_severidad(clases: list[str]) -> list[str]:
    """Las clases de la mas grave a la mejor; las que no conoce, al final y en su orden."""
    conocidas = [c for c in ORDEN_SEVERIDAD if c in clases]
    return conocidas + [c for c in clases if c not in ORDEN_SEVERIDAD]


def reordenar_matriz(
    matriz: list[list[int]], clases: list[str], orden: list[str] | tuple[str, ...]
) -> tuple[list[list[int]], list[str]]:
    """La misma matriz con filas y columnas permutadas a la vez.

    Permutar solo las filas (o solo los rotulos) dejaria la diagonal fuera de la
    diagonal: los aciertos se leerian como confusiones.
    """
    nuevas = [c for c in orden if c in clases] + [c for c in clases if c not in orden]
    indice = [clases.index(c) for c in nuevas]
    return [[matriz[i][j] for j in indice] for i in indice], nuevas


def matriz_normalizada_por_fila(matriz: list[list[int]]) -> list[list[float | None]]:
    """Cada celda como porcentaje de su clase real.

    Una clase sin casos reales no tiene porcentaje: su fila queda en `None`, no en
    cero. Un cero diria que el modelo no acerto ninguno de sus casos, y no tuvo
    ninguno que acertar.
    """
    normalizada: list[list[float | None]] = []
    for fila in matriz:
        total = sum(fila)
        normalizada.append([100 * n / total if total else None for n in fila])
    return normalizada


def umbrales_de_confianza(puntos: int = PUNTOS_DE_UMBRAL) -> list[float]:
    """Los umbrales de 0 a 1, equiespaciados. Como fraccion exacta: `i / 100`."""
    return [i / (puntos - 1) for i in range(puntos)]


def curvas_por_umbral(
    reales: list[str],
    probas: dict[str, list[float]],
    clases: list[str],
    umbrales: list[float],
) -> dict[str, dict[str, list[float]]]:
    """Precision y recall de cada clase, uno contra el resto, a cada umbral.

    A un umbral `t`, la clase `c` "se predice" en las filas con `P(c|x) >= t`:

    - precision_c(t) = aciertos / filas que pasan el umbral;
    - recall_c(t) = aciertos / casos reales de `c`.

    Donde ninguna fila pasa el umbral la precision **no existe** y vale NaN: no
    hubo ninguna prediccion que pudiera estar bien o mal. Dibujarla como cero
    contaria una caida que no ocurrio, y es la regla de `puntos_medidos` llevada a
    las curvas. Por la misma razon una clase sin casos reales no tiene recall.

    Esto no es la regla de decision del modelo —que elige la clase de mayor
    probabilidad—: con umbrales bajos una fila cuenta para varias clases a la vez.
    Es la pregunta "si solo aviso cuando estoy al menos asi de seguro, cuanto
    acierto y cuantos se me escapan", que es la que la lamina contesta.
    """
    import numpy as np

    reales_arr = np.asarray(reales)
    curvas: dict[str, dict[str, list[float]]] = {"precision": {}, "recall": {}}
    for clase in clases:
        proba = np.asarray(probas[clase], dtype=float)
        positivos = reales_arr == clase
        n_positivos = int(positivos.sum())
        precision, recall = [], []
        for umbral in umbrales:
            pasa = proba >= umbral
            n_pasa = int(pasa.sum())
            aciertos = int((pasa & positivos).sum())
            precision.append(aciertos / n_pasa if n_pasa else math.nan)
            recall.append(aciertos / n_positivos if n_positivos else math.nan)
        curvas["precision"][clase] = precision
        curvas["recall"][clase] = recall
    return curvas


def media_entre_clases(series: dict[str, list[float]]) -> list[float]:
    """La media de las clases en cada punto, solo donde **todas** midieron.

    Donde una clase se corta, la media tambien (NaN). Promediar las que quedan
    hacia saltar la linea: al salir `RiesgoCritico` —la peor— la media subia de
    golpe, y eso se leia como una mejora del modelo cuando era una clase menos en
    la cuenta.
    """
    largo = len(next(iter(series.values()), []))
    medias = []
    for k in range(largo):
        valores = [v[k] for v in series.values()]
        if not valores or any(math.isnan(x) for x in valores):
            medias.append(math.nan)
        else:
            medias.append(sum(valores) / len(valores))
    return medias


def _a_float(valor: float | None) -> float:
    return math.nan if valor is None else float(valor)


def serie_de_entrenamiento(registro: list[dict], metrica: str) -> tuple[list[int], list[float]]:
    """Las iteraciones y los valores de una metrica, con los `None` como hueco (NaN).

    matplotlib corta la linea en un NaN. Un `None` pasado a cero la haria caer a
    una perdida perfecta en la iteracion que no se registro.
    """
    return (
        [int(e["iteracion"]) for e in registro],
        [_a_float(e.get(metrica)) for e in registro],
    )


def leer_registro_entrenamiento(modelo_dir: Path) -> dict | None:
    """El registro de iteraciones, o `None` si el modelo es de antes de que existiera."""
    path = modelo_dir / REGISTRO_ENTRENAMIENTO
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def leer_predicciones(modelo_dir: Path, modelo: str) -> Any:
    """Las predicciones fuera de pliegue de un modelo, o `None` si no estan.

    Devuelve un `DataFrame`; pandas se importa aca para que leer los JSON no lo
    necesite.
    """
    path = modelo_dir / PREDICCIONES_FUERA_DE_PLIEGUE
    if not path.exists():
        return None
    import pandas as pd

    tabla = pd.read_parquet(path)
    if "modelo" in tabla.columns:
        tabla = tabla[tabla["modelo"] == modelo].reset_index(drop=True)
    return tabla if len(tabla) else None


def _registros_del_modelo(
    registro: dict | None, modelo: str, clave: str = "registro"
) -> tuple[list[dict], list[list[dict]]]:
    """El registro del modelo final y los de cada pliegue, solo los que tienen datos.

    `clave` elige cual: `registro` (iteraciones del GBT) o `registro_oob` (la
    evaluacion fuera de bolsa del Random Forest).
    """
    if not registro:
        return [], []
    final = registro.get("modelo_final") or {}
    del_final = (final.get(clave) or []) if final.get("modelo") == modelo else []
    pliegues = ((registro.get("modelos") or {}).get(modelo) or {}).get("pliegues") or []
    return del_final, [p[clave] for p in pliegues if p.get(clave)]


def serie_oob(registro: list[dict], metrica: str) -> tuple[list[int], list[float]]:
    """Arboles y valor fuera de bolsa, con los `None` como hueco."""
    return [int(e["arboles"]) for e in registro], [_a_float(e.get(metrica)) for e in registro]


def _curvas_del_modelo(
    predicciones: Any, clases: list[str]
) -> tuple[dict[str, dict[str, list[float]]], list[float]] | None:
    if predicciones is None:
        return None
    faltan = [c for c in clases if c not in predicciones.columns]
    if faltan:
        logger.warning("Las predicciones fuera de pliegue no traen la probabilidad de %s", faltan)
        return None
    umbrales = umbrales_de_confianza()
    probas = {c: predicciones[c].astype(float).tolist() for c in clases}
    return curvas_por_umbral(predicciones["real"].tolist(), probas, clases, umbrales), umbrales


def _leer_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path.name} en {path.parent}. Corre primero: python -m ue6_ia.cli train"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------


def _colores(clases: list[str]) -> list[str]:
    return [COLOR_POR_CLASE.get(c, COLOR_NEUTRO) for c in clases]


def _lamina_matriz_confusion(plt, matriz: list[list[int]], clases: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matriz, cmap="Blues")

    ax.set_xticks(range(len(clases)), clases, rotation=30, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion fuera de muestra\n(suma de los pliegues)")

    # El numero va escrito en cada celda: el color solo da el orden de magnitud y
    # el documento necesita el conteo exacto.
    mayor = max((n for fila in matriz for n in fila), default=0)
    for i, fila in enumerate(matriz):
        for j, n in enumerate(fila):
            ax.text(
                j, i, str(n), ha="center", va="center", color="white" if n > mayor / 2 else "black"
            )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_distribucion(plt, distribucion: dict, out: Path) -> None:
    clases = sorted(distribucion, key=lambda c: distribucion[c], reverse=True)
    conteos = [distribucion[c] for c in clases]
    total = sum(conteos)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if total:
        barras = ax.bar(clases, conteos, color=_colores(clases))
        ax.set_ylabel("Filas del dataset")
        for barra, n in zip(barras, conteos, strict=True):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                n,
                f"{n}\n({n / total:.1%})",
                ha="center",
                va="bottom",
            )
        ax.margins(y=0.18)
    else:
        # Sin conteos no hay desbalance que mostrar, y el titulo de esta lamina
        # es una afirmacion sobre el dataset: dibujar el eje vacio debajo de el
        # lo convierte en una afirmacion sin respaldo que el documento cita
        # igual. Se emite la lamina — un hueco en la numeracion seria peor — y
        # dice en su cara que el metadata no trajo los conteos. Mismo trato que
        # `_lamina_importancias` le da a la importancia que no se guardo.
        #
        # El `or 1` que estaba en el divisor tapaba esto: repartia porcentajes
        # sobre un total inventado en vez de avisar que no habia total.
        logger.warning(
            "El metadata no trae distribucion_clases: la lamina sale diciendolo, sin barras"
        )
        ax.text(
            0.5,
            0.5,
            "El metadata no guardo la distribucion de clases",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLOR_NEUTRO,
        )
        ax.set_axis_off()
    ax.set_title("Distribucion de clases: por que el accuracy solo no alcanza")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _barras_medidas(
    ax, valores: list[float | None], corrimiento: float, ancho: float, etiqueta: str, color: str
) -> None:
    """Las barras de los pliegues que midieron, y un `n/d` donde no hubo medicion."""
    indices, medidos = puntos_medidos(valores)
    ax.bar([i + corrimiento for i in indices], medidos, ancho, label=etiqueta, color=color)
    for i, v in enumerate(valores):
        if v is None:
            ax.text(
                i + corrimiento,
                0.02,
                "n/d",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color=COLOR_NEUTRO,
            )


def _lamina_accuracy(plt, bloque: dict, en_entrenamiento: float | None, out: Path) -> None:
    accuracy = serie_por_pliegue(bloque, "accuracy")
    linea_base = serie_por_pliegue(bloque, "linea_base")
    etiquetas = [f"P{i + 1}" for i in range(len(accuracy))]
    x = range(len(accuracy))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _barras_medidas(ax, accuracy, -ancho / 2, ancho, "Accuracy del modelo", "#2166ac")
    _barras_medidas(
        ax, linea_base, ancho / 2, ancho, "Linea base (clase mas frecuente)", COLOR_NEUTRO
    )

    if en_entrenamiento is not None:
        # La distancia entre esta linea y las barras azules es la memorizacion.
        ax.axhline(
            en_entrenamiento,
            color="#b2182b",
            linestyle="--",
            label=f"Acierto en entrenamiento ({en_entrenamiento:.2f})",
        )

    ax.set_xticks(list(x), etiquetas)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporcion de aciertos")
    ax.set_title("Accuracy por pliegue contra su linea base")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_por_clase(plt, medidas: dict, clases: list[str], out: Path) -> None:
    nombres = ["precision", "recall", "f1"]
    x = range(len(clases))
    ancho = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, nombre in enumerate(nombres):
        # Una clase sin casos reales da `None`, y `metricas._por_clase` devuelve
        # un F1 de 0.0 genuino cuando el modelo fallo del todo. Si el hueco se
        # dibujara como cero, las dos cosas quedarian como la misma barra.
        valores = [medidas["por_clase"][c][nombre] for c in clases]
        _barras_medidas(ax, valores, (k - 1) * ancho, ancho, nombre.capitalize(), f"C{k}")

    ax.set_xticks(list(x), clases, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Precision, recall y F1 por clase (fuera de muestra)")
    ax.legend(fontsize=8)

    # Lo que manda no es el promedio: es cuantos chicos en riesgo real detecta.
    if RIESGO_CRITICO in clases:
        ax.get_xticklabels()[clases.index(RIESGO_CRITICO)].set_color("#b2182b")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def cajas_de_dispersion(bloque: dict, metricas: list[str]) -> dict[str, list[float]]:
    """Cada metrica dibujable con su etiqueta, diciendo cuantos pliegues la sostienen.

    Una caja armada con cuatro de cinco pliegues se dibuja igual de firme que
    una armada con los cinco, y ahi esta el problema: el pliegue que falta no
    midio peor, no midio — no trajo un solo caso de la clase — y el grafico no
    tenia como decirlo. El aviso al pie solo aparecia cuando la metrica no habia
    medido en ningun pliegue, que es el caso que menos confunde, porque entonces
    la caja directamente no esta.

    Asi que la cuenta va en la etiqueta de la caja, donde se lee junto con ella.
    Una metrica completa no lleva aclaracion: decir "2 de 2" en todas las cajas
    seria ruido en el caso normal y haria menos visible el caso raro.

    Se conserva el orden en que se pidieron las metricas, y no el del reporte,
    para que dos corridas del mismo modelo produzcan la misma lamina.
    """
    total = len(bloque.get("pliegues") or [])
    cajas: dict[str, list[float]] = {}
    for metrica in metricas:
        medidos = [v for v in serie_por_pliegue(bloque, metrica) if v is not None]
        if not medidos:
            continue
        etiqueta = metrica
        if len(medidos) < total:
            etiqueta = f"{metrica}\n({len(medidos)} de {total} pliegues)"
        cajas[etiqueta] = medidos
    return cajas


def _lamina_dispersion(plt, bloque: dict, out: Path) -> None:
    metricas = ["accuracy", "macro_f1", "recall_riesgo_critico", "linea_base"]
    con_datos = cajas_de_dispersion(bloque, metricas)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not con_datos:
        # Pliegues con matriz pero sin una sola metrica medida: un reporte de
        # una version que las nombraba distinto, o una corrida cortada despues
        # de escribir la matriz. La guarda de `generar_figuras` no lo ve, porque
        # la matriz esta, y `boxplot([])` contesta "Dimensions of labels and X
        # must be compatible" — el mismo tipo de mensaje que esa guarda existe
        # para no mostrarle a nadie. Se emite la lamina diciendolo, igual que
        # hacen la de importancias y la de distribucion.
        logger.warning(
            "Ningun pliegue midio alguna de las metricas %s: la lamina de "
            "dispersion sale sin cajas",
            metricas,
        )
        ax.text(
            0.5,
            0.5,
            "Ningun pliegue midio estas metricas",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLOR_NEUTRO,
        )
        ax.set_axis_off()
    else:
        ax.boxplot(list(con_datos.values()), tick_labels=list(con_datos), showmeans=True)
        # Los puntos encima de la caja: con cinco pliegues el resumen de cinco
        # numeros esconde mas de lo que muestra.
        for i, valores in enumerate(con_datos.values(), start=1):
            ax.plot([i] * len(valores), valores, "o", color="#2166ac", alpha=0.6, markersize=5)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Valor en cada pliegue")
        ax.tick_params(axis="x", labelsize=8)

        dibujadas = {e.split("\n")[0] for e in con_datos}
        faltantes = [m for m in metricas if m not in dibujadas]
        if faltantes:
            ax.text(
                0.5,
                0.02,
                f"Sin datos en ningun pliegue: {', '.join(faltantes)}",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                color=COLOR_NEUTRO,
            )

    ax.set_title("Cuanto se mueve cada metrica segun quien quede afuera")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_importancias(plt, pares: list[tuple[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if pares:
        nombres = [n for n, _ in pares][::-1]
        valores = [v for _, v in pares][::-1]
        ax.barh(nombres, valores, color="#2166ac")
        ax.set_xlabel(f"Importancia ({IMPORTANCIA_POR_DEFECTO})")
    else:
        # El entrenamiento pudo no haberla extraido. La lamina se emite igual y
        # lo dice: un hueco en la numeracion del documento seria peor.
        ax.text(
            0.5,
            0.5,
            "El entrenamiento no guardo la importancia de variables",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLOR_NEUTRO,
        )
        ax.set_axis_off()
    ax.set_title("Que mira el modelo para decidir")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_matriz_normalizada(plt, matriz: list[list[int]], clases: list[str], out: Path) -> None:
    conteos, orden = reordenar_matriz(matriz, clases, ORDEN_SEVERIDAD)
    porcentajes = matriz_normalizada_por_fila(conteos)
    imagen = [[math.nan if v is None else v for v in fila] for fila in porcentajes]
    etiquetas = [etiqueta_de_clase(c) for c in orden]

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    mapa = ax.imshow(imagen, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(mapa, ax=ax, fraction=0.046, pad=0.04, label="% de la clase real")

    ax.set_xticks(range(len(orden)), etiquetas, rotation=30, ha="right")
    ax.set_yticks(range(len(orden)), etiquetas)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(
        "Matriz de confusión normalizada por clase real\n(fuera de muestra, suma de los pliegues)"
    )

    for i, fila in enumerate(porcentajes):
        for j, pct in enumerate(fila):
            texto = "n/d" if pct is None else f"{pct:.1f}%\n({conteos[i][j]})"
            claro = pct is not None and pct > 50
            ax.text(j, i, texto, ha="center", va="center", color="white" if claro else "black")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


_PANELES_ENTRENAMIENTO = (
    ("training_loss", "Pérdida en entrenamiento", "Pérdida (log loss)"),
    ("validation_loss", "Pérdida en el conjunto de parada temprana", "Pérdida (log loss)"),
    ("training_accuracy", "Accuracy en entrenamiento", "Accuracy"),
    ("validation_accuracy", "Accuracy en el conjunto de parada temprana", "Accuracy"),
)


def _lamina_resultados_entrenamiento(
    plt, final: list[dict], pliegues: list[list[dict]], iteraciones: dict, modelo: str, out: Path
) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    detencion = iteraciones.get("iteraciones_entrenadas")
    mejor = iteraciones.get("iteracion_mejor_validacion")

    for eje, (metrica, titulo, eje_y) in zip(ejes.flat, _PANELES_ENTRENAMIENTO, strict=True):
        for k, registro in enumerate(pliegues):
            x, y = serie_de_entrenamiento(registro, metrica)
            eje.plot(
                x,
                y,
                color=COLOR_NEUTRO,
                linewidth=0.9,
                alpha=0.45,
                label="Pliegues de la validación cruzada" if k == 0 else None,
            )
        if final:
            x, y = serie_de_entrenamiento(final, metrica)
            eje.plot(x, y, color="#2166ac", linewidth=2.2, label="Modelo final")
        if detencion is not None:
            eje.axvline(
                detencion,
                color="#b2182b",
                linestyle="--",
                linewidth=1,
                label=f"Detención del modelo final ({detencion})",
            )
        if mejor is not None and mejor != detencion:
            eje.axvline(
                mejor,
                color="#b2182b",
                linestyle=":",
                linewidth=1,
                label=f"Mejor pérdida de validación ({mejor})",
            )
        eje.set_title(titulo, fontsize=10)
        eje.set_ylabel(eje_y)
        eje.grid(alpha=0.3)
    for eje in ejes[1]:
        eje.set_xlabel("Iteración (árboles por clase)")
    ejes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Resultados del entrenamiento: {modelo}")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_resultados_oob(
    plt, final: list[dict], pliegues: list[list[dict]], modelo: str, out: Path
) -> None:
    hay_perdida = any(e.get("oob_loss") is not None for r in [final, *pliegues] for e in r)
    paneles = [("oob_accuracy", "Accuracy fuera de bolsa (OOB)", "Accuracy OOB")]
    if hay_perdida:
        paneles.insert(0, ("oob_loss", "Pérdida fuera de bolsa (OOB)", "Pérdida OOB (log loss)"))
    fig, ejes = plt.subplots(1, len(paneles), figsize=(5.5 * len(paneles), 4.8), squeeze=False)

    for eje, (metrica, titulo, eje_y) in zip(ejes[0], paneles, strict=True):
        for k, registro in enumerate(pliegues):
            x, y = serie_oob(registro, metrica)
            eje.plot(
                x,
                y,
                color=COLOR_NEUTRO,
                linewidth=0.9,
                alpha=0.45,
                marker=".",
                label="Pliegues de la validación cruzada" if k == 0 else None,
            )
        if final:
            x, y = serie_oob(final, metrica)
            eje.plot(x, y, color="#2166ac", linewidth=2.2, marker="o", label="Modelo final")
        eje.set_title(titulo, fontsize=10)
        eje.set_xlabel("Número de árboles")
        eje.set_ylabel(eje_y)
        eje.grid(alpha=0.3)
    ejes[0][0].legend(fontsize=7)
    fig.suptitle(
        f"Evolución del entrenamiento fuera de bolsa: {modelo}\n"
        "(OOB muestrea filas, no estudiantes: no es una medición sobre estudiantes no vistos)",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_comparacion(plt, resumen: dict, out: Path) -> None:
    filas = resumen["candidatos"]
    nombres = [f["candidato"] for f in filas]
    y = list(range(len(filas)))[::-1]
    paneles = (
        ("macro_f1", "Macro F1 (media ± desvío entre pliegues)"),
        ("recall_riesgo_critico", "Recall de Riesgo crítico (media ± desvío)"),
    )
    fig, ejes = plt.subplots(1, 2, figsize=(11, 0.9 * len(filas) + 2.2), sharey=True)
    for eje, (metrica, titulo) in zip(ejes, paneles, strict=True):
        if metrica == "macro_f1" and resumen["banda"] is not None:
            bajo, alto = resumen["banda"]
            eje.axvspan(bajo, alto, color="#fddbc7", alpha=0.6, label="Banda de empate")
        for fila, posicion in zip(filas, y, strict=True):
            medida = fila[metrica]
            media = medida.get("media")
            if media is None:
                eje.text(0.02, posicion, "n/d", va="center", color=COLOR_NEUTRO)
                continue
            color = "#b2182b" if fila["ganador"] else "#2166ac"
            desvio = medida.get("desvio")
            eje.errorbar(
                media,
                posicion,
                # Sin desvio medido no hay barra: una de ancho cero se leeria
                # como una medicion estable. Y la barra no baja de cero, porque
                # un recall negativo no existe.
                xerr=None if desvio is None else [[min(desvio, media)], [desvio]],
                fmt="D" if fila["ganador"] else "o",
                color=color,
                capsize=4,
                markersize=8 if fila["ganador"] else 6,
            )
            texto = f"{media:.3f}"
            if medida.get("pliegues") is not None and metrica == "recall_riesgo_critico":
                texto += f" ({medida['pliegues']} pliegues)"
            if desvio is None:
                texto += " (sin desvío)"
            eje.annotate(
                texto,
                (media, posicion),
                textcoords="offset points",
                xytext=(0, 9),
                ha="left",
                fontsize=8,
            )
        eje.set_xlim(0, 1)
        eje.set_title(titulo, fontsize=10)
        eje.grid(axis="x", alpha=0.3)
    ejes[0].set_yticks(
        y,
        [
            f"{n}{'  ← ganador' if f['ganador'] else ''}"
            for n, f in zip(nombres, filas, strict=True)
        ],
    )
    ejes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Comparación de candidatos sobre estudiantes no vistos", fontsize=11)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_curva_confianza(
    plt,
    curvas: dict[str, list[float]],
    umbrales: list[float],
    clases: list[str],
    metrica: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for clase in ordenar_por_severidad(clases):
        ax.plot(
            umbrales,
            curvas[clase],
            color=COLOR_POR_CLASE.get(clase, COLOR_NEUTRO),
            linewidth=1.3,
            label=etiqueta_de_clase(clase),
        )

    media = media_entre_clases(curvas)
    rotulo = "Todas las clases"
    if metrica == "Precisión":
        medidos = [(v, t) for v, t in zip(media, umbrales, strict=True) if not math.isnan(v)]
        if medidos:
            valor, umbral = max(medidos)
            rotulo += f" ({valor:.2f} con confianza {umbral:.2f})"
    elif media and not math.isnan(media[0]):
        rotulo += f" ({media[0]:.2f} con confianza 0.00)"
    ax.plot(umbrales, media, color="#222222", linewidth=3, label=rotulo)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confianza (probabilidad asignada a la clase)")
    ax.set_ylabel(metrica)
    ax.set_title(f"Curva {metrica.lower()}–confianza (fuera de muestra)")
    ax.grid(alpha=0.3)
    # Esquinas sin curvas: la precision arranca abajo a la izquierda y la de la
    # clase critica cae a cero abajo a la derecha; el recall baja hacia la derecha.
    ax.legend(fontsize=8, loc="upper left" if metrica == "Precisión" else "upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def generar_figuras(modelo_dir: Path, salida: Path) -> list[Path]:
    """Escribe las laminas a partir de los reportes de `modelo_dir`.

    Las ocho de `ARCHIVOS_ESPERADOS` salen siempre. Las de `ARCHIVOS_OPCIONALES`
    salen si el entrenamiento dejo su entrada, y si no se omiten con un aviso: un
    modelo entrenado antes de que se persistieran no puede voltear las demas.

    Todo o nada, de verdad: las que se dibujan van a una carpeta temporal y recien
    al terminar se mueven a destino. Escribiendo directo, una falla en la cuarta
    dejaria tres figuras nuevas junto a tres viejas, y el documento citaria
    graficos de dos corridas distintas creyendo que son la misma.
    """
    reporte = _leer_json(modelo_dir / REPORTE_VALIDACION)
    metadata = _leer_json(modelo_dir / METADATA)

    # Backend sin ventana: esto corre en un CLI y en CI, donde no hay pantalla
    # que abrir. Se fija antes de importar pyplot o matplotlib elige por su
    # cuenta y falla sin display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bloque = _bloque_del_modelo_elegido(reporte, metadata)
    clases = clases_del_bloque(bloque, metadata)
    matriz = matriz_confusion_acumulada(bloque)

    # Un reporte truncado deja el bloque sin un solo pliegue, y entonces no hay
    # nada fuera de muestra que graficar. Se corta aca, antes de dibujar y antes
    # de tocar el destino, porque la primera lamina llama a `imshow([])` y
    # matplotlib contesta con un TypeError que no nombra ni el archivo ni la
    # causa. El error tiene que decir que falta el reporte, no como se llama la
    # funcion que se cayo.
    if not matriz:
        raise ValueError(
            f"El reporte de {modelo_dir} no trae un solo pliegue con matriz de confusion: "
            f"no hay resultados fuera de muestra que graficar. Volve a correr la validacion "
            f"cruzada antes de generar las figuras."
        )

    elegido = candidato_elegido(metadata) or bloque.get("modelo") or ""
    registro = leer_registro_entrenamiento(modelo_dir)
    final, de_pliegues = _registros_del_modelo(registro, elegido)
    final_oob, de_pliegues_oob = _registros_del_modelo(registro, elegido, "registro_oob")
    iteraciones = ((registro or {}).get("modelo_final") or {}).get("iteraciones") or {}
    curvas = _curvas_del_modelo(leer_predicciones(modelo_dir, elegido), clases)
    hay_evolucion = bool(final or de_pliegues or final_oob or de_pliegues_oob)
    if not hay_evolucion:
        logger.warning(
            "Se omite resultados_entrenamiento.png: %s no trae iteraciones de %s "
            "(falta el archivo, o el modelo no registra iteraciones). Reentrena con train.",
            REGISTRO_ENTRENAMIENTO,
            elegido,
        )
    if curvas is None:
        logger.warning(
            "Se omiten las curvas de confianza: falta %s con las probabilidades de %s. "
            "Reentrena con train.",
            PREDICCIONES_FUERA_DE_PLIEGUE,
            elegido,
        )

    salida.mkdir(parents=True, exist_ok=True)

    # La temporal cuelga de la propia salida para caer en el mismo sistema de
    # archivos: asi el movimiento final es un rename y no una copia a medias.
    with tempfile.TemporaryDirectory(dir=salida) as tmp:
        taller = Path(tmp)
        borrador = {nombre: taller / nombre for nombre in ARCHIVOS_ESPERADOS}
        if hay_evolucion:
            borrador["resultados_entrenamiento.png"] = taller / "resultados_entrenamiento.png"
        if curvas is not None:
            for nombre in ("curva_precision_confianza.png", "curva_recall_confianza.png"):
                borrador[nombre] = taller / nombre

        _lamina_matriz_confusion(plt, matriz, clases, borrador["matriz_confusion.png"])
        _lamina_distribucion(
            plt, metadata.get("distribucion_clases") or {}, borrador["distribucion_clases.png"]
        )
        _lamina_accuracy(
            plt,
            bloque,
            metadata.get("acierto_en_entrenamiento"),
            borrador["accuracy_vs_linea_base.png"],
        )
        _lamina_por_clase(
            plt, metricas_por_clase(matriz, clases), clases, borrador["metricas_por_clase.png"]
        )
        _lamina_dispersion(plt, bloque, borrador["dispersion_pliegues.png"])
        _lamina_importancias(
            plt, importancias_ordenadas(metadata), borrador["importancia_variables.png"]
        )
        _lamina_matriz_normalizada(
            plt, matriz, clases, borrador["matriz_confusion_normalizada.png"]
        )
        _lamina_comparacion(
            plt, resumen_de_candidatos(reporte, metadata), borrador["comparacion_candidatos.png"]
        )
        if final or de_pliegues:
            _lamina_resultados_entrenamiento(
                plt,
                final,
                de_pliegues,
                iteraciones,
                elegido,
                borrador["resultados_entrenamiento.png"],
            )
        elif hay_evolucion:
            _lamina_resultados_oob(
                plt, final_oob, de_pliegues_oob, elegido, borrador["resultados_entrenamiento.png"]
            )
        if curvas is not None:
            por_umbral, umbrales = curvas
            _lamina_curva_confianza(
                plt,
                por_umbral["precision"],
                umbrales,
                clases,
                "Precisión",
                borrador["curva_precision_confianza.png"],
            )
            _lamina_curva_confianza(
                plt,
                por_umbral["recall"],
                umbrales,
                clases,
                "Recall",
                borrador["curva_recall_confianza.png"],
            )

        escritas = []
        for nombre, origen in borrador.items():
            destino = salida / nombre
            origen.replace(destino)
            escritas.append(destino)

    logger.info("Escritas %d figuras en %s", len(escritas), salida)
    return escritas


def resumen_para_documentar(modelo_dir: Path) -> dict[str, Any]:
    """Los numeros que el documento cita al pie de cada lamina.

    Se devuelven ya calculados para que el texto no tenga que volver a abrir los
    JSON y arriesgarse a citar una corrida distinta de la que se grafico.
    """
    reporte = _leer_json(modelo_dir / REPORTE_VALIDACION)
    metadata = _leer_json(modelo_dir / METADATA)
    bloque = _bloque_del_modelo_elegido(reporte, metadata)
    clases = clases_del_bloque(bloque, metadata)
    matriz = matriz_confusion_acumulada(bloque)
    medidas = metricas_por_clase(matriz, clases)
    conteos, orden = reordenar_matriz(matriz, clases, ORDEN_SEVERIDAD)
    elegido = candidato_elegido(metadata) or bloque.get("modelo") or ""
    registro = leer_registro_entrenamiento(modelo_dir) or {}
    final = dict(registro.get("modelo_final") or {})
    ultima = (final.pop("registro", None) or [None])[-1]
    ultima_oob = (final.pop("registro_oob", None) or [None])[-1]
    curvas = _curvas_del_modelo(leer_predicciones(modelo_dir, elegido), clases)

    candidatos = resumen_de_candidatos(reporte, metadata)
    return {
        "modelo": metadata.get("modelo"),
        "candidato": elegido,
        "ponderacion_clases": metadata.get("ponderacion_clases"),
        "seleccion": metadata.get("seleccion"),
        "candidatos": {
            c["candidato"]: {
                m: (reporte[c["candidato"]].get("resumen") or {}).get(m)
                for m in ("accuracy", "linea_base", "macro_f1", "recall_riesgo_critico")
            }
            for c in candidatos["candidatos"]
        },
        "n_filas": metadata.get("n_filas"),
        "n_estudiantes": metadata.get("n_estudiantes"),
        "n_pliegues": bloque.get("n_pliegues"),
        "resumen_pliegues": bloque.get("resumen"),
        "acierto_en_entrenamiento": metadata.get("acierto_en_entrenamiento"),
        "fuera_de_muestra": {
            "accuracy": medidas["accuracy"],
            "macro_f1": medidas["macro_f1"],
            "recall_riesgo_critico": medidas["recall_riesgo_critico"],
            "linea_base": medidas["linea_base"],
            "por_clase": medidas["por_clase"],
            "promedios": promedios_por_clase(medidas["por_clase"]),
            "matriz": {
                "clases": orden,
                "conteos": conteos,
                "porcentaje_por_fila": matriz_normalizada_por_fila(conteos),
            },
        },
        "particiones_por_pliegue": bloque.get("particiones"),
        "modelo_final": (
            {**final, "ultima_iteracion": ultima, "ultimo_punto_oob": ultima_oob} if final else None
        ),
        "curvas_confianza": _resumen_de_curvas(curvas, clases),
    }


def _resumen_de_curvas(curvas: tuple | None, clases: list[str]) -> dict | None:
    """Los puntos de las curvas que el texto cita: el umbral 0.5 y la mejor precision."""
    if curvas is None:
        return None
    por_umbral, umbrales = curvas
    # El mas cercano a 0.5 y no `index(0.5)`: con una cantidad par de puntos
    # el 0.5 exacto no esta en la grilla.
    medio = min(range(len(umbrales)), key=lambda i: abs(umbrales[i] - 0.5))

    def punto(valor: float) -> float | None:
        return None if math.isnan(valor) else round(valor, 4)

    resumen: dict[str, dict] = {}
    series = {c: (por_umbral["precision"][c], por_umbral["recall"][c]) for c in clases}
    series["todas_las_clases"] = (
        media_entre_clases(por_umbral["precision"]),
        media_entre_clases(por_umbral["recall"]),
    )
    for nombre, (precision, recall) in series.items():
        medidos = [(v, t) for v, t in zip(precision, umbrales, strict=True) if not math.isnan(v)]
        mejor = max(medidos) if medidos else None
        resumen[nombre] = {
            "precision_en_0.5": punto(precision[medio]),
            "recall_en_0.5": punto(recall[medio]),
            "recall_en_0": punto(recall[0]),
            "precision_maxima": None if mejor is None else round(mejor[0], 4),
            "umbral_de_precision_maxima": None if mejor is None else mejor[1],
        }
    return resumen
