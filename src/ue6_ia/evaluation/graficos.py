"""Las laminas que documentan al modelo, y de donde sale cada una.

POR QUE ESTE MODULO NO IMPORTA TENSORFLOW
-----------------------------------------
Lee los JSON que el entrenamiento ya dejo en `models/tfdf_riesgo/` y nada mas.
TF-DF no tiene wheel para Windows, asi que el entrenamiento corre en WSL2; si
esto cargara el modelo, las figuras del documento solo se podrian regenerar
desde alli. Leyendo los reportes se regeneran en cualquier maquina con
matplotlib, que es donde se escribe el documento.

El precio es que lo que no quedo guardado no se puede graficar: la curva de
entrenamiento del GBT (logloss contra numero de arboles) vive en
`make_inspector().training_logs()` y el entrenamiento no la persiste, asi que no
hay lamina para ella.

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
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from .metricas import RIESGO_CRITICO, resumen_clasificacion

logger = logging.getLogger(__name__)

REPORTE_VALIDACION = "reporte_validacion_cruzada.json"
METADATA = "metadata.json"

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
)

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


def _bloque_del_modelo_elegido(reporte: dict, metadata: dict) -> dict:
    """El bloque del modelo que gano la comparacion.

    El reporte guarda un bloque por modelo candidato. Graficar el que aparece
    primero mostraria un modelo que quiza no es el que quedo sirviendo.
    """
    elegido = metadata.get("modelo")
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
            ax.text(j, i, str(n), ha="center", va="center",
                    color="white" if n > mayor / 2 else "black")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _lamina_distribucion(plt, distribucion: dict, out: Path) -> None:
    clases = sorted(distribucion, key=lambda c: distribucion[c], reverse=True)
    conteos = [distribucion[c] for c in clases]
    total = sum(conteos) or 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(clases, conteos, color=_colores(clases))
    ax.set_ylabel("Filas del dataset")
    ax.set_title("Distribucion de clases: por que el accuracy solo no alcanza")
    for barra, n in zip(barras, conteos, strict=True):
        ax.text(barra.get_x() + barra.get_width() / 2, n,
                f"{n}\n({n / total:.1%})", ha="center", va="bottom")
    ax.margins(y=0.18)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _barras_medidas(ax, valores: list[float | None], corrimiento: float, ancho: float,
                    etiqueta: str, color: str) -> None:
    """Las barras de los pliegues que midieron, y un `n/d` donde no hubo medicion."""
    indices, medidos = puntos_medidos(valores)
    ax.bar([i + corrimiento for i in indices], medidos, ancho, label=etiqueta, color=color)
    for i, v in enumerate(valores):
        if v is None:
            ax.text(i + corrimiento, 0.02, "n/d", ha="center", va="bottom",
                    fontsize=7, rotation=90, color=COLOR_NEUTRO)


def _lamina_accuracy(plt, bloque: dict, en_entrenamiento: float | None, out: Path) -> None:
    accuracy = serie_por_pliegue(bloque, "accuracy")
    linea_base = serie_por_pliegue(bloque, "linea_base")
    etiquetas = [f"P{i + 1}" for i in range(len(accuracy))]
    x = range(len(accuracy))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _barras_medidas(ax, accuracy, -ancho / 2, ancho, "Accuracy del modelo", "#2166ac")
    _barras_medidas(ax, linea_base, ancho / 2, ancho,
                    "Linea base (clase mas frecuente)", COLOR_NEUTRO)

    if en_entrenamiento is not None:
        # La distancia entre esta linea y las barras azules es la memorizacion.
        ax.axhline(en_entrenamiento, color="#b2182b", linestyle="--",
                   label=f"Acierto en entrenamiento ({en_entrenamiento:.2f})")

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
        _barras_medidas(ax, valores, (k - 1) * ancho, ancho, nombre.capitalize(),
                        f"C{k}")

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


def _lamina_dispersion(plt, bloque: dict, out: Path) -> None:
    metricas = ["accuracy", "macro_f1", "recall_riesgo_critico", "linea_base"]
    series = {m: [v for v in serie_por_pliegue(bloque, m) if v is not None] for m in metricas}
    con_datos = {m: v for m, v in series.items() if v}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(list(con_datos.values()), tick_labels=list(con_datos), showmeans=True)
    # Los puntos encima de la caja: con cinco pliegues el resumen de cinco
    # numeros esconde mas de lo que muestra.
    for i, valores in enumerate(con_datos.values(), start=1):
        ax.plot([i] * len(valores), valores, "o", color="#2166ac", alpha=0.6, markersize=5)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor en cada pliegue")
    ax.set_title("Cuanto se mueve cada metrica segun quien quede afuera")

    faltantes = [m for m in metricas if not series[m]]
    if faltantes:
        ax.text(0.5, 0.02, f"Sin datos en ningun pliegue: {', '.join(faltantes)}",
                transform=ax.transAxes, ha="center", fontsize=7, color=COLOR_NEUTRO)

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
        ax.text(0.5, 0.5, "El entrenamiento no guardo la importancia de variables",
                transform=ax.transAxes, ha="center", va="center", color=COLOR_NEUTRO)
        ax.set_axis_off()
    ax.set_title("Que mira el modelo para decidir")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def generar_figuras(modelo_dir: Path, salida: Path) -> list[Path]:
    """Escribe las seis laminas a partir de los reportes de `modelo_dir`.

    Todo o nada, de verdad: las seis se dibujan en una carpeta temporal y recien
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

    salida.mkdir(parents=True, exist_ok=True)

    # La temporal cuelga de la propia salida para caer en el mismo sistema de
    # archivos: asi el movimiento final es un rename y no una copia a medias.
    with tempfile.TemporaryDirectory(dir=salida) as tmp:
        taller = Path(tmp)
        borrador = {nombre: taller / nombre for nombre in ARCHIVOS_ESPERADOS}

        _lamina_matriz_confusion(plt, matriz, clases, borrador["matriz_confusion.png"])
        _lamina_distribucion(
            plt, metadata.get("distribucion_clases") or {}, borrador["distribucion_clases.png"]
        )
        _lamina_accuracy(
            plt, bloque, metadata.get("acierto_en_entrenamiento"),
            borrador["accuracy_vs_linea_base.png"],
        )
        _lamina_por_clase(
            plt, metricas_por_clase(matriz, clases), clases, borrador["metricas_por_clase.png"]
        )
        _lamina_dispersion(plt, bloque, borrador["dispersion_pliegues.png"])
        _lamina_importancias(
            plt, importancias_ordenadas(metadata), borrador["importancia_variables.png"]
        )

        escritas = []
        for nombre, origen in borrador.items():
            final = salida / nombre
            origen.replace(final)
            escritas.append(final)

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
    medidas = metricas_por_clase(matriz_confusion_acumulada(bloque), clases)

    return {
        "modelo": metadata.get("modelo"),
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
        },
    }
