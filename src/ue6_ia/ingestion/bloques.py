"""Donde empieza y termina cada dimension en una fila de encabezados.

Una hoja de area reparte sus columnas en bloques: el encabezado del bloque
('SABER - 45'), despues una columna por criterio que el docente definio, y al
final la de promedio. El numero de criterios cambia por area, por gestion y por
docente, asi que la unica forma estable de leerlos es **por posicion relativa
entre el encabezado y su promedio**, nunca por indice fijo.

Los encabezados vienen con las letras espaciadas ('P R O M E D I O') y con la
ponderacion pegada al nombre ('SABER - 35' en 2023, 'SABER - 45' en 2025). Ambas
cosas se normalizan antes de comparar, de modo que el mismo bloque se reconoce en
los cuatro templates.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PROMEDIO = "PROMEDIO"

# Los nombres con que una hoja anuncia un bloque. Sirven para saber donde termina
# el anterior: una dimension sin promedio propio (SER en 2023 es una sola columna
# con el valor ya resuelto) se comeria los criterios del que sigue.
_ENCABEZADOS = ("SER", "SABER", "HACER", "DECIDIR", "AUTOEVALUACION")


def normalizar(valor: object) -> str:
    """Encabezado sin acentos, en mayusculas y sin espacios."""
    if valor is None:
        return ""
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return "".join(texto.split()).upper()


@dataclass(frozen=True)
class Bloque:
    """El tramo de columnas que ocupa una dimension.

    `inicio` es la columna del encabezado y `promedio` la del total del bloque.
    Los criterios son lo que queda en el medio.
    """

    # Normalizado, no como lo paso quien llamo: es la forma con la que se comparo.
    dimension: str
    inicio: int
    promedio: int


def aplanar_cabecera(filas: list[list]) -> list[str]:
    """Las filas de encabezado vistas como una sola.

    Una hoja de area reparte el encabezado en dos alturas: el nombre del bloque
    ('SABER - 45') va en una fila y su 'P R O M E D I O' en la de abajo. Mirar una
    sola las junta mal: buscando el promedio de SABER en la fila del nombre se
    encuentra recien el de HACER, y los dos bloques terminan compartiendo columnas.
    """
    ancho = max((len(f) for f in filas), default=0)
    aplanada = []
    for col in range(ancho):
        partes = [
            normalizar(fila[col])
            for fila in filas
            if col < len(fila) and normalizar(fila[col])
        ]
        aplanada.append(" ".join(partes))
    return aplanada


def _nombra_dimension(texto: str, dimension: str) -> bool:
    """Si el texto ES el nombre de esa dimension, y no una palabra que lo contiene.

    No alcanza con el prefijo. `normalizar` borra los espacios, asi que el criterio
    'Ser responsable con sus tareas' queda 'SERRESPONSABLECONSUSTAREAS', que empieza
    con SER: leido como encabezado, corta el bloque de SABER y le vacia todos los
    criterios sin que nada falle. Y en estos registros los criterios arrancan con
    verbo — 'Ser...', 'Hacer...', 'Decidir...' — justamente asi.

    Un encabezado de verdad trae la ponderacion pegada ('SABER-45', 'SER-10',
    'AUTOEVALUACION(5)'), de modo que lo que sigue al nombre nunca es otra letra.
    """
    if not texto.startswith(dimension):
        return False
    resto = texto[len(dimension):]
    return not resto or not resto[0].isalpha()


def _abre_bloque(texto: str, dimension: str) -> bool:
    """Si esta columna anuncia el bloque de esa dimension.

    Descarta la columna de total ('PROMEDIO SER'), que nombra la dimension pero la
    cierra en vez de abrirla.
    """
    return _nombra_dimension(texto, dimension) and _PROMEDIO not in texto


def _es_de_otro_bloque(texto: str, objetivo: str) -> bool:
    """Si la columna pertenece al bloque de otra dimension.

    A diferencia de `_abre_bloque`, no le importa que la columna traiga tambien un
    promedio: cuando el nombre del vecino y su total caen en la misma columna, el
    texto dice 'HACER-40 PROMEDIO', y sigue siendo territorio de HACER. Exigir que
    no hubiera promedio dejaba pasar justo ese caso, y el bloque anterior cerraba
    ahi quedandose con los criterios ajenos.

    'PROMEDIO SER' no cuenta: empieza por PROMEDIO, no por el nombre.
    """
    return any(_nombra_dimension(texto, otro) for otro in _ENCABEZADOS if otro != objetivo)


def localizar_bloque(cabecera: list[str], dimension: str) -> Bloque | None:
    """El bloque de esa dimension, o None si no tiene criterios propios en la hoja.

    Devuelve None en dos casos que conviene no confundir con un error: un area
    tecnica que el docente de curso no dicta viene sin bloques, y una dimension
    que la hoja trae ya resuelta en una sola columna —SER y DECIDIR en 2023 y
    2025, que se califican en la hoja 'EVAL SER Y DECIDIR'— no abre ningun tramo.
    """
    objetivo = normalizar(dimension)
    inicio = next(
        (col for col, texto in enumerate(cabecera) if _abre_bloque(texto, objetivo)),
        None,
    )
    if inicio is None:
        return None

    for col in range(inicio + 1, len(cabecera)):
        texto = cabecera[col]
        if not texto:
            continue
        # El vecino se descarta ANTES que el promedio, y el orden importa: el texto
        # de una columna es la union de sus dos alturas, asi que la del vecino puede
        # decir 'HACER-40 PROMEDIO'. Preguntando primero por el promedio, esa columna
        # cerraria el bloque de SABER y le entregaria los criterios de HACER — el
        # solapamiento que este modulo existe para evitar.
        if _es_de_otro_bloque(texto, objetivo):
            # DEBUG y no WARNING a proposito: para SER y DECIDIR este corte es lo
            # esperado en tres de los cuatro templates, y avisarlo por cada area de
            # cada curso llenaria el log de ruido hasta que nadie lo lea. Queda
            # rastreable, y avisar cuando una dimension que SI deberia traer
            # criterios vuelve vacia le toca a quien conoce el template.
            logger.debug(
                "Bloque '%s' abre en %d y lo corta el encabezado vecino de la "
                "columna %d: esta dimension no trae criterios propios en esta hoja.",
                objetivo,
                inicio,
                col,
            )
            return None
        # 'PROMEDIO TRIMESTRAL' cierra la fila entera y no pertenece a ningun bloque.
        if _PROMEDIO in texto and "TRIMESTRAL" not in texto:
            return Bloque(dimension=objetivo, inicio=inicio, promedio=col)

    # Encabezado presente y ningun promedio detras: la hoja no tiene la forma que
    # este modulo sabe leer. Distinto de las dos ausencias legitimas de arriba, y
    # por eso se dice en voz alta en vez de devolver un vacio indistinguible.
    logger.warning(
        "Bloque '%s' abre en la columna %d y no se le encuentra promedio: "
        "la hoja no sigue el formato conocido y sus criterios quedan sin leer.",
        objetivo,
        inicio,
    )
    return None


def _aporta_criterio(texto: str) -> bool:
    """Si la columna del encabezado lleva ademas el nombre de un criterio.

    El texto de una columna es la union de sus alturas, un token por celda. El
    primero es el nombre del bloque; si hay alguno mas que no sea el promedio, esa
    misma columna arranca los criterios.

    Pasa de verdad: en 2025 la columna 3 dice 'SABER - 45' arriba y
    'DESCRIPCION DE LA VACACION' abajo, y el alumno tiene nota ahi. Empezar a
    contar desde la siguiente perdia el primer criterio de cada bloque.
    """
    tokens = texto.split(" ")
    return any(t and _PROMEDIO not in t for t in tokens[1:])


def columnas_de_criterios(cabecera: list[str], dimension: str) -> list[int]:
    """Las columnas con las notas por criterio de esa dimension.

    Es lo que el loader venia salteando: leia solo la columna de promedio y tiraba
    las notas individuales, que son justamente las que el modelo necesita para
    distinguir 90/90/20 de 67/67/66.
    """
    bloque = localizar_bloque(cabecera, dimension)
    if bloque is None:
        return []
    primera = [bloque.inicio] if _aporta_criterio(cabecera[bloque.inicio]) else []
    return primera + list(range(bloque.inicio + 1, bloque.promedio))
