"""Derivacion de la etiqueta de riesgo academico (target del modelo).

Categorias (doc HU-019 / tabla RiskPredictions):
  Sobresaliente | SinRiesgo | EnRiesgo | RiesgoCritico

Regla base sobre el promedio del trimestre (0-100) + override por situacion
final ('perdio el año' -> RiesgoCritico). Umbrales en config.yaml -> riesgo.
"""

from __future__ import annotations

import unicodedata

RIESGO_CRITICO = "RiesgoCritico"
EN_RIESGO = "EnRiesgo"
SIN_RIESGO = "SinRiesgo"
SOBRESALIENTE = "Sobresaliente"

CLASES = [RIESGO_CRITICO, EN_RIESGO, SIN_RIESGO, SOBRESALIENTE]

_TERMINOS_REPROBADO = ("PERDIO", "REPROB", "RETIR", "ABANDON")


def _normalizar(texto: str) -> str:
    """Sin acentos y en mayusculas, que es como el centralizador escribe.

    El match era `in` sobre el texto crudo en minusculas, y la planilla dice
    'REPROBADO' y 'RETIRADO' en mayusculas: no coincidia nunca. El override
    quedaba muerto y cada fila de un chico que perdio el año conservaba la
    categoria que le daba su promedio.
    """
    limpio = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return limpio.upper()


def clasificar_riesgo(promedio: float | None, umbrales: dict) -> str | None:
    """Mapea un promedio numerico a una categoria de riesgo.

    Un promedio ausente no tiene categoria. Vale tanto para `None` como para
    `NaN`, que es lo que deja un merge sin pareja: comparar NaN con un umbral da
    False siempre, asi que sin esta guarda cae hasta el ultimo return y todo lo
    que no se pudo etiquetar sale **Sobresaliente**.
    """
    if promedio is None or promedio != promedio:
        return None
    if promedio <= umbrales["riesgo_critico_max"]:
        return RIESGO_CRITICO
    if promedio <= umbrales["en_riesgo_max"]:
        return EN_RIESGO
    if promedio <= umbrales["sin_riesgo_max"]:
        return SIN_RIESGO
    return SOBRESALIENTE


def aplicar_override_situacion(clase: str | None, situacion: str | None) -> str | None:
    """Si la situacion final indica reprobacion/retiro, fuerza RiesgoCritico.

    Sobre una fila que **ya tiene** categoria. Sin `clase` no hay nada que
    corregir: una fila sin etiqueta es una de tercer trimestre, que no tiene
    trimestre siguiente del cual predecir. Dandole categoria igual, sobrevivian al
    filtro solo las de los chicos que perdieron el año, y `trimestre == 3` quedaba
    perfectamente correlacionado con RiesgoCritico — una fuga que el modelo lee a
    traves de `progress_pct` y los `count`, y que ademas inflaba la proporcion de
    la clase sobre la que estan calibradas todas las metricas.
    """
    if clase is None:
        return None
    if situacion and any(t in _normalizar(str(situacion)) for t in _TERMINOS_REPROBADO):
        return RIESGO_CRITICO
    return clase
