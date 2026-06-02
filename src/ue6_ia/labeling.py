"""Derivacion de la etiqueta de riesgo academico (target del modelo).

Categorias (doc HU-019 / tabla RiskPredictions):
  Sobresaliente | SinRiesgo | EnRiesgo | RiesgoCritico

Regla base sobre el promedio del trimestre (0-100) + override por situacion
final ('perdio el año' -> RiesgoCritico). Umbrales en config.yaml -> riesgo.
"""

from __future__ import annotations

RIESGO_CRITICO = "RiesgoCritico"
EN_RIESGO = "EnRiesgo"
SIN_RIESGO = "SinRiesgo"
SOBRESALIENTE = "Sobresaliente"

CLASES = [RIESGO_CRITICO, EN_RIESGO, SIN_RIESGO, SOBRESALIENTE]

_TERMINOS_REPROBADO = ("perdio", "perdió", "reprob", "retir", "abandon")


def clasificar_riesgo(promedio: float | None, umbrales: dict) -> str | None:
    """Mapea un promedio numerico a una categoria de riesgo."""
    if promedio is None:
        return None
    if promedio <= umbrales["riesgo_critico_max"]:
        return RIESGO_CRITICO
    if promedio <= umbrales["en_riesgo_max"]:
        return EN_RIESGO
    if promedio <= umbrales["sin_riesgo_max"]:
        return SIN_RIESGO
    return SOBRESALIENTE


def aplicar_override_situacion(clase: str | None, situacion: str | None) -> str | None:
    """Si la situacion final indica reprobacion/retiro, fuerza RiesgoCritico."""
    if situacion and any(t in situacion for t in _TERMINOS_REPROBADO):
        return RIESGO_CRITICO
    return clase
