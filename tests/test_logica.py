"""Pruebas de la logica pura (no requieren TensorFlow)."""

from __future__ import annotations

from ue6_ia.ingestion.discovery import parse_folder_name
from ue6_ia.labeling import (
    EN_RIESGO,
    RIESGO_CRITICO,
    SIN_RIESGO,
    SOBRESALIENTE,
    aplicar_override_situacion,
    clasificar_riesgo,
)
from ue6_ia.preprocessing.cleaning import normalizar_nombre

UMBRALES = {"riesgo_critico_max": 50, "en_riesgo_max": 66, "sin_riesgo_max": 84}


def test_normalizar_nombre():
    assert normalizar_nombre("ARGANDOÑA  Vargas ") == "ARGANDONA VARGAS"
    assert normalizar_nombre("  Céspedes   Castellón ") == "CESPEDES CASTELLON"
    assert normalizar_nombre(None) == ""


def test_clasificar_riesgo():
    assert clasificar_riesgo(41, UMBRALES) == RIESGO_CRITICO
    assert clasificar_riesgo(50, UMBRALES) == RIESGO_CRITICO
    assert clasificar_riesgo(55, UMBRALES) == EN_RIESGO
    assert clasificar_riesgo(75, UMBRALES) == SIN_RIESGO
    assert clasificar_riesgo(91, UMBRALES) == SOBRESALIENTE
    assert clasificar_riesgo(None, UMBRALES) is None


def test_override_situacion():
    # 'perdio el año' fuerza RiesgoCritico aunque el promedio fuera mayor
    assert aplicar_override_situacion(SIN_RIESGO, "perdio el año") == RIESGO_CRITICO
    assert aplicar_override_situacion(SIN_RIESGO, "") == SIN_RIESGO
    assert aplicar_override_situacion(SOBRESALIENTE, None) == SOBRESALIENTE


def test_parse_folder():
    from pathlib import Path

    c = parse_folder_name(Path("5º B Registro Pedag. 2026 Elizabeth Argandoña Vargas"))
    assert c is not None
    assert c.grado == 5
    assert c.paralelo == "B"
    assert c.gestion == 2026
    assert "Argandoña" in c.docente

    assert parse_folder_name(Path("QUINTO DE PRIMARIA 2024")) is None
