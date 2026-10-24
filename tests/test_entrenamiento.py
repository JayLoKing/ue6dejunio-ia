"""La parte del entrenamiento que no necesita TensorFlow para probarse.

La parada temprana del GBT se decide contra un conjunto de validacion. Si ese
conjunto se arma por fila, el mismo chico queda en el ajuste y en la parada, y la
curva de validacion sale optimista. Aca se afirma que se arma por estudiante.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from ue6_ia.training.train import (
    describir_particion,
    iteraciones_del_registro,
    particion_parada_temprana,
    registro_desde_entradas,
    tabla_predicciones,
)

CLASES = ["EnRiesgo", "RiesgoCritico", "SinRiesgo", "Sobresaliente"]


def _grupos_y_etiquetas(n_estudiantes: int = 30, filas: int = 5) -> tuple[list[str], list[str]]:
    grupos, etiquetas = [], []
    for e in range(n_estudiantes):
        for f in range(filas):
            grupos.append(f"E{e:02d}")
            etiquetas.append(CLASES[(e + f) % len(CLASES)])
    return grupos, etiquetas


class TestParticionParadaTemprana:
    def test_ningun_estudiante_queda_de_los_dos_lados(self):
        grupos, etiquetas = _grupos_y_etiquetas()

        ajuste, parada = particion_parada_temprana(grupos, etiquetas, fraccion=0.1, semilla=42)

        assert {grupos[i] for i in ajuste}.isdisjoint({grupos[i] for i in parada})

    def test_cubre_todas_las_filas_una_sola_vez(self):
        grupos, etiquetas = _grupos_y_etiquetas()

        ajuste, parada = particion_parada_temprana(grupos, etiquetas, fraccion=0.1, semilla=42)

        assert sorted([*ajuste, *parada]) == list(range(len(grupos)))

    def test_aparta_cerca_del_diez_por_ciento_de_los_estudiantes(self):
        grupos, etiquetas = _grupos_y_etiquetas(n_estudiantes=30)

        _, parada = particion_parada_temprana(grupos, etiquetas, fraccion=0.1, semilla=42)

        assert 2 <= len({grupos[i] for i in parada}) <= 4

    def test_es_reproducible_con_la_misma_semilla(self):
        grupos, etiquetas = _grupos_y_etiquetas()

        primera = particion_parada_temprana(grupos, etiquetas, semilla=7)
        segunda = particion_parada_temprana(grupos, etiquetas, semilla=7)

        assert primera == segunda

    def test_con_un_solo_estudiante_no_hay_con_que_parar(self):
        with pytest.raises(ValueError, match="estudiantes"):
            particion_parada_temprana(["E1"] * 4, ["EnRiesgo"] * 4)


class TestDescribirParticion:
    def test_cuenta_filas_y_estudiantes_de_cada_lado_con_su_porcentaje(self):
        particion = describir_particion(
            ajuste=["A", "A", "B", "C"],
            parada=["D"],
            validacion=["E", "E", "F"],
            total_filas=8,
            total_estudiantes=6,
        )

        assert particion["filas_entrenamiento"] == 4
        assert particion["estudiantes_entrenamiento"] == 3
        assert particion["filas_parada"] == 1
        assert particion["estudiantes_parada"] == 1
        assert particion["filas_validacion"] == 3
        assert particion["estudiantes_validacion"] == 2
        assert particion["pct_filas_entrenamiento"] == pytest.approx(50.0)
        assert particion["pct_estudiantes_validacion"] == pytest.approx(100 * 2 / 6)

    def test_un_estudiante_en_dos_conjuntos_es_un_error(self):
        with pytest.raises(RuntimeError, match="estudiantes"):
            describir_particion(
                ajuste=["A", "B"],
                parada=["B"],
                validacion=["C"],
                total_filas=4,
                total_estudiantes=3,
            )

    def test_el_solapamiento_con_la_validacion_tambien_se_detecta(self):
        with pytest.raises(RuntimeError):
            describir_particion(
                ajuste=["A"], parada=[], validacion=["A"], total_filas=2, total_estudiantes=1
            )


class _Entrada(SimpleNamespace):
    """Imita una entrada del protobuf: campos opcionales que pueden no estar."""

    def HasField(self, nombre: str) -> bool:  # noqa: N802 - la API del protobuf se llama asi
        return getattr(self, nombre, None) is not None


class TestRegistroDesdeEntradas:
    def test_lee_perdida_y_accuracy_de_cada_iteracion(self):
        entradas = [
            _Entrada(
                number_of_trees=1,
                training_loss=1.2,
                training_secondary_metrics=[0.5],
                validation_loss=1.3,
                validation_secondary_metrics=[0.45],
            ),
            _Entrada(
                number_of_trees=2,
                training_loss=1.0,
                training_secondary_metrics=[0.6],
                validation_loss=1.25,
                validation_secondary_metrics=[0.5],
            ),
        ]

        registro = registro_desde_entradas(entradas)

        assert registro == [
            {
                "iteracion": 1,
                "training_loss": 1.2,
                "training_accuracy": 0.5,
                "validation_loss": 1.3,
                "validation_accuracy": 0.45,
            },
            {
                "iteracion": 2,
                "training_loss": 1.0,
                "training_accuracy": 0.6,
                "validation_loss": 1.25,
                "validation_accuracy": 0.5,
            },
        ]

    def test_un_campo_ausente_es_none_y_no_cero(self):
        # El protobuf devuelve 0.0 por un campo que nadie escribio. Ese cero se
        # graficaria como una perdida perfecta.
        entradas = [
            _Entrada(
                number_of_trees=1,
                training_loss=0.9,
                training_secondary_metrics=[],
                validation_loss=None,
                validation_secondary_metrics=[],
            )
        ]

        (fila,) = registro_desde_entradas(entradas)

        assert fila["validation_loss"] is None
        assert fila["training_accuracy"] is None
        assert fila["validation_accuracy"] is None

    def test_un_nan_tampoco_se_guarda_como_numero(self):
        entradas = [
            _Entrada(
                number_of_trees=3,
                training_loss=math.nan,
                training_secondary_metrics=[0.7],
                validation_loss=1.0,
                validation_secondary_metrics=[0.6],
            )
        ]

        (fila,) = registro_desde_entradas(entradas)

        assert fila["training_loss"] is None


class TestIteracionesDelRegistro:
    def test_distingue_donde_paro_de_la_mejor_validacion(self):
        registro = [
            {"iteracion": 1, "validation_loss": 1.0},
            {"iteracion": 2, "validation_loss": 0.8},
            {"iteracion": 3, "validation_loss": 0.9},
        ]

        assert iteraciones_del_registro(registro) == {
            "iteraciones_entrenadas": 3,
            "iteracion_mejor_validacion": 2,
        }

    def test_sin_registro_no_inventa_iteraciones(self):
        assert iteraciones_del_registro(None) == {
            "iteraciones_entrenadas": None,
            "iteracion_mejor_validacion": None,
        }


class TestTablaPredicciones:
    def test_una_columna_por_clase_en_el_orden_de_la_matriz(self):
        proba = np.array([[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]])

        tabla = tabla_predicciones(
            modelo="gradient_boosted_trees",
            pliegue=3,
            filas=[10, 11],
            reales=["RiesgoCritico", "SinRiesgo"],
            proba=proba,
            clases=CLASES,
        )

        assert list(tabla.columns) == ["modelo", "pliegue", "fila", "real", "predicha", *CLASES]
        assert tabla["predicha"].tolist() == ["RiesgoCritico", "EnRiesgo"]
        assert tabla["RiesgoCritico"].tolist() == [0.7, 0.2]
        assert tabla["pliegue"].tolist() == [3, 3]

    def test_una_matriz_con_otro_numero_de_clases_es_un_error(self):
        with pytest.raises(ValueError, match="clases"):
            tabla_predicciones(
                modelo="gbt",
                pliegue=1,
                filas=[0],
                reales=["EnRiesgo"],
                proba=np.array([[0.5, 0.5]]),
                clases=CLASES,
            )
