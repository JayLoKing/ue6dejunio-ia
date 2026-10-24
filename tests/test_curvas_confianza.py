"""Las laminas nuevas: curvas contra el umbral de confianza, matriz normalizada y
curvas de entrenamiento.

Como en `test_graficos.py`, se prueba el calculo, no el dibujo. La regla que mas
pesa: **un umbral en el que ninguna fila pasa no tiene precision**. Dibujarla como
cero diria que el modelo se equivoca en todas, cuando no hizo ninguna prediccion.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from ue6_ia.evaluation.graficos import (
    ARCHIVOS_ESPERADOS,
    ARCHIVOS_OPCIONALES,
    curvas_por_umbral,
    generar_figuras,
    leer_predicciones,
    leer_registro_entrenamiento,
    matriz_normalizada_por_fila,
    media_entre_clases,
    reordenar_matriz,
    resumen_de_candidatos,
    serie_de_entrenamiento,
)

CLASES = ["EnRiesgo", "RiesgoCritico", "SinRiesgo", "Sobresaliente"]


def _nan(valor: float) -> bool:
    return isinstance(valor, float) and math.isnan(valor)


class TestCurvasPorUmbral:
    reales = ["A", "A", "B", "B"]
    probas = {"A": [0.9, 0.4, 0.6, 0.1], "B": [0.1, 0.6, 0.4, 0.9]}
    umbrales = [0.0, 0.5, 0.95]

    def test_precision_entre_las_filas_que_pasan_el_umbral(self):
        curvas = curvas_por_umbral(self.reales, self.probas, ["A", "B"], self.umbrales)

        assert curvas["precision"]["A"][0] == pytest.approx(0.5)
        assert curvas["precision"]["A"][1] == pytest.approx(0.5)

    def test_sin_filas_que_pasen_no_hay_precision_y_no_es_cero(self):
        curvas = curvas_por_umbral(self.reales, self.probas, ["A", "B"], self.umbrales)

        assert _nan(curvas["precision"]["A"][2])

    def test_recall_sobre_todos_los_positivos_de_la_clase(self):
        curvas = curvas_por_umbral(self.reales, self.probas, ["A", "B"], self.umbrales)

        assert curvas["recall"]["A"] == pytest.approx([1.0, 0.5, 0.0])

    def test_una_clase_sin_casos_reales_no_tiene_recall(self):
        curvas = curvas_por_umbral(
            ["A", "A"], {"A": [0.9, 0.8], "B": [0.1, 0.2]}, ["A", "B"], [0.0, 0.5]
        )

        assert all(_nan(v) for v in curvas["recall"]["B"])

    def test_el_largo_de_cada_curva_es_el_de_los_umbrales(self):
        curvas = curvas_por_umbral(self.reales, self.probas, ["A", "B"], self.umbrales)

        assert len(curvas["precision"]["B"]) == len(self.umbrales)


class TestMediaEntreClases:
    def test_promedia_donde_todas_las_clases_midieron(self):
        media = media_entre_clases({"A": [0.4, 0.2], "B": [0.6, 0.8]})

        assert media == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_se_corta_donde_alguna_clase_deja_de_medir(self):
        # Promediar solo las que quedan hacia saltar la linea negra justo donde
        # una clase se cortaba: una clase menos en la cuenta, no una mejora.
        media = media_entre_clases({"A": [0.4, math.nan], "B": [0.6, 0.8]})

        assert media[0] == pytest.approx(0.5)
        assert _nan(media[1])

    def test_si_ninguna_midio_la_media_tampoco(self):
        media = media_entre_clases({"A": [math.nan], "B": [math.nan]})

        assert _nan(media[0])


class TestMatrizNormalizada:
    def test_cada_fila_suma_cien(self):
        normalizada = matriz_normalizada_por_fila([[3, 1], [0, 4]])

        assert normalizada == [[pytest.approx(75.0), pytest.approx(25.0)], [0.0, 100.0]]

    def test_una_clase_sin_casos_reales_no_tiene_porcentaje(self):
        assert matriz_normalizada_por_fila([[0, 0], [1, 1]])[0] == [None, None]

    def test_reordena_filas_y_columnas_a_la_vez(self):
        matriz = [[1, 2], [3, 4]]

        reordenada, clases = reordenar_matriz(matriz, ["B", "A"], ["A", "B"])

        assert clases == ["A", "B"]
        assert reordenada == [[4, 3], [2, 1]]

    def test_una_clase_fuera_del_orden_va_al_final(self):
        _, clases = reordenar_matriz([[1, 0], [0, 1]], ["X", "A"], ["A"])

        assert clases == ["A", "X"]


class TestSerieDeEntrenamiento:
    def test_un_none_se_dibuja_como_hueco(self):
        registro = [
            {"iteracion": 1, "validation_loss": 1.0},
            {"iteracion": 2, "validation_loss": None},
        ]

        x, y = serie_de_entrenamiento(registro, "validation_loss")

        assert x == [1, 2]
        assert y[0] == 1.0 and _nan(y[1])


def _modelo_completo(tmp_path, con_registro: bool, con_predicciones: bool):
    modelo_dir = tmp_path / "tfdf_riesgo"
    modelo_dir.mkdir()
    matriz = [[2, 1, 0, 0], [1, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]]
    bloque = {
        "modelo": "gradient_boosted_trees",
        "n_pliegues": 1,
        "clases": CLASES,
        "resumen": {},
        "pliegues": [
            {
                "accuracy": 0.8,
                "macro_f1": 0.5,
                "recall_riesgo_critico": 0.5,
                "linea_base": 0.3,
                "matriz_confusion": matriz,
                "n": 10,
            }
        ],
    }
    (modelo_dir / "reporte_validacion_cruzada.json").write_text(
        json.dumps({"gradient_boosted_trees": bloque}), encoding="utf-8"
    )
    (modelo_dir / "metadata.json").write_text(
        json.dumps({"modelo": "gradient_boosted_trees", "distribucion_clases": {"A": 1}}),
        encoding="utf-8",
    )
    if con_registro:
        registro = [
            {
                "iteracion": i,
                "training_loss": 1.0 / i,
                "training_accuracy": 0.5 + i / 100,
                "validation_loss": 1.2 / i,
                "validation_accuracy": 0.4 + i / 100,
            }
            for i in range(1, 6)
        ]
        (modelo_dir / "registro_entrenamiento.json").write_text(
            json.dumps(
                {
                    "modelo_elegido": "gradient_boosted_trees",
                    "modelos": {
                        "gradient_boosted_trees": {
                            "pliegues": [{"pliegue": 1, "registro": registro}]
                        }
                    },
                    "modelo_final": {
                        "modelo": "gradient_boosted_trees",
                        "registro": registro,
                        "iteraciones": {"iteraciones_entrenadas": 5},
                    },
                }
            ),
            encoding="utf-8",
        )
    if con_predicciones:
        filas = []
        for k, real in enumerate(CLASES * 3):
            proba = {c: (0.7 if c == real else 0.1) for c in CLASES}
            filas.append(
                {
                    "modelo": "gradient_boosted_trees",
                    "pliegue": 1,
                    "fila": k,
                    "real": real,
                    "predicha": real,
                    **proba,
                }
            )
        pd.DataFrame(filas).to_parquet(modelo_dir / "predicciones_fuera_de_pliegue.parquet")
    return modelo_dir


class TestLecturaDeLosArchivosNuevos:
    def test_sin_registro_devuelve_none(self, tmp_path):
        assert leer_registro_entrenamiento(tmp_path) is None

    def test_sin_predicciones_devuelve_none(self, tmp_path):
        assert leer_predicciones(tmp_path, "gradient_boosted_trees") is None

    def test_filtra_las_predicciones_del_modelo_pedido(self, tmp_path):
        modelo_dir = _modelo_completo(tmp_path, con_registro=False, con_predicciones=True)

        assert leer_predicciones(modelo_dir, "random_forest") is None
        assert len(leer_predicciones(modelo_dir, "gradient_boosted_trees")) == 12


class TestGenerarFigurasNuevas:
    def test_con_todo_escribe_las_once(self, tmp_path):
        modelo_dir = _modelo_completo(tmp_path, con_registro=True, con_predicciones=True)

        escritas = generar_figuras(modelo_dir, tmp_path / "figures")

        assert {p.name for p in escritas} == set(ARCHIVOS_ESPERADOS) | set(ARCHIVOS_OPCIONALES)
        assert len(escritas) == 11

    def test_sin_los_archivos_nuevos_salen_las_demas_y_no_revienta(self, tmp_path):
        modelo_dir = _modelo_completo(tmp_path, con_registro=False, con_predicciones=False)

        escritas = generar_figuras(modelo_dir, tmp_path / "figures")

        assert {p.name for p in escritas} == set(ARCHIVOS_ESPERADOS)
        assert "matriz_confusion_normalizada.png" in ARCHIVOS_ESPERADOS


class TestBandaSinDesvio:
    def test_sin_ningun_desvio_medido_no_hay_banda(self):
        # Un ancho de cero achicaria la banda a un punto y la lamina diria que
        # no hubo empate, cuando el ancho nunca se calculo.
        bloque = {"resumen": {"macro_f1": {"media": 0.5, "desvio": None}}}

        resumen = resumen_de_candidatos({"a": bloque, "b": bloque}, {"candidato": "a"})

        assert resumen["banda"] is None

    def test_un_desvio_faltante_no_achica_la_banda_de_los_medidos(self):
        medido = {"resumen": {"macro_f1": {"media": 0.6, "desvio": 0.05}}}
        sin_desvio = {"resumen": {"macro_f1": {"media": 0.5, "desvio": None}}}

        resumen = resumen_de_candidatos({"a": medido, "b": sin_desvio}, {"candidato": "a"})

        assert resumen["banda"] == (pytest.approx(0.55), pytest.approx(0.6))
