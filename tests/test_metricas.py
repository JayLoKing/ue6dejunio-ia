"""Metricas de un clasificador de riesgo, y por que estas y no otras."""

from __future__ import annotations

import pytest

from ue6_ia.evaluation.metricas import (
    linea_base_mayoritaria,
    resumen_clasificacion,
    resumir_pliegues,
)

CLASES = ["RiesgoCritico", "EnRiesgo", "SinRiesgo", "Sobresaliente"]


class TestLineaBase:
    """Un accuracy sin su linea base no dice nada."""

    def test_es_la_proporcion_de_la_clase_mas_frecuente(self):
        y = ["SinRiesgo"] * 65 + ["Sobresaliente"] * 61 + ["EnRiesgo"] * 43 + ["RiesgoCritico"] * 12
        assert linea_base_mayoritaria(y) == pytest.approx(65 / 181)

    def test_sin_datos_no_hay_linea_base(self):
        assert linea_base_mayoritaria([]) is None


class TestResumenClasificacion:
    def test_un_clasificador_perfecto_da_uno(self):
        y = ["RiesgoCritico", "SinRiesgo", "EnRiesgo"]
        r = resumen_clasificacion(y, y, CLASES)
        assert r["accuracy"] == pytest.approx(1.0)
        assert r["macro_f1"] == pytest.approx(1.0)

    def test_reporta_el_recall_de_cada_clase_por_separado(self):
        # El macro promedio esconde justo a la clase rara.
        y_true = ["RiesgoCritico", "SinRiesgo", "SinRiesgo", "SinRiesgo"]
        y_pred = ["SinRiesgo", "SinRiesgo", "SinRiesgo", "SinRiesgo"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)

        assert r["accuracy"] == pytest.approx(0.75)
        assert r["por_clase"]["RiesgoCritico"]["recall"] == pytest.approx(0.0)
        assert r["por_clase"]["SinRiesgo"]["recall"] == pytest.approx(1.0)

    def test_el_recall_de_riesgo_critico_se_expone_aparte(self):
        """Es LA metrica: un falso negativo es un chico que reprueba sin aviso."""
        y_true = ["RiesgoCritico", "RiesgoCritico", "SinRiesgo"]
        y_pred = ["RiesgoCritico", "SinRiesgo", "SinRiesgo"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)
        assert r["recall_riesgo_critico"] == pytest.approx(0.5)

    def test_una_clase_ausente_no_inventa_un_uno(self):
        # Sin ejemplos de esa clase el recall no es perfecto: es desconocido.
        y = ["SinRiesgo", "SinRiesgo"]
        r = resumen_clasificacion(y, y, CLASES)
        assert r["por_clase"]["RiesgoCritico"]["soporte"] == 0
        assert r["por_clase"]["RiesgoCritico"]["recall"] is None

    def test_la_matriz_de_confusion_sigue_el_orden_de_las_clases(self):
        y_true = ["RiesgoCritico", "Sobresaliente"]
        y_pred = ["RiesgoCritico", "Sobresaliente"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)
        matriz = r["matriz_confusion"]
        assert len(matriz) == len(CLASES)
        assert matriz[0][0] == 1
        assert matriz[3][3] == 1


class TestResumirPliegues:
    """Con 34 estudiantes un solo corte es una anecdota, no una medicion."""

    def test_devuelve_media_y_desvio_de_cada_metrica(self):
        pliegues = [
            {"accuracy": 0.8, "macro_f1": 0.6},
            {"accuracy": 0.6, "macro_f1": 0.4},
        ]
        r = resumir_pliegues(pliegues)
        assert r["accuracy"]["media"] == pytest.approx(0.7)
        assert r["accuracy"]["desvio"] == pytest.approx(0.1)

    def test_ignora_los_pliegues_donde_la_metrica_no_existe(self):
        # Un pliegue sin RiesgoCritico no puede reportar su recall, y promediarlo
        # como cero castigaria al modelo por un reparto que no eligio.
        pliegues = [
            {"recall_riesgo_critico": 0.5},
            {"recall_riesgo_critico": None},
            {"recall_riesgo_critico": 0.7},
        ]
        r = resumir_pliegues(pliegues)
        assert r["recall_riesgo_critico"]["media"] == pytest.approx(0.6)
        assert r["recall_riesgo_critico"]["pliegues"] == 2

    def test_una_metrica_sin_ningun_valor_queda_en_none(self):
        r = resumir_pliegues([{"recall_riesgo_critico": None}])
        assert r["recall_riesgo_critico"]["media"] is None

    def test_un_solo_pliegue_no_tiene_desvio(self):
        r = resumir_pliegues([{"accuracy": 0.9}])
        assert r["accuracy"]["media"] == pytest.approx(0.9)
        assert r["accuracy"]["desvio"] == pytest.approx(0.0)


class TestEleccionDeModelo:
    """Con macro F1 empatado, decide a cuantos chicos en riesgo detecta."""

    def _comparacion(self, macro_a, recall_a, macro_b, recall_b):
        def res(macro, recall):
            return {"resumen": {
                "macro_f1": {"media": macro},
                "recall_riesgo_critico": {"media": recall},
            }}
        return {"a": res(macro_a, recall_a), "b": res(macro_b, recall_b)}

    def test_una_milesima_de_macro_f1_no_decide(self):
        from ue6_ia.training.train import _elegir
        # Lo medido: 0.6529 vs 0.6541 de macro F1, con desvio 0.13.
        elegido = _elegir(self._comparacion(0.6529, 0.3067, 0.6541, 0.2400))
        assert elegido == "a"

    def test_una_diferencia_real_de_macro_f1_si_decide(self):
        from ue6_ia.training.train import _elegir
        # Un modelo claramente peor no se salva por detectar mas criticos.
        elegido = _elegir(self._comparacion(0.40, 0.90, 0.70, 0.10))
        assert elegido == "b"


class TestF1SinSoporte:
    """Un pliegue sin casos de una clase no midio su F1: no lo midio mal."""

    def test_sin_casos_reales_pero_con_predicciones_el_f1_no_existe(self):
        # Antes daba 0.0, y ese cero entraba al macro F1 que elige el modelo:
        # castigaba al modelo por un reparto de pliegues que no eligio.
        y_true = ["SinRiesgo", "SinRiesgo"]
        y_pred = ["RiesgoCritico", "SinRiesgo"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)

        critico = r["por_clase"]["RiesgoCritico"]
        assert critico["soporte"] == 0
        assert critico["recall"] is None
        assert critico["f1"] is None
        assert critico["precision"] == pytest.approx(0.0)

    def test_con_casos_reales_y_ninguna_prediccion_el_f1_si_es_cero(self):
        # Aca si se midio: habia chicos en riesgo y no detecto a ninguno.
        y_true = ["RiesgoCritico", "SinRiesgo"]
        y_pred = ["SinRiesgo", "SinRiesgo"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)

        critico = r["por_clase"]["RiesgoCritico"]
        assert critico["soporte"] == 1
        assert critico["recall"] == pytest.approx(0.0)
        assert critico["f1"] == pytest.approx(0.0)

    def test_el_macro_f1_no_arrastra_clases_que_el_pliegue_no_tuvo(self):
        y_true = ["SinRiesgo", "SinRiesgo"]
        y_pred = ["RiesgoCritico", "SinRiesgo"]
        r = resumen_clasificacion(y_true, y_pred, CLASES)
        # Solo SinRiesgo tuvo soporte: su F1 es el unico que promedia.
        assert r["macro_f1"] == pytest.approx(r["por_clase"]["SinRiesgo"]["f1"])
