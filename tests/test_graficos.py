"""Los graficos que documentan el modelo, y de donde sale cada numero.

Se prueba la lectura de los reportes, no el dibujo. Que una barra quede azul o
roja no es una afirmacion sobre el modelo; que la matriz de confusion sea la
suma de los cinco pliegues si lo es, y eso es lo que se afirma aca.
"""

from __future__ import annotations

import json

import pytest

from ue6_ia.evaluation.graficos import (
    ARCHIVOS_ESPERADOS,
    clases_del_bloque,
    expandir_matriz,
    generar_figuras,
    importancias_ordenadas,
    matriz_confusion_acumulada,
    metricas_por_clase,
    nombre_de_feature,
    puntos_medidos,
    serie_por_pliegue,
)

CLASES = ["EnRiesgo", "RiesgoCritico", "SinRiesgo", "Sobresaliente"]


def _pliegue(matriz: list[list[int]], accuracy: float, linea_base: float) -> dict:
    return {
        "accuracy": accuracy,
        "macro_f1": 0.5,
        "recall_riesgo_critico": 0.3,
        "linea_base": linea_base,
        "matriz_confusion": matriz,
        "n": sum(sum(f) for f in matriz),
    }


@pytest.fixture
def bloque() -> dict:
    """Un modelo con dos pliegues, suficiente para probar la acumulacion."""
    return {
        "modelo": "gradient_boosted_trees",
        "n_pliegues": 2,
        "clases": CLASES,
        "resumen": {
            "accuracy": {"media": 0.6, "desvio": 0.02, "pliegues": 2},
            "recall_riesgo_critico": {"media": 0.3, "desvio": 0.1, "pliegues": 1},
        },
        "pliegues": [
            _pliegue([[2, 1, 0, 0], [1, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]], 0.8, 0.4),
            _pliegue([[1, 0, 1, 0], [0, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 1]], 0.7, 0.35),
        ],
    }


class TestNombreDeFeature:
    """TF-DF devuelve el nombre envuelto en su propia contabilidad interna."""

    def test_desenvuelve_el_nombre_real(self):
        assert nombre_de_feature('"knowing_mean" (1; #26)') == "knowing_mean"

    def test_un_nombre_ya_limpio_queda_igual(self):
        assert nombre_de_feature("attendance_pct") == "attendance_pct"


class TestMatrizAcumulada:
    def test_suma_los_pliegues_celda_por_celda(self, bloque):
        # Cada pliegue evalua estudiantes que el modelo no vio: sumarlos da la
        # matriz fuera de muestra sobre el dataset entero, que es la honesta.
        # La de `reporte_en_muestra.json` mide memorizacion.
        assert matriz_confusion_acumulada(bloque) == [
            [3, 1, 1, 0],
            [1, 3, 0, 0],
            [0, 0, 5, 1],
            [0, 0, 1, 3],
        ]

    def test_sin_pliegues_no_inventa_una_matriz(self):
        assert matriz_confusion_acumulada({"clases": CLASES, "pliegues": []}) == []


class TestExpandirMatriz:
    """La matriz y las listas que la produjeron dicen lo mismo."""

    def test_reconstruye_las_listas_de_verdad_y_prediccion(self):
        matriz = [[1, 1], [0, 2]]
        y_true, y_pred = expandir_matriz(matriz, ["A", "B"])

        assert sorted(y_true) == ["A", "A", "B", "B"]
        assert sorted(y_pred) == ["A", "B", "B", "B"]
        assert len(y_true) == len(y_pred) == 4


    def test_una_matriz_que_no_cuadra_con_las_clases_es_un_error_no_un_dibujo(self):
        # Las clases pueden venir del bloque y la matriz de los pliegues. Si
        # discrepan, indexar igual publicaria el recall de RiesgoCritico bajo
        # otro nombre — justo la metrica que manda, mal rotulada.
        with pytest.raises(ValueError, match="no coincide"):
            expandir_matriz([[1, 0], [0, 1]], ["A", "B", "C"])


class TestClasesDelBloque:
    def test_las_toma_del_bloque_cuando_estan(self, bloque):
        assert clases_del_bloque(bloque, {}) == CLASES

    def test_sin_clases_en_el_bloque_avisa_antes_de_caer_al_metadata(self, caplog):
        b = {"pliegues": []}
        metadata = {"distribucion_clases": {"SinRiesgo": 2, "EnRiesgo": 1}}

        with caplog.at_level("WARNING"):
            clases = clases_del_bloque(b, metadata)

        assert clases == ["EnRiesgo", "SinRiesgo"]
        assert caplog.records, "el reemplazo silencioso es justo lo que no puede pasar"

    def test_sin_clases_en_ningun_lado_falla_en_vez_de_dibujar_ejes_vacios(self):
        # Una lamina titulada "por que el accuracy solo no alcanza" sobre un eje
        # en blanco es peor que no emitirla: el documento la citaria como prueba.
        with pytest.raises(ValueError, match="clases"):
            clases_del_bloque({}, {})


class TestMetricasPorClase:
    def test_las_deriva_de_la_matriz_con_las_reglas_ya_auditadas(self, bloque):
        medidas = metricas_por_clase(matriz_confusion_acumulada(bloque), CLASES)

        # RiesgoCritico: 3 aciertos sobre 4 reales, 3 sobre 4 predichos.
        assert medidas["por_clase"]["RiesgoCritico"]["recall"] == pytest.approx(3 / 4)
        assert medidas["por_clase"]["RiesgoCritico"]["precision"] == pytest.approx(3 / 4)
        assert medidas["n"] == 19

    def test_una_clase_sin_casos_reales_no_mide_cero_sino_nada(self):
        # La misma posicion que sostiene `metricas.py`: un pliegue sin un solo
        # caso de la clase no midio mal, no midio. Un cero aqui entraria al
        # promedio y castigaria al modelo por un reparto que no eligio.
        matriz = [[2, 0], [0, 0]]
        medidas = metricas_por_clase(matriz, ["A", "B"])

        assert medidas["por_clase"]["B"]["recall"] is None
        assert medidas["por_clase"]["B"]["f1"] is None


class TestSeriePorPliegue:
    def test_devuelve_el_valor_de_cada_pliegue_en_orden(self, bloque):
        assert serie_por_pliegue(bloque, "accuracy") == [0.8, 0.7]
        assert serie_por_pliegue(bloque, "linea_base") == [0.4, 0.35]

    def test_un_pliegue_que_no_midio_queda_como_hueco_no_como_cero(self):
        b = {"pliegues": [{"recall_riesgo_critico": 0.5}, {"recall_riesgo_critico": None}]}
        assert serie_por_pliegue(b, "recall_riesgo_critico") == [0.5, None]


class TestPuntosMedidos:
    """Un pliegue que no midio no entra al grafico como cero."""

    def test_deja_afuera_los_huecos_y_conserva_la_posicion_de_los_demas(self):
        indices, valores = puntos_medidos([0.8, None, 0.6])

        assert indices == [0, 2]
        assert valores == [0.8, 0.6]

    def test_un_cero_real_si_entra(self):
        # Cero y "no medido" son cosas distintas: el modelo que erro todos los
        # casos midio, y mal. Confundirlos borra justo esa diferencia.
        indices, valores = puntos_medidos([0.0, None])

        assert indices == [0]
        assert valores == [0.0]

    def test_sin_una_sola_medicion_no_hay_nada_que_dibujar(self):
        assert puntos_medidos([None, None]) == ([], [])


class TestImportancias:
    def test_ordena_de_mayor_a_menor_y_corta_en_el_tope(self):
        metadata = {
            "importancia_variables": {
                "INV_MEAN_MIN_DEPTH": [
                    ['"doing_mean" (1; #19)', 0.26],
                    ['"knowing_mean" (1; #26)', 0.27],
                    ['"attendance_pct" (1; #1)', 0.22],
                ]
            }
        }
        top = importancias_ordenadas(metadata, "INV_MEAN_MIN_DEPTH", tope=2)

        assert top == [("knowing_mean", 0.27), ("doing_mean", 0.26)]

    def test_una_metrica_que_el_entrenamiento_no_guardo_da_vacio(self):
        # `_importancias` traga su excepcion a proposito: la importancia es
        # informativa. Pedir una que no esta no puede voltear la generacion.
        assert importancias_ordenadas({"importancia_variables": {}}, "NO_EXISTE") == []

    def test_un_metadata_sin_importancias_da_vacio(self):
        assert importancias_ordenadas({}, "INV_MEAN_MIN_DEPTH") == []


class TestGenerarFiguras:
    def test_escribe_una_figura_por_cada_lamina_declarada(self, tmp_path, bloque):
        modelo_dir = tmp_path / "tfdf_riesgo"
        modelo_dir.mkdir()
        (modelo_dir / "reporte_validacion_cruzada.json").write_text(
            json.dumps({"gradient_boosted_trees": bloque}), encoding="utf-8"
        )
        (modelo_dir / "metadata.json").write_text(
            json.dumps({
                "modelo": "gradient_boosted_trees",
                "n_filas": 19,
                "n_estudiantes": 4,
                "acierto_en_entrenamiento": 0.92,
                "distribucion_clases": {
                    "EnRiesgo": 5, "RiesgoCritico": 4, "SinRiesgo": 6, "Sobresaliente": 4
                },
                "importancia_variables": {
                    "INV_MEAN_MIN_DEPTH": [['"knowing_mean" (1; #26)', 0.27]]
                },
            }),
            encoding="utf-8",
        )
        salida = tmp_path / "figures"

        escritas = generar_figuras(modelo_dir, salida)

        assert {p.name for p in escritas} == set(ARCHIVOS_ESPERADOS)
        for p in escritas:
            assert p.exists() and p.stat().st_size > 0

    def test_un_pliegue_sin_medir_no_voltea_la_generacion(self, tmp_path, bloque):
        # El pliegue que no trajo un caso critico deja `None`. La lamina tiene
        # que emitirse igual, marcandolo, no reventar ni fabricar un cero.
        bloque["pliegues"][1]["recall_riesgo_critico"] = None
        bloque["pliegues"][1]["accuracy"] = None
        modelo_dir = tmp_path / "tfdf_riesgo"
        modelo_dir.mkdir()
        (modelo_dir / "reporte_validacion_cruzada.json").write_text(
            json.dumps({"gradient_boosted_trees": bloque}), encoding="utf-8"
        )
        (modelo_dir / "metadata.json").write_text(
            json.dumps({"modelo": "gradient_boosted_trees", "distribucion_clases": {"A": 1}}),
            encoding="utf-8",
        )

        escritas = generar_figuras(modelo_dir, tmp_path / "figures")

        assert {p.name for p in escritas} == set(ARCHIVOS_ESPERADOS)

    def test_sin_reporte_de_validacion_avisa_en_vez_de_romper_a_medias(self, tmp_path):
        # Un modelo entrenado por una version vieja no trae el reporte. Escribir
        # media tanda de figuras dejaria un documento citando graficos que
        # describen otra corrida.
        with pytest.raises(FileNotFoundError):
            generar_figuras(tmp_path / "no_existe", tmp_path / "figures")
