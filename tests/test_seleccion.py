"""Ponderacion de clases, candidatos y por que gana el que gana.

La regla de eleccion no cambia (la prueban `test_metricas.TestEleccionDeModelo`);
aca se afirma que su explicacion dice lo mismo que ella decide, y que los pesos
salen solo de las filas con que se ajusta.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ue6_ia.evaluation.graficos import (
    ARCHIVOS_ESPERADOS,
    _bloque_del_modelo_elegido,
    generar_figuras,
    resumen_de_candidatos,
)
from ue6_ia.training.train import (
    _elegir,
    candidatos,
    columna_de_pesos,
    explicar_seleccion,
    pesos_balanceados,
    registro_oob_desde_logs,
)

CLASES = ["EnRiesgo", "RiesgoCritico", "SinRiesgo", "Sobresaliente"]


class TestPesosBalanceados:
    def test_cada_clase_pesa_lo_mismo_en_total(self):
        etiquetas = ["A", "A", "A", "B"]

        pesos = pesos_balanceados(etiquetas)

        assert pesos == {"A": pytest.approx(4 / 6), "B": pytest.approx(2.0)}
        assert 3 * pesos["A"] == pytest.approx(1 * pesos["B"])

    def test_una_clase_ausente_no_tiene_peso(self):
        # No hay filas con que calcularlo; inventarle uno no cambia nada del
        # ajuste y ensucia el registro de pesos.
        assert set(pesos_balanceados(["A", "B"])) == {"A", "B"}

    def test_sin_filas_es_un_error(self):
        with pytest.raises(ValueError):
            pesos_balanceados([])

    def test_la_columna_aplica_los_pesos_ya_calculados(self):
        # La parada temprana usa los pesos del ajuste, no unos propios.
        assert columna_de_pesos(["B", "A", "C"], {"A": 0.5, "B": 2.0}) == [2.0, 0.5, 1.0]


class TestCandidatos:
    def test_combina_modelos_y_ponderaciones_con_claves_estables(self):
        assert candidatos(
            ["gradient_boosted_trees", "random_forest"], ["ninguna", "balanceada"]
        ) == [
            ("gradient_boosted_trees", "gradient_boosted_trees", "ninguna"),
            ("random_forest", "random_forest", "ninguna"),
            ("gradient_boosted_trees_ponderado", "gradient_boosted_trees", "balanceada"),
            ("random_forest_ponderado", "random_forest", "balanceada"),
        ]

    def test_acepta_un_valor_suelto(self):
        assert candidatos(["random_forest"], "balanceada") == [
            ("random_forest_ponderado", "random_forest", "balanceada")
        ]

    def test_una_ponderacion_desconocida_es_un_error(self):
        with pytest.raises(ValueError, match="ponderacion"):
            candidatos(["random_forest"], ["inventada"])


def _comparacion() -> dict:
    def res(macro, desvio, recall):
        return {
            "resumen": {
                "macro_f1": {"media": macro, "desvio": desvio, "pliegues": 5},
                "recall_riesgo_critico": {"media": recall, "desvio": 0.1, "pliegues": 4},
            }
        }

    return {
        "gbt": res(0.534, 0.062, 0.062),
        "rf": res(0.563, 0.095, 0.125),
        "rf_p": res(0.550, 0.040, 0.300),
        "malo": res(0.400, 0.010, 0.900),
    }


class TestExplicarSeleccion:
    def test_el_ganador_es_el_de_la_regla(self):
        comparacion = _comparacion()

        assert explicar_seleccion(comparacion)["ganador"] == _elegir(comparacion) == "rf_p"

    def test_dice_quien_fijo_la_banda_y_quien_quedo_adentro(self):
        seleccion = explicar_seleccion(_comparacion())

        assert seleccion["mejor_macro_f1"] == {"candidato": "rf", "valor": 0.563}
        assert seleccion["banda"] == {"ancho": 0.095, "candidato": "rf"}
        assert [c["candidato"] for c in seleccion["dentro_de_la_banda"]] == ["gbt", "rf", "rf_p"]

    def test_los_excluidos_llevan_su_motivo(self):
        seleccion = explicar_seleccion(_comparacion())

        (excluido,) = seleccion["excluidos"]
        assert excluido["candidato"] == "malo"
        assert "banda" in excluido["motivo"]
        assert seleccion["criterio"]

    def test_un_solo_candidato_en_la_banda_gana_por_macro_f1(self):
        comparacion = _comparacion()
        for bloque in comparacion.values():
            bloque["resumen"]["macro_f1"]["desvio"] = 0.001

        seleccion = explicar_seleccion(comparacion)

        assert seleccion["ganador"] == "rf"
        assert "macro F1" in seleccion["criterio"]


class TestRegistroOob:
    def test_lee_arboles_accuracy_y_perdida_fuera_de_bolsa(self):
        logs = [
            SimpleNamespace(
                num_trees=1, evaluation=SimpleNamespace(num_examples=105, accuracy=0.9, loss=0.34)
            ),
            SimpleNamespace(
                num_trees=50, evaluation=SimpleNamespace(num_examples=300, accuracy=0.95, loss=None)
            ),
        ]

        assert registro_oob_desde_logs(logs) == [
            {"arboles": 1, "oob_accuracy": 0.9, "oob_loss": 0.34, "oob_ejemplos": 105},
            {"arboles": 50, "oob_accuracy": 0.95, "oob_loss": None, "oob_ejemplos": 300},
        ]


def _bloque(macro: float, recall: float) -> dict:
    matriz = [[2, 1, 0, 0], [1, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]]
    return {
        "clases": CLASES,
        "resumen": {
            "macro_f1": {"media": macro, "desvio": 0.05, "pliegues": 1},
            "recall_riesgo_critico": {"media": recall, "desvio": 0.1, "pliegues": 1},
        },
        "pliegues": [
            {
                "accuracy": 0.8,
                "macro_f1": macro,
                "recall_riesgo_critico": recall,
                "linea_base": 0.3,
                "matriz_confusion": matriz,
                "n": 10,
            }
        ],
    }


class TestCandidatosEnLasLaminas:
    def test_el_bloque_elegido_se_busca_por_candidato(self):
        reporte = {"random_forest": _bloque(0.5, 0.1), "random_forest_ponderado": _bloque(0.6, 0.3)}
        metadata = {"modelo": "random_forest", "candidato": "random_forest_ponderado"}

        assert _bloque_del_modelo_elegido(reporte, metadata) is reporte["random_forest_ponderado"]

    def test_resumen_de_candidatos_marca_al_ganador_y_la_banda(self):
        reporte = {"a": _bloque(0.5, 0.1), "b": _bloque(0.6, 0.3)}
        metadata = {
            "candidato": "b",
            "seleccion": {"mejor_macro_f1": {"valor": 0.6}, "banda": {"ancho": 0.05}},
        }

        resumen = resumen_de_candidatos(reporte, metadata)

        assert [c["candidato"] for c in resumen["candidatos"]] == ["a", "b"]
        assert [c["ganador"] for c in resumen["candidatos"]] == [False, True]
        assert resumen["banda"] == (pytest.approx(0.55), pytest.approx(0.6))
        assert resumen["candidatos"][1]["recall_riesgo_critico"]["media"] == 0.3

    def test_sin_seleccion_guardada_la_banda_sale_del_desvio(self):
        resumen = resumen_de_candidatos({"a": _bloque(0.5, 0.1)}, {"modelo": "a"})

        assert resumen["banda"] == (pytest.approx(0.45), pytest.approx(0.5))

    def test_un_rf_con_registro_oob_dibuja_su_evolucion(self, tmp_path):
        modelo_dir = tmp_path / "tfdf_riesgo"
        modelo_dir.mkdir()
        (modelo_dir / "reporte_validacion_cruzada.json").write_text(
            json.dumps({"random_forest": {**_bloque(0.5, 0.1), "modelo": "random_forest"}}),
            encoding="utf-8",
        )
        (modelo_dir / "metadata.json").write_text(
            json.dumps({"modelo": "random_forest", "candidato": "random_forest"}),
            encoding="utf-8",
        )
        oob = [
            {"arboles": a, "oob_accuracy": 0.5 + a / 1000, "oob_loss": 1 / a, "oob_ejemplos": 10}
            for a in (1, 10, 50, 300)
        ]
        (modelo_dir / "registro_entrenamiento.json").write_text(
            json.dumps(
                {
                    "modelos": {
                        "random_forest": {
                            "tipo_modelo": "random_forest",
                            "pliegues": [{"pliegue": 1, "registro": None, "registro_oob": oob}],
                        }
                    },
                    "modelo_final": {
                        "modelo": "random_forest",
                        "tipo_modelo": "random_forest",
                        "registro": None,
                        "registro_oob": oob,
                    },
                }
            ),
            encoding="utf-8",
        )

        escritas = {p.name for p in generar_figuras(modelo_dir, tmp_path / "figures")}

        assert "resultados_entrenamiento.png" in escritas
        assert "comparacion_candidatos.png" in ARCHIVOS_ESPERADOS
        assert "comparacion_candidatos.png" in escritas


class TestTablaDeSeleccion:
    def test_marca_ganador_banda_y_excluidos_en_cien_columnas(self):
        from ue6_ia.evaluation.consola import _lineas_planas, tabla_seleccion
        from ue6_ia.training.train import explicar_seleccion

        comparacion = _comparacion()
        tabla = tabla_seleccion(comparacion, explicar_seleccion(comparacion))

        estados = {fila[0]: fila[-1] for fila in tabla.filas}
        assert estados["rf_p"].startswith("Ganador")
        assert estados["gbt"] == "En la banda"
        assert estados["malo"].startswith("Fuera de la banda")
        assert all(len(linea) <= 100 for linea in _lineas_planas([tabla]))

    def test_un_excluido_sin_macro_f1_no_muestra_una_distancia_inventada(self):
        from ue6_ia.evaluation.consola import tabla_seleccion

        seleccion = {
            "ganador": "a",
            "mejor_macro_f1": {"valor": 0.6},
            "excluidos": [{"candidato": "b", "macro_f1": None}],
        }

        tabla = tabla_seleccion({"a": _bloque(0.6, 0.3), "b": _bloque(0.4, 0.1)}, seleccion)

        estados = {fila[0]: fila[-1] for fila in tabla.filas}
        assert estados["b"] == "Fuera de la banda (n/d)"
