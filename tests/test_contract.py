"""Pruebas del contrato de features (logica pura, sin TensorFlow ni pandas)."""

from __future__ import annotations

import math

import pytest

from ue6_ia.contract import (
    DIMENSIONES,
    FEATURE_COLS,
    ObservacionMateria,
    expandir,
    puede_predecir,
)


def obs(**over) -> ObservacionMateria:
    base = {
        "being": [8.0],
        "knowing": [36.0],
        "doing": [32.0],
        "deciding": [4.0],
        "attendance_pct": 95.0,
        "criterios_planificados": 8,
    }
    base.update(over)
    return ObservacionMateria(**base)


class TestDimensiones:
    """Los topes son los de la RM 0001/2026, y los mismos que la BDD hace cumplir."""

    def test_los_topes_suman_cien(self):
        assert sum(d.tope for d in DIMENSIONES) == 100

    def test_los_nombres_son_los_de_la_bdd(self):
        assert [d.nombre for d in DIMENSIONES] == [
            "being",
            "knowing",
            "doing",
            "deciding",
        ]


class TestDisparo:
    """El docente pidió: se predice recién con al menos una nota en las cuatro."""

    def test_con_una_nota_en_cada_dimension_predice(self):
        assert puede_predecir(obs()) is True

    def test_sin_notas_en_una_dimension_no_predice(self):
        assert puede_predecir(obs(deciding=[])) is False

    def test_no_alcanza_con_muchas_notas_en_tres_dimensiones(self):
        assert puede_predecir(obs(knowing=[40.0, 41.0, 39.0], being=[])) is False


class TestEscala:
    """Cada nota se expresa como porcentaje de su propio tope.

    Sin esto un 5 de Being (la mitad) y un 5 de Knowing (un noveno) entran al
    modelo como el mismo numero, y ningun umbral sobre notas bajas significa lo
    mismo en dos dimensiones distintas.
    """

    def test_la_nota_maxima_de_cada_dimension_vale_cien(self):
        f = expandir(obs(being=[10.0], knowing=[45.0], doing=[40.0], deciding=[5.0]))
        for d in DIMENSIONES:
            assert f[f"{d.nombre}_mean"] == pytest.approx(100.0)

    def test_la_mitad_del_tope_vale_cincuenta(self):
        f = expandir(obs(being=[5.0], knowing=[22.5]))
        assert f["being_mean"] == pytest.approx(50.0)
        assert f["knowing_mean"] == pytest.approx(50.0)

    def test_una_nota_sobre_el_tope_se_rechaza(self):
        with pytest.raises(ValueError, match="tope"):
            expandir(obs(deciding=[6.0]))

    def test_una_nota_negativa_se_rechaza(self):
        with pytest.raises(ValueError, match="negativa"):
            expandir(obs(being=[-1.0]))


class TestEstadisticos:
    """El promedio esconde que 90/90/20 y 67/67/66 son cosas distintas."""

    def test_una_sola_nota_no_tiene_dispersion_ni_tendencia(self):
        f = expandir(obs(knowing=[36.0]))
        assert f["knowing_count"] == 1
        assert f["knowing_std"] == pytest.approx(0.0)
        assert f["knowing_trend"] == pytest.approx(0.0)
        assert f["knowing_min"] == f["knowing_max"] == f["knowing_mean"]

    def test_distingue_dos_series_con_el_mismo_promedio(self):
        pareja = expandir(obs(knowing=[45.0, 45.0, 9.0]))
        pareja_estable = expandir(obs(knowing=[33.0, 33.0, 33.0]))

        assert pareja["knowing_mean"] == pytest.approx(pareja_estable["knowing_mean"], abs=1.0)
        assert pareja["knowing_std"] > pareja_estable["knowing_std"]
        assert pareja["knowing_min"] < pareja_estable["knowing_min"]

    def test_cuenta_las_notas_por_debajo_de_la_nota_de_aprobacion(self):
        # 45 -> 100%, 22 -> 48.9%, 20 -> 44.4%: dos por debajo de 51.
        f = expandir(obs(knowing=[45.0, 22.0, 20.0]))
        assert f["knowing_below"] == 2

    def test_la_tendencia_es_la_ultima_menos_la_primera(self):
        subiendo = expandir(obs(doing=[20.0, 30.0, 36.0]))
        bajando = expandir(obs(doing=[36.0, 30.0, 20.0]))

        assert subiendo["doing_trend"] > 0
        assert bajando["doing_trend"] < 0
        assert subiendo["doing_trend"] == pytest.approx(-bajando["doing_trend"])

    def test_un_criterio_nuevo_cambia_los_valores_no_la_cantidad_de_features(self):
        antes = expandir(obs(knowing=[36.0, 40.0]))
        despues = expandir(obs(knowing=[36.0, 40.0, 12.0]))

        assert list(antes.keys()) == list(despues.keys())
        assert antes["knowing_mean"] != despues["knowing_mean"]


class TestProgreso:
    """`count` dice cuantas notas hay, no cuantas faltan."""

    def test_el_progreso_compara_lo_calificado_con_lo_planificado(self):
        f = expandir(obs(being=[8.0], knowing=[36.0], doing=[32.0], deciding=[4.0],
                         criterios_planificados=8))
        assert f["progress_pct"] == pytest.approx(50.0)

    def test_sin_plan_el_progreso_es_desconocido_no_cero(self):
        f = expandir(obs(criterios_planificados=None))
        assert f["progress_pct"] is None

    def test_mismas_notas_distinto_plan_dan_progresos_distintos(self):
        temprano = expandir(obs(criterios_planificados=12))
        tarde = expandir(obs(criterios_planificados=4))
        assert temprano["progress_pct"] < tarde["progress_pct"]


class TestContratoUnico:
    """La misma funcion en entrenamiento y en inferencia, o el modelo recibe en
    produccion features que nunca vio y se degrada en silencio."""

    def test_expandir_devuelve_exactamente_las_columnas_declaradas(self):
        assert list(expandir(obs()).keys()) == FEATURE_COLS

    def test_el_orden_no_depende_de_los_datos(self):
        a = list(expandir(obs()).keys())
        b = list(expandir(obs(knowing=[10.0, 20.0, 30.0], attendance_pct=None)).keys())
        assert a == b

    def test_son_treinta_features_detras_de_cinco_variables(self):
        # 7 estadisticos x 4 dimensiones + asistencia + progreso
        assert len(FEATURE_COLS) == 30

    def test_la_asistencia_ausente_es_desconocida_no_cero(self):
        f = expandir(obs(attendance_pct=None))
        assert f["attendance_pct"] is None

    def test_ninguna_feature_es_nan(self):
        # NaN y None no son lo mismo para TF-DF: None es "falta", NaN es basura.
        for v in expandir(obs()).values():
            assert v is None or not math.isnan(float(v))
