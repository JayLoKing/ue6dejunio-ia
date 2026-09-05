"""Pruebas del contrato de features (logica pura, sin TensorFlow ni pandas)."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

import pytest

from ue6_ia.contract import (
    DIMENSIONES,
    ESCALA_VIGENTE,
    ESCALAS_POR_GESTION,
    FEATURE_COLS,
    Escala,
    ObservacionMateria,
    escala_de,
    expandir,
    puede_predecir,
)


@contextmanager
def caplog_vacio():
    """Captura los avisos del modulo para poder afirmar que NO hubo ninguno."""
    registros: list[logging.LogRecord] = []

    class Recolector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            registros.append(record)

    log = logging.getLogger("ue6_ia.contract")
    handler = Recolector(level=logging.WARNING)
    log.addHandler(handler)
    try:
        yield registros
    finally:
        log.removeHandler(handler)


def obs(**over) -> ObservacionMateria:
    # Notas validas bajo cualquiera de las escalas, para que un test sobre la
    # ponderacion de 2023 no rebote por un default pensado para la de hoy.
    base = {
        "being": [4.0],
        "knowing": [30.0],
        "doing": [30.0],
        "deciding": [4.0],
        "attendance_pct": 95.0,
        "criterios_planificados": 8,
    }
    base.update(over)
    return ObservacionMateria(**base)


class TestDimensiones:
    """Los nombres son los que la BDD acepta; cuanto vale cada uno lo dice `Escala`."""

    def test_los_nombres_son_los_de_la_bdd(self):
        assert list(DIMENSIONES) == ["being", "knowing", "doing", "deciding"]

    def test_las_dimensiones_no_cargan_su_propio_tope(self):
        # Un tope pegado a la dimension seria una segunda fuente de verdad para un
        # numero que cambia con la gestion, y el primero en leerlo normalizaria
        # datos de 2023 contra los topes de hoy sin que falle nada.
        assert all(isinstance(d, str) for d in DIMENSIONES)


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
            assert f[f"{d}_mean"] == pytest.approx(100.0)

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


class TestEscalaPorGestion:
    """La escuela no ponderó siempre igual, y el modelo tiene que ver una sola escala.

    Medido en los encabezados de los registros: 2023-2024 usan Saber 35 y Hacer
    35; 2025 ya usa 45 y 40. Normalizar todo contra los topes de hoy leería un
    35/35 perfecto como un 78, que es la clase de error que no avisa.
    """

    def test_la_escala_vigente_es_la_de_la_rm(self):
        assert (ESCALA_VIGENTE.being, ESCALA_VIGENTE.knowing) == (10.0, 45.0)
        assert (ESCALA_VIGENTE.doing, ESCALA_VIGENTE.deciding) == (40.0, 5.0)

    def test_toda_escala_suma_cien(self):
        for gestion, esc in ESCALAS_POR_GESTION.items():
            assert esc.total() == 100.0, f"gestion {gestion} no suma 100"

    def test_una_gestion_desconocida_usa_la_vigente(self):
        assert escala_de(2031) is ESCALA_VIGENTE
        assert escala_de(None) is ESCALA_VIGENTE

    def test_una_gestion_sin_mapear_avisa(self, caplog):
        """Normalizar contra la escala equivocada no rompe nada, y ese es el peligro."""
        with caplog.at_level(logging.WARNING, logger="ue6_ia.contract"):
            escala_de(2019)
        assert "2019" in caplog.text

    def test_sin_gestion_no_avisa(self):
        # Es el caso del sistema, donde la vigente es la respuesta correcta.
        with caplog_vacio() as registros:
            escala_de(None)
        assert registros == []

    def test_la_suma_admite_pesos_no_enteros(self):
        # Un reparto valido con decimales no puede rebotar por como flota el binario.
        Escala(being=10.0, knowing=45.1, doing=39.9, deciding=5.0)

    def test_una_dimension_sin_puntos_se_rechaza(self):
        # Suma 100 igual, pero un tope en cero pasa las guardas de rango y revienta
        # recien al dividir, con un error que no dice cual dimension fue.
        with pytest.raises(ValueError, match="being"):
            Escala(being=0.0, knowing=50.0, doing=50.0, deciding=0.0)

    def test_dos_mil_veintitres_pondera_distinto(self):
        assert escala_de(2023).knowing == 35.0
        assert escala_de(2025).knowing == 45.0

    def test_el_maximo_de_su_epoca_vale_cien_en_cualquier_gestion(self):
        vieja = escala_de(2023)
        f = expandir(obs(being=[10.0], knowing=[35.0], doing=[35.0], deciding=[10.0]),
                     escala=vieja)
        assert f["knowing_mean"] == pytest.approx(100.0)
        assert f["doing_mean"] == pytest.approx(100.0)

    def test_la_misma_nota_significa_distinto_en_distinta_gestion(self):
        nota = obs(knowing=[35.0])
        assert expandir(nota, escala=escala_de(2023))["knowing_mean"] == pytest.approx(100.0)
        hoy = expandir(nota, escala=escala_de(2025))["knowing_mean"]
        assert hoy == pytest.approx(77.8, abs=0.1)

    def test_una_nota_valida_en_2023_no_se_rechaza_por_el_tope_de_hoy(self):
        # Decidir valia 10 puntos entonces y vale 5 ahora.
        expandir(obs(deciding=[8.0]), escala=escala_de(2023))
        with pytest.raises(ValueError, match="tope"):
            expandir(obs(deciding=[8.0]), escala=ESCALA_VIGENTE)

    def test_una_escala_que_no_suma_cien_se_rechaza(self):
        with pytest.raises(ValueError, match="100"):
            Escala(being=10.0, knowing=45.0, doing=40.0, deciding=99.0)


class TestEstadisticos:
    """El promedio esconde que 90/90/20 y 67/67/66 son cosas distintas."""

    def test_una_sola_nota_no_tiene_dispersion(self):
        f = expandir(obs(knowing=[36.0]))
        assert f["knowing_count"] == 1
        # La dispersion de un punto SI es cero: esto se midio.
        assert f["knowing_std"] == pytest.approx(0.0)
        assert f["knowing_min"] == f["knowing_max"] == f["knowing_mean"]

    def test_una_sola_nota_no_tiene_tendencia_medible(self):
        # La pendiente de un punto no existe. Un cero se leeria como "se midio y
        # no cambio", que es la confusion entre faltante y medido.
        assert expandir(obs(doing=[30.0]))["doing_trend"] is None

    def test_con_dos_notas_ya_hay_tendencia(self):
        assert expandir(obs(doing=[20.0, 30.0]))["doing_trend"] == pytest.approx(25.0)

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
