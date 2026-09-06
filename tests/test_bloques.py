"""Deteccion de los bloques de criterios en una hoja de registro.

Los encabezados son los medidos en los registros reales de 2023, 2025 y 2026, con
su forma de verdad: el nombre del bloque en una fila y su 'P R O M E D I O' en la
de abajo, con las letras espaciadas.
"""

from __future__ import annotations

import logging

from ue6_ia.ingestion.bloques import (
    aplanar_cabecera,
    columnas_de_criterios,
    localizar_bloque,
    normalizar,
)


def _fila(ancho: int, **celdas: str) -> list:
    fila = [None] * ancho
    for col, valor in celdas.items():
        fila[int(col[1:])] = valor
    return fila


# 2025, hoja LENG. SER y DECIDIR llegan ya resueltos desde 'EVAL SER Y DECIDIR'.
CAB_2025 = aplanar_cabecera([
    _fila(22, c1="DIMENSIONES", c2="P R O M E D I O     S E R", c3="SABER - 45",
          c11="HACER - 40", c19="P R O M E D I O    D E C I D I R",
          c20="AUTOEVALUACIÓN - SER Y D", c21="PROMEDIO TRIMESTRAL"),
    _fila(22, c3="DESCRIPCION DE LA VACACION", c4="HISTORIA DE SANDRA",
          c5="ME DUERMO TEMPRANO", c6="Comprencion del valor", c7="11 SELLOS",
          c8="LIBRO NOVELAS CORTAS", c9="12-04 POESIA",
          c10="P   R   O   M   E   D   I   O",
          c11="Desarrollo de habilidades", c12="CARATULA",
          c13="ELABORACION DE PROGRAMA", c14="ESCRIBIR PAPELOGRAFO",
          c15="VOLUMRN DE VOZ", c16="GESTUACION Y MOVIMIENTO", c17="EVALUACION",
          c18="P   R   O   M   E   D   I   O"),
])

# 2023: los bloques se anuncian con su nombre y su ponderacion, sin PROMEDIO.
CAB_2023 = aplanar_cabecera([
    _fila(23, c1="DIMENSIONES", c2="S E R - 10", c3="SABER - 35",
          c11="HACER - 35", c19="D E C I D I R - 10",
          c20="AUTOEVALUACIÓN - SER 5", c21="AUTOEVALUACIÓN - DECIDIR 5",
          c22="PROMEDIO TRIMESTRAL"),
    _fila(23, c3="LECTURA", c4="ESCRITURA", c5="Comprencion del valor",
          c6="crit 4", c7="crit 5", c8="crit 6", c9="crit 7",
          c10="P   R   O   M   E   D   I   O",
          c11="Desarrollo de habilidades", c12="crit b", c13="pancarta",
          c14="cuadernos", c15="crit e", c16="crit f", c17="crit g",
          c18="P   R   O   M   E   D   I   O"),
])

# 2026: SER vuelve a la hoja de area con criterios propios.
CAB_2026 = aplanar_cabecera([
    _fila(22, c1="DIMENSIONES", c2="SER - 10", c5="P R O M E D I O     S E R",
          c6="SABER - 45", c13="HACER - 40", c20="AUTOEVALUACIÓN (5)",
          c21="PROMEDIO TRIMESTRAL"),
    _fila(22, c12="P   R   O   M   E   D   I   O",
          c19="P   R   O   M   E   D   I   O"),
])


class TestNormalizar:
    def test_borra_acentos_espacios_y_mayusculas(self):
        assert normalizar("P R O M E D I O") == "PROMEDIO"
        assert normalizar("AUTOEVALUACIÓN (5)") == "AUTOEVALUACION(5)"

    def test_una_celda_vacia_es_texto_vacio(self):
        assert normalizar(None) == ""


class TestAplanarCabecera:
    """El nombre del bloque y su promedio viven en filas distintas."""

    def test_junta_las_dos_alturas_en_una_sola_columna(self):
        plana = aplanar_cabecera([
            _fila(3, c1="SABER - 45"),
            _fila(3, c1="Lectura", c2="P R O M E D I O"),
        ])
        assert plana[1].startswith("SABER-45")
        assert "LECTURA" in plana[1]
        assert plana[2] == "PROMEDIO"

    def test_una_columna_sin_nada_queda_vacia(self):
        assert aplanar_cabecera([_fila(2, c1="SABER")])[0] == ""


class TestLocalizarBloque:
    def test_encuentra_saber_con_su_promedio_de_la_fila_de_abajo(self):
        bloque = localizar_bloque(CAB_2025, "SABER")
        assert (bloque.inicio, bloque.promedio) == (3, 10)

    def test_hacer_no_comparte_promedio_con_saber(self):
        saber = localizar_bloque(CAB_2025, "SABER")
        hacer = localizar_bloque(CAB_2025, "HACER")
        assert (hacer.inicio, hacer.promedio) == (11, 18)
        assert saber.promedio != hacer.promedio

    def test_reconoce_el_mismo_bloque_con_otra_ponderacion(self):
        assert localizar_bloque(CAB_2023, "SABER").inicio == 3
        assert localizar_bloque(CAB_2025, "SABER").inicio == 3

    def test_no_toma_el_promedio_trimestral_como_cierre(self):
        assert localizar_bloque(CAB_2025, "SABER").promedio != 21

    def test_una_dimension_ya_resuelta_no_abre_bloque(self):
        # 2025 trae 'PROMEDIO SER': un total, no un encabezado.
        assert localizar_bloque(CAB_2025, "SER") is None

    def test_una_dimension_sin_promedio_propio_no_se_come_el_bloque_vecino(self):
        # 2023 pone 'S E R - 10' pegado a 'SABER - 35'. Sin esto, SER reclamaria
        # como suyos los criterios de SABER.
        assert localizar_bloque(CAB_2023, "SER") is None

    def test_un_area_que_el_docente_no_dicta_no_tiene_bloques(self):
        assert localizar_bloque(aplanar_cabecera([_fila(2, c1="DIMENSIONES")]), "SABER") is None

    def test_el_vecino_gana_al_promedio_cuando_comparten_columna(self):
        """El texto de una columna es la union de sus dos alturas.

        Si el encabezado del vecino y su promedio caen en la misma columna, esa
        columna dice 'HACER-40 PROMEDIO'. Preguntando primero por el promedio,
        SABER la tomaria como su cierre y se quedaria con los criterios de HACER.
        """
        cab = aplanar_cabecera([
            _fila(5, c1="SABER - 45", c3="HACER - 40"),
            _fila(5, c2="Lectura", c3="P R O M E D I O"),
        ])
        assert localizar_bloque(cab, "SABER") is None

    def test_un_encabezado_sin_promedio_detras_se_avisa(self, caplog):
        # Distinto de las ausencias legitimas: la hoja no tiene la forma esperada.
        cab = aplanar_cabecera([_fila(3, c1="SABER - 45"), _fila(3, c2="Lectura")])
        with caplog.at_level(logging.WARNING, logger="ue6_ia.ingestion.bloques"):
            assert localizar_bloque(cab, "SABER") is None
        assert "SABER" in caplog.text

    def test_un_criterio_que_empieza_como_una_dimension_no_corta_el_bloque(self):
        """`normalizar` borra los espacios, y los criterios arrancan con verbo.

        'Ser responsable con sus tareas' queda 'SERRESPONSABLE...', que empieza con
        SER. Tomado por encabezado, corta SABER y le vacia los criterios sin que
        nada falle.
        """
        cab = aplanar_cabecera([
            _fila(5, c1="SABER - 45"),
            _fila(5, c2="Ser responsable con sus tareas",
                  c3="Hacer uso de instrumentos",
                  c4="P R O M E D I O"),
        ])
        assert columnas_de_criterios(cab, "SABER") == [1, 2, 3]

    def test_un_criterio_asi_tampoco_abre_un_bloque_ajeno(self):
        cab = aplanar_cabecera([
            _fila(4, c1="SABER - 45"),
            _fila(4, c2="Hacer uso de instrumentos", c3="P R O M E D I O"),
        ])
        assert localizar_bloque(cab, "HACER") is None

    def test_las_ausencias_legitimas_no_avisan(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ue6_ia.ingestion.bloques"):
            localizar_bloque(CAB_2025, "SER")
            localizar_bloque(CAB_2023, "SER")
        assert caplog.text == ""


class TestColumnasDeCriterios:
    """Las notas por criterio son las columnas entre el encabezado y su promedio."""

    def test_saber_2025_incluye_el_criterio_que_comparte_columna_con_el_titulo(self):
        # La col 3 dice 'SABER - 45' arriba y 'DESCRIPCION DE LA VACACION' abajo,
        # y el alumno tiene nota ahi. Arrancar en la 4 perdia un criterio por bloque.
        assert columnas_de_criterios(CAB_2025, "SABER") == [3, 4, 5, 6, 7, 8, 9]

    def test_hacer_2025(self):
        assert columnas_de_criterios(CAB_2025, "HACER") == [11, 12, 13, 14, 15, 16, 17]

    def test_los_bloques_no_se_pisan(self):
        saber = set(columnas_de_criterios(CAB_2025, "SABER"))
        hacer = set(columnas_de_criterios(CAB_2025, "HACER"))
        assert saber and hacer and saber.isdisjoint(hacer)

    def test_el_promedio_nunca_es_un_criterio(self):
        assert 10 not in columnas_de_criterios(CAB_2025, "SABER")
        assert 18 not in columnas_de_criterios(CAB_2025, "HACER")

    def test_el_tramo_incluye_la_columna_del_encabezado(self):
        """Suele llevar tambien el primer criterio, y cuando no, viene vacia.

        En 2023 la columna de DECIDIR trae el valor sin nombrar criterio debajo;
        en 2026 la de SER es solo rotulo y los alumnos no tienen nada ahi.
        Incluirla siempre acierta en los dos casos.
        """
        assert columnas_de_criterios(CAB_2026, "SER") == [2, 3, 4]

    def test_funciona_igual_en_2023(self):
        assert columnas_de_criterios(CAB_2023, "SABER") == [3, 4, 5, 6, 7, 8, 9]
        assert columnas_de_criterios(CAB_2023, "HACER") == [11, 12, 13, 14, 15, 16, 17]

    def test_ser_no_aporta_criterios_donde_llega_resuelto(self):
        assert columnas_de_criterios(CAB_2023, "SER") == []
        assert columnas_de_criterios(CAB_2025, "SER") == []

    def test_en_2026_ser_si_tiene_criterios_propios(self):
        assert columnas_de_criterios(CAB_2026, "SABER") == [6, 7, 8, 9, 10, 11]
