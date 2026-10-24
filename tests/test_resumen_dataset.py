"""El resumen que `build-dataset` imprime al terminar, y que se cuenta y que no.

Cuenta filas, nunca personas: el resumen termina en una captura de pantalla del
documento, asi que no puede llevar un solo nombre.
"""

from __future__ import annotations

import pandas as pd

from ue6_ia.evaluation.consola import lineas_resumen_dataset
from ue6_ia.pipeline import ResumenDataset, completar_resumen
from ue6_ia.preprocessing.features import TARGET_COL


def _etiquetado() -> pd.DataFrame:
    """Dos estudiantes, una materia, tres trimestres: el tercero no tiene etiqueta."""
    filas = []
    for nombre in ("ANA PEREZ", "LUIS ROJAS"):
        for trimestre, clase in ((1, "SinRiesgo"), (2, "EnRiesgo"), (3, None)):
            filas.append(
                {
                    "gestion": 2024 if nombre.startswith("ANA") else 2025,
                    "trimestre": trimestre,
                    "nombre_key": nombre,
                    "area": "Matematicas",
                    TARGET_COL: clase,
                }
            )
    # Una fila de 1er trimestre cuyo siguiente no aparecio: tampoco tiene etiqueta.
    filas.append(
        {
            "gestion": 2025,
            "trimestre": 1,
            "nombre_key": "LUIS ROJAS",
            "area": "Lenguaje",
            TARGET_COL: None,
        }
    )
    return pd.DataFrame(filas)


class TestCompletarResumen:
    def test_cuenta_etiquetadas_y_las_que_quedan_sin_etiqueta_por_trimestre(self):
        etiquetado = _etiquetado()
        final = etiquetado[etiquetado[TARGET_COL].notna()].reset_index(drop=True)

        resumen = completar_resumen(ResumenDataset(gestiones=[2024, 2025]), etiquetado, final)

        assert resumen.filas_con_etiqueta == 4
        assert resumen.filas_sin_etiqueta == 3
        assert resumen.filas_sin_etiqueta_por_trimestre == {1: 1, 3: 2}
        assert resumen.filas_descartadas_sin_etiqueta == 3

    def test_describe_el_dataset_final_sin_nombres(self):
        etiquetado = _etiquetado()
        final = etiquetado[etiquetado[TARGET_COL].notna()].reset_index(drop=True)

        resumen = completar_resumen(ResumenDataset(gestiones=[2024, 2025]), etiquetado, final)

        assert resumen.filas_finales == 4
        assert resumen.estudiantes == 2
        assert resumen.materias == 1
        assert resumen.filas_por_gestion == {2024: 2, 2025: 2}
        assert resumen.filas_por_trimestre == {1: 2, 2: 2}
        assert resumen.distribucion_clases == {"EnRiesgo": 2, "SinRiesgo": 2}

    def test_sin_filtrar_no_descarta_nada(self):
        etiquetado = _etiquetado()

        resumen = completar_resumen(ResumenDataset(gestiones=[2024]), etiquetado, etiquetado)

        assert resumen.filas_descartadas_sin_etiqueta == 0
        assert resumen.filas_finales == len(etiquetado)


class TestLineasDelResumen:
    def test_el_texto_no_lleva_nombres_de_estudiantes(self):
        etiquetado = _etiquetado()
        final = etiquetado[etiquetado[TARGET_COL].notna()].reset_index(drop=True)
        resumen = completar_resumen(ResumenDataset(gestiones=[2024, 2025]), etiquetado, final)
        resumen.notas_fuera_de_rango = {"knowing": 3}

        texto = "\n".join(lineas_resumen_dataset(resumen))

        assert "ANA" not in texto and "LUIS" not in texto
        assert "Saber" in texto
        assert all(len(linea) <= 100 for linea in texto.splitlines())

    def test_una_dimension_sin_descartes_dice_cero_porque_se_conto(self):
        # Aca el cero si es una medicion: se revisaron las notas y ninguna estaba
        # fuera de rango. No es un dato faltante.
        resumen = ResumenDataset(gestiones=[2024])

        texto = "\n".join(lineas_resumen_dataset(resumen))

        assert "Ser" in texto
