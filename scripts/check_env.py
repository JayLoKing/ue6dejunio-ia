"""Verifica que el entorno este listo para entrenar.

Uso:  python scripts/check_env.py

En Linux nativo (Mint, Ubuntu) corre tal cual. En Windows hay que estar adentro
de WSL2: TF Decision Forests no publica wheel para Windows.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("VERIFICACION DE ENTORNO - UE6 IA")
    print("=" * 60)
    print(f"Python   : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    ok = True
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 11):
        print(f"  [!] Se recomienda Python 3.11 (tienes {major}.{minor}). "
              "TF/TF-DF no soportan 3.13/3.14.")
        ok = False

    if platform.system() == "Windows":
        print("  [!] Estas en Windows nativo. TF Decision Forests NO funciona aqui. "
              "Usa WSL2, o Linux nativo. Ver README.md.")
        ok = False

    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}  (entrenamiento en CPU)")
    except Exception as e:  # noqa: BLE001 - cualquier fallo de import es un no
        print(f"  [!] TensorFlow no importable: {e}")
        ok = False

    try:
        import tensorflow_decision_forests as tfdf

        print(f"TF-DF     : {tfdf.__version__}")
    except Exception as e:  # noqa: BLE001 - idem: importa que no anda, no por que
        print(f"  [!] TF-DF no importable: {e}")
        ok = False

    for mod in ("pandas", "pyxlsb", "openpyxl", "fastapi"):
        try:
            m = __import__(mod)
            print(f"{mod:<10}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001 - el reporte lista el fallo y sigue
            print(f"  [!] {mod} no importable: {e}")
            ok = False

    ok = _revisar_datos() and ok

    print("=" * 60)
    print("RESULTADO:", "LISTO" if ok else "HAY PROBLEMAS (ver [!] arriba)")
    return 0 if ok else 1


def _revisar_datos() -> bool:
    """Que la ruta de registros exista y tenga adentro lo que el pipeline busca.

    Es el error mas facil de cometer al mover el proyecto de maquina, y el que
    peor avisa: sin las carpetas, `discover_cursos` no encuentra ningun curso y el
    dataset sale vacio con una sola linea de log.
    """
    print("-" * 60)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ue6_ia.config import get_config

        cfg = get_config()
    except Exception as e:  # noqa: BLE001 - sin config no hay nada que verificar
        print(f"  [!] No se pudo leer la configuracion: {e}")
        return False

    print(f"UE6_DATA_ROOT: {cfg.data_root}")
    if not cfg.data_root.is_dir():
        print("  [!] Esa carpeta no existe. En Linux es una ruta normal "
              "(/home/usuario/...); el prefijo /mnt/ es solo de WSL.")
        return False

    # Los nombres los declara config.yaml: repetirlos aca haria que renombrar uno
    # dejara este chequeo avisando de una carpeta que el pipeline lee sin problema.
    ok = True
    for etiqueta, ruta in (("pasados", cfg.dir_pasados), ("2026", cfg.dir_2026)):
        if ruta.is_dir():
            print(f"  ok  {ruta.name}")
        else:
            print(f"  [!] falta la carpeta de registros {etiqueta}: {ruta}")
            ok = False
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
