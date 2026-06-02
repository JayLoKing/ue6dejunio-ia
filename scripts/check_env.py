"""Verifica que el entorno este listo para entrenar.

Uso (dentro de WSL2):  python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys


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
              "Usa WSL2 (Ubuntu). Ver README.md.")
        ok = False

    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}  (entrenamiento en CPU)")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] TensorFlow no importable: {e}")
        ok = False

    try:
        import tensorflow_decision_forests as tfdf

        print(f"TF-DF     : {tfdf.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] TF-DF no importable: {e}")
        ok = False

    for mod in ("pandas", "pyxlsb", "openpyxl", "fastapi"):
        try:
            m = __import__(mod)
            print(f"{mod:<10}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"  [!] {mod} no importable: {e}")
            ok = False

    print("=" * 60)
    print("RESULTADO:", "LISTO" if ok else "HAY PROBLEMAS (ver [!] arriba)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
