"""Inspector de hojas Excel para mapear layout (xlsb y xlsx).

Uso:
  python scripts/inspect_sheet.py "<archivo>"              # lista hojas
  python scripts/inspect_sheet.py "<archivo>" "<hoja>" 20  # vuelca 20 filas

Util para confirmar los indices de columna en config.yaml (p.ej. asistencia).
"""

from __future__ import annotations

import sys
from pathlib import Path

# permitir importar el paquete sin instalar
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ue6_ia.ingestion.excel_reader import list_sheets, read_sheet_grid  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    path = Path(sys.argv[1])
    if len(sys.argv) == 2:
        print("HOJAS:", list_sheets(path))
        return
    sheet = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    grid = read_sheet_grid(path, sheet)
    for i, row in enumerate(grid[:n]):
        vals = [("" if c is None else c) for c in row]
        while vals and vals[-1] == "":
            vals.pop()
        print(i, vals)


if __name__ == "__main__":
    main()
