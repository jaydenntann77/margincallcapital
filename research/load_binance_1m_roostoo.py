"""Shim → research/db/load_binance_1m_roostoo.py (see research/db/README.md)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET = _HERE / "db" / "load_binance_1m_roostoo.py"


def main() -> None:
    r = subprocess.run([sys.executable, str(_TARGET)] + sys.argv[1:])
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
