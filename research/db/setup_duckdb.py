#!/usr/bin/env python3
"""
Run the legacy 16-token Binance loader and verify DuckDB.

Usage (repo root):
    python research/db/setup_duckdb.py
    python research/db/setup_duckdb.py --verify-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB_SCRIPT = Path(__file__).resolve().parent / "load_binance_1m_roostoo.py"


def ensure_data_dir() -> Path:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ data/ exists: {data_dir}")
    return data_dir


def run_loader(db_path: Path, months: int) -> None:
    cmd = [
        sys.executable,
        str(DB_SCRIPT),
        "--db",
        str(db_path),
        "--months",
        str(months),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def verify_db(db_path: Path) -> bool:
    import duckdb

    if not db_path.exists():
        print(f"✗ Database not found: {db_path}")
        return False

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        symbols = con.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE interval='1h' ORDER BY symbol"
        ).fetchall()
        intervals = con.execute("SELECT DISTINCT interval FROM ohlcv").fetchall()
        print("\n=== DuckDB verification ===")
        print(f"  Path: {db_path}")
        print(f"  Total rows: {rows:,}")
        print(f"  Symbols (1h): {[s[0] for s in symbols]}")
        print(f"  Intervals: {[i[0] for i in intervals]}")
        if rows == 0:
            print("✗ No data in ohlcv.")
            return False
        print("✓ OK")
        print("\nNext: EDA → research/Koo an's EDA.ipynb | full universe → research/db/README.md")
        return True
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Setup DuckDB (legacy 16-token loader)")
    p.add_argument("--db", default="data/research_1m.duckdb")
    p.add_argument("--months", type=int, default=3)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    db_path = ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    ensure_data_dir()

    if not args.verify_only:
        run_loader(db_path, args.months)

    sys.exit(0 if verify_db(db_path) else 1)


if __name__ == "__main__":
    main()
