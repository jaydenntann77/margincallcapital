# DuckDB refresh & inspect

All scripts that **download / rebuild** OHLCV and the **inspection** notebook live here.  
EDA notebooks under `research/` only **read** the database.

## Refresh — full Roostoo universe (~67 USD pairs)

From **repo root**:

```bash
source .venv-research/bin/activate
python research/db/load_1m_roostoo_universe.py --db data/research_roostoo_1m.duckdb --months 3
python research/db/load_1m_roostoo_universe.py --no-fetch-5s   # faster, no 5s bars
```

Outputs:

- `data/research_roostoo_1m.duckdb`
- `data/roostoo_universe.json` (snapshot of Roostoo `exchangeInfo`)

Intervals stored: `1m`, `5m`, `15m`, `1h`, and optionally `5s` (unless `--no-fetch-5s`).

## Refresh — legacy 16-token Binance list

```bash
python research/db/load_binance_1m_roostoo.py --db data/research_1m.duckdb --months 3
```

Or:

```bash
python research/db/setup_duckdb.py
```

## Inspect

```bash
jupyter notebook research/db/inspect_duckdb.ipynb
```

## Why did data stop around the 1st of the month?

Two common causes (both addressed in the current loader):

1. **Old month list** — Previously, “last N months” used a rough `30 * i` day offset, which often **dropped the current UTC calendar month** from the download list. The loader now uses **calendar months** and includes the **current month**.

2. **Missing monthly zip for the current month** — Binance Vision’s **monthly** file `*-1m-YYYY-MM.zip` often **does not exist** until the month is complete. The loader **falls back to daily** zips under `data/spot/daily/klines/...` for any month where the monthly file is missing, so you get data through **today (UTC)** when Vision publishes daily files.

After refreshing, confirm `MAX(ts)` per interval in `inspect_duckdb.ipynb`.

## Layout

| File | Purpose |
|------|---------|
| `roostoo_universe.py` | Fetch Roostoo `exchangeInfo`, parse USD pairs |
| `binance_vision.py` | Calendar month list, monthly + daily Vision klines |
| `load_1m_roostoo_universe.py` | Full universe → DuckDB (Binance + Coinbase fallback) |
| `load_binance_1m_roostoo.py` | Fixed token list → DuckDB |
| `setup_duckdb.py` | Run legacy loader + verify |
| `inspect_duckdb.ipynb` | Schema, counts, date ranges |

## Backward-compatible entry points

These forward to `research/db/`:

- `research/load_1m_roostoo_universe.py`
- `research/load_binance_1m_roostoo.py`
- `research/setup_duckdb.py`
