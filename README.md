# Cascading Delays: What Actually Makes U.S. Flights Late

CSI 2810 team project. Using the BTS Reporting Carrier On-Time Performance data for 2025 (7.0 million flights), we rebuild each aircraft's daily sequence of flights from its tail number to measure how delay propagates, decompose delay minutes by cause, and predict 15-minute departure delays two hours ahead.

## Setup

1. Install the course stack: `pip install -r requirements.txt`
2. Copy `config.example.py` to `config.py` and set `DATA_DIR` to your local copy of the shared Google Drive data folder (the folder containing `raw/` and `processed/`). `config.py` is gitignored, so each teammate keeps their own path.
3. In Google Drive for desktop, mark `raw/bts` as *Available offline* so the 1.2 GB of CSVs stay on disk between runs.

## Run order

| Notebook | What it does | Reads | Writes |
|---|---|---|---|
| `notebooks/02_progress_submission.ipynb` | cleaning of all 12 months, aircraft rotations, EDA, baseline and logistic regression models | `raw/bts/` | `processed/flights_2025_clean.parquet`, `figures/*.png` |

The notebook runs top to bottom in about 2 minutes once the raw files are local. All random steps use `SEED = 42`.

## Layout

```
config.example.py   template for config.py (DATA_DIR)
notebooks/          analysis notebooks, numbered in run order
src/                reusable code
  load.py           reading the monthly CSVs with compact dtypes
  clean.py          type conversions, scheduled timestamps, missing values, flags, checks
  rotations.py      rebuilding each aircraft's daily sequence of flights
  features.py       model features available two hours before departure
  evaluate.py       PR-AUC, threshold selection, precision and recall
  viz.py            shared plot styling
figures/            exported PNGs for the summary and slides
reports/            proposal, progress summary, final summary
```

Data never goes in git. It lives in the shared Drive folder: `raw/bts/` (monthly BTS CSVs, never modified), `raw/weather/` and `raw/reference/` (filled later by download scripts), and `processed/` (pipeline output).
