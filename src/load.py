"""Loading the monthly BTS Reporting Carrier On-Time Performance CSVs.

The raw files are never modified. Everything here reads them into memory with
explicit dtypes so the full year fits comfortably in RAM.
"""

import re
from pathlib import Path

import pandas as pd
from pandas.api.types import union_categoricals

# Month name prefixes as they appear in the downloaded file names
# (e.g. TransStatsJAN2025.csv, TransStatsJUNE2025.csv, TransStatsAUG2025.csv.csv).
MONTH_PREFIX = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Airport/market ID columns we never use (ORIGIN and DEST codes cover them).
DROP_COLS = ["DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID", "DEST_CITY_MARKET_ID"]

# dtypes applied at read time. Text columns become categories (a handful of
# distinct values repeated millions of times), numbers become 32-bit or smaller.
# Columns that can be blank (delays, flags, causes) must stay float because a
# plain integer column cannot hold NaN; they are converted in src/clean.py.
READ_DTYPES = {
    "YEAR": "int16",
    "MONTH": "int8",
    "DAY_OF_MONTH": "int8",
    "DAY_OF_WEEK": "int8",
    "FL_DATE": "category",          # parsed to datetime later (only ~31 values per file)
    "OP_UNIQUE_CARRIER": "category",
    "TAIL_NUM": "category",
    "OP_CARRIER_FL_NUM": "int32",
    "ORIGIN": "category",
    "ORIGIN_STATE_ABR": "category",
    "DEST": "category",
    "DEST_STATE_ABR": "category",
    "CRS_DEP_TIME": "int16",        # HHMM stored as a number, e.g. 835 = 08:35
    "DEP_DELAY": "float32",
    "DEP_DELAY_NEW": "float32",
    "DEP_DEL15": "float32",
    "DEP_TIME_BLK": "category",
    "CRS_ARR_TIME": "int16",
    "ARR_DELAY": "float32",
    "ARR_DEL15": "float32",
    "CANCELLED": "float32",
    "CANCELLATION_CODE": "category",
    "DIVERTED": "float32",
    "CRS_ELAPSED_TIME": "float32",
    "DISTANCE": "float32",
    "CARRIER_DELAY": "float32",
    "WEATHER_DELAY": "float32",
    "NAS_DELAY": "float32",
    "SECURITY_DELAY": "float32",
    "LATE_AIRCRAFT_DELAY": "float32",
}
USE_COLS = list(READ_DTYPES)
CATEGORY_COLS = [c for c, t in READ_DTYPES.items() if t == "category"]


def find_month_files(raw_bts_dir):
    """Map month number -> CSV path, using the month name inside each file name.

    File names are inconsistent (JUNE vs JUN, a doubled .csv.csv extension), so we
    match on the first three letters of the month name instead of an exact pattern.
    """
    raw_bts_dir = Path(raw_bts_dir)
    files = {}
    for path in sorted(raw_bts_dir.glob("*.csv*")):
        match = re.search(r"([A-Za-z]{3,9})(20\d\d)", path.name.replace("TransStats", ""))
        if match is None or match.group(1)[:3].upper() not in MONTH_PREFIX:
            raise ValueError(f"Cannot tell which month this file holds: {path.name}")
        month = MONTH_PREFIX[match.group(1)[:3].upper()]
        if month in files:
            raise ValueError(f"Two files for month {month}: {files[month].name}, {path.name}")
        files[month] = path
    missing = sorted(set(range(1, 13)) - set(files))
    if missing:
        raise ValueError(f"No file found for months {missing}")
    return dict(sorted(files.items()))


def load_month(path):
    """Read one monthly CSV with the columns and dtypes we need."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in USE_COLS if c not in header]
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns {missing}")
    return pd.read_csv(path, usecols=USE_COLS, dtype=READ_DTYPES)


def _align_categories(frames):
    """Give every month the same category list per column so concat keeps them as
    categories (otherwise pandas silently falls back to slow, memory-hungry strings)."""
    for col in CATEGORY_COLS:
        union = union_categoricals([f[col] for f in frames]).categories.sort_values()
        for f in frames:
            f[col] = f[col].cat.set_categories(union)
    return frames


def load_year(raw_bts_dir, verbose=True):
    """Load all twelve months.

    Returns
    -------
    df : DataFrame with one row per scheduled flight.
    file_log : DataFrame with one row per file (name, size on disk, rows read).
    """
    files = find_month_files(raw_bts_dir)
    frames, log = [], []
    for month, path in files.items():
        frame = load_month(path)
        months_inside = sorted(frame["MONTH"].unique().tolist())
        if months_inside != [month]:
            raise ValueError(f"{path.name} should hold month {month} but contains {months_inside}")
        frames.append(frame)
        log.append({
            "month": month,
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "rows": len(frame),
        })
        if verbose:
            print(f"  loaded {path.name:<28} {len(frame):>9,} rows")
    frames = _align_categories(frames)
    df = pd.concat(frames, ignore_index=True)
    return df, pd.DataFrame(log)
