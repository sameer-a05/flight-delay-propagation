"""Cleaning steps for the BTS on-time table.

Each function updates the DataFrame in place and also returns it, so the
notebook can run the steps one at a time and check the result after each one.
Raw files are never touched; the cleaned table is saved to processed/.
"""

import numpy as np
import pandas as pd

CAUSE_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
FL_DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"   # e.g. "1/1/2025 12:00:00 AM"

# Scheduled block speed outside this range (miles per hour) means the scheduled
# duration is a data-entry error: jets cruise around 500 mph, and even the shortest
# regional hops average well above 30 mph gate to gate.
MIN_SCHED_MPH = 30
MAX_SCHED_MPH = 650

# Departure delays at or above 12 hours (roughly the 99.9th percentile) usually mean
# the flight was re-timed to a later slot rather than held at the gate.
EXTREME_DELAY_MIN = 720

# Guam and Saipan are the only airports in the data west of the International Date
# Line (UTC+10). Their clocks are 20 hours ahead of Honolulu, which the +/-12 hour
# wrap in add_schedule_times cannot see, so those flights get a one-day correction.
DATELINE_AIRPORTS = ["GUM", "SPN"]


def is_one(s):
    """True where a 0/1 column equals 1, treating missing values as False."""
    return s.eq(1).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Type conversions
# ---------------------------------------------------------------------------

def parse_flight_date(df):
    """FL_DATE arrives as text like '1/1/2025 12:00:00 AM'. Parse it to a real date.

    There are only 365 distinct strings, so we parse the category labels once and
    map them back to the rows instead of parsing seven million strings.
    Returns the frame and the number of rows whose parsed date disagrees with
    YEAR / MONTH / DAY_OF_MONTH (expected to be 0).
    """
    labels = df["FL_DATE"].cat.categories
    parsed = pd.DatetimeIndex(pd.to_datetime(labels, format=FL_DATE_FORMAT))
    df["FL_DATE"] = parsed[df["FL_DATE"].cat.codes.to_numpy()]
    expected = pd.to_datetime(
        pd.DataFrame({"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY_OF_MONTH"]})
    )
    mismatches = int((df["FL_DATE"] != expected).sum())
    return df, mismatches


def convert_flags(df):
    """0/1 flags load as float because some are blank. Make them integers.

    CANCELLED and DIVERTED are never blank, so they become plain int8.
    DEP_DEL15 is blank for cancelled flights and ARR_DEL15 for cancelled or diverted
    flights; those use pandas' nullable Int8 so 'unknown' stays distinct from 0.
    """
    for col in ["CANCELLED", "DIVERTED"]:
        df[col] = df[col].astype("int8")
    for col in ["DEP_DEL15", "ARR_DEL15"]:
        df[col] = df[col].astype("Int8")
    return df


def unify_airport_categories(df):
    """Give ORIGIN and DEST the same list of airport codes so they can be compared
    directly (needed to check that one leg's destination is the next leg's origin)."""
    airports = df["ORIGIN"].cat.categories.union(df["DEST"].cat.categories)
    df["ORIGIN"] = df["ORIGIN"].cat.set_categories(airports)
    df["DEST"] = df["DEST"].cat.set_categories(airports)
    return df


def standardize_tails(df):
    """Allegiant (G4) reports tail numbers without the leading 'N' of a US
    registration (e.g. '812NV' instead of 'N812NV'). Add it back so every tail
    follows the same format. Returns the frame and the number of tails changed."""
    cats = df["TAIL_NUM"].cat.categories
    fixed = [t if t.startswith("N") else "N" + t for t in cats]
    if len(set(fixed)) != len(fixed):
        raise ValueError("Adding 'N' would merge two different tails; check before standardizing.")
    n_changed = sum(a != b for a, b in zip(cats, fixed))
    df["TAIL_NUM"] = df["TAIL_NUM"].cat.rename_categories(fixed)
    return df, n_changed


# ---------------------------------------------------------------------------
# Scheduled times
# ---------------------------------------------------------------------------

def hhmm_to_minutes(hhmm):
    """835 -> 515 (08:35 is 515 minutes after midnight). 2400 -> 1440, i.e. the
    midnight at the end of the day, which keeps it after 2359 when sorting."""
    hhmm = hhmm.astype("int32")
    return (hhmm // 100) * 60 + hhmm % 100


def fix_scheduled_elapsed(df):
    """Replace impossible scheduled durations with the route's median duration.

    A duration is impossible if it is missing, zero or negative, or implies a
    gate-to-gate speed outside MIN_SCHED_MPH..MAX_SCHED_MPH.
    Adds the flag column `elapsed_imputed`.
    """
    elapsed = df["CRS_ELAPSED_TIME"]
    mph = df["DISTANCE"] / (elapsed / 60)
    bad = elapsed.isna() | (elapsed <= 0) | (mph < MIN_SCHED_MPH) | (mph > MAX_SCHED_MPH)
    route_median = (
        df.loc[~bad].groupby(["ORIGIN", "DEST"], observed=True)["CRS_ELAPSED_TIME"].median()
    )
    bad_routes = list(zip(df.loc[bad, "ORIGIN"], df.loc[bad, "DEST"]))
    df.loc[bad, "CRS_ELAPSED_TIME"] = route_median.reindex(bad_routes).to_numpy()
    df["elapsed_imputed"] = bad.to_numpy()
    return df


def add_schedule_times(df):
    """Build scheduled departure/arrival timestamps (local time at each airport).

    BTS gives times as HHMM in each airport's local clock and no time zones. We use
    the identity  arrival = departure + elapsed + (time-zone shift)  to recover the
    shift between the two airports' clocks, which is always a whole number of hours
    for correct records. That lets us place each arrival on the right calendar day:
    red-eyes land the next morning, while e.g. a 07:00 ATL departure can land
    06:50 the same day in Central time.

    Adds: dep_min, arr_min, dep_hour, arr_hour, sched_dep, sched_arr,
    tz_shift_min, overnight, schedule_inconsistent, actual_dep.
    """
    dep_min = hhmm_to_minutes(df["CRS_DEP_TIME"])
    arr_min = hhmm_to_minutes(df["CRS_ARR_TIME"])
    elapsed = df["CRS_ELAPSED_TIME"].astype("float64")

    # Destination clock minus origin clock, wrapped into [-12 h, +12 h).
    shift = ((arr_min - (dep_min + elapsed) + 720) % 1440) - 720
    # Date-line crossings: flying west into Guam/Saipan adds a calendar day,
    # flying east out of them takes one away.
    into_dateline = df["DEST"].isin(DATELINE_AIRPORTS) & ~df["ORIGIN"].isin(DATELINE_AIRPORTS)
    out_of_dateline = df["ORIGIN"].isin(DATELINE_AIRPORTS) & ~df["DEST"].isin(DATELINE_AIRPORTS)
    shift = shift + 1440 * into_dateline - 1440 * out_of_dateline
    # Whole days between the departure date and the arrival date (usually 0 or 1).
    day_offset = ((dep_min + elapsed + shift - arr_min) / 1440).round()
    # If elapsed is still unknown, fall back to "arrives the next day if the clock wrapped".
    day_offset = day_offset.fillna((arr_min < dep_min).astype(float))

    df["dep_min"] = dep_min.astype("int16")
    df["arr_min"] = arr_min.astype("int16")
    df["dep_hour"] = ((dep_min // 60) % 24).astype("int8")
    df["arr_hour"] = ((arr_min // 60) % 24).astype("int8")
    df["sched_dep"] = df["FL_DATE"] + pd.to_timedelta(dep_min, unit="min")
    df["sched_arr"] = df["FL_DATE"] + pd.to_timedelta(day_offset * 1440 + arr_min, unit="min")
    df["tz_shift_min"] = shift.astype("float32")
    df["overnight"] = (day_offset >= 1).to_numpy()
    df["schedule_inconsistent"] = (shift.notna() & (shift % 60 != 0)).to_numpy()
    # Actual pushback time (local). An outcome, so it is only used to order the legs an
    # aircraft really flew (src/rotations.py) and never as a model feature.
    df["actual_dep"] = df["sched_dep"] + pd.to_timedelta(df["DEP_DELAY"], unit="min")
    return df


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def fill_cause_columns(df):
    """The five cause columns are blank by design when a flight arrived less than
    15 minutes late. For those completed flights the true value is 0 minutes.
    Cancelled and diverted flights never arrived, so their causes stay blank.

    A handful of 15+ minute late arrivals report zero minutes for every cause,
    which contradicts the BTS rule that causes sum to the arrival delay. We mark
    those as unattributed (blank) instead of treating them as zero-cause delays.
    Adds the flag column `cause_unattributed`.
    """
    on_time_arrival = df["ARR_DEL15"].eq(0).fillna(False).astype(bool)
    df.loc[on_time_arrival, CAUSE_COLS] = 0.0
    late_arrival = is_one(df["ARR_DEL15"])
    unattributed = late_arrival & (df[CAUSE_COLS].sum(axis=1) == 0)
    df.loc[unattributed, CAUSE_COLS] = np.nan
    df["cause_unattributed"] = unattributed.to_numpy()
    return df


# ---------------------------------------------------------------------------
# Outliers and suspicious records (flagged, not deleted)
# ---------------------------------------------------------------------------

def flag_extreme_delays(df):
    """Flag departure delays of 12+ hours. They are real records and stay in the
    table, but they can dominate averages, so plots and regressions cap them."""
    df["extreme_delay"] = (df["DEP_DELAY"] >= EXTREME_DELAY_MIN).fillna(False).to_numpy()
    return df


def flag_tail_conflicts(df):
    """Flag flown flights where the same aircraft is recorded departing twice at the
    same scheduled minute (usually from two different airports), which is
    physically impossible and points to a mislabeled tail number."""
    flown = (df["CANCELLED"] == 0) & df["TAIL_NUM"].notna()
    key = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "FL_DATE", "dep_min"]
    dup = df.loc[flown].duplicated(key, keep=False)
    df["tail_conflict"] = False
    df.loc[dup[dup].index, "tail_conflict"] = True
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def consistency_checks(df):
    """Rule checks that should all return 0 violations on the cleaned table."""
    cancelled = df["CANCELLED"] == 1
    diverted = df["DIVERTED"] == 1
    completed = ~cancelled & ~diverted
    late_arr = is_one(df["ARR_DEL15"])
    attributed = late_arr & ~df["cause_unattributed"]
    cause_sum = df[CAUSE_COLS].sum(axis=1, min_count=5)
    dep_known = df["DEP_DELAY"].notna()
    arr_known = df["ARR_DELAY"].notna()

    checks = {
        "flight both cancelled and diverted": (cancelled & diverted).sum(),
        "missing DEP_DELAY on a flight that was not cancelled": (~dep_known & ~cancelled).sum(),
        "missing ARR_DELAY on a completed flight": (~arr_known & completed).sum(),
        "ARR_DELAY present on a cancelled or diverted flight": (arr_known & ~completed).sum(),
        "DEP_DEL15 disagrees with DEP_DELAY >= 15": (
            dep_known & (df["DEP_DEL15"].astype("float") != (df["DEP_DELAY"] >= 15))
        ).sum(),
        "ARR_DEL15 disagrees with ARR_DELAY >= 15": (
            arr_known & (df["ARR_DEL15"].astype("float") != (df["ARR_DELAY"] >= 15))
        ).sum(),
        "DEP_DELAY_NEW != max(DEP_DELAY, 0)": (
            dep_known & ((df["DEP_DELAY"].clip(lower=0) - df["DEP_DELAY_NEW"]).abs() > 0.5)
        ).sum(),
        "cause minutes do not sum to ARR_DELAY (late arrivals)": (
            attributed & ((cause_sum - df["ARR_DELAY"]).abs() > 0.5)
        ).sum(),
        "cause columns blank on a completed flight": (
            completed & arr_known & ~df["cause_unattributed"] & df[CAUSE_COLS].isna().any(axis=1)
        ).sum(),
        "missing TAIL_NUM on a flight that was not cancelled": (
            df["TAIL_NUM"].isna() & ~cancelled
        ).sum(),
        "exact duplicate rows": df.duplicated().sum(),
        "duplicate carrier + flight number + date + origin": df.duplicated(
            ["OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "FL_DATE", "ORIGIN"]
        ).sum(),
    }
    return pd.DataFrame({"violations": pd.Series(checks).astype(int)})
