"""Features for the departure-delay classifier (Q2).

Rule for everything in this file: a feature may only use information that would
exist about two hours before the flight's scheduled departure. That means the
published schedule, the calendar, and outcomes of *other* flights that finished
before the cutoff. Outcome columns of the flight itself (delays, causes,
cancellation, diversion) are never used.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

CUTOFF_MIN = 120   # predictions are made two hours before scheduled departure

# Major US travel holidays in and around 2025. A flight within three days of one
# falls in the holiday travel window.
MAJOR_HOLIDAYS = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-05-26",  # Memorial Day
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    "2026-01-01",  # New Year's Day (affects the last days of December)
])
HOLIDAY_WINDOW_DAYS = 3

# Scheduled ground times below this are not something an airline would plan; in
# this data they appear when an aircraft was swapped onto a flight during the day
# (see the notebook), so they reflect day-of decisions rather than the schedule.
MIN_PLAUSIBLE_TURN_MIN = 30

TURN_BINS = [MIN_PLAUSIBLE_TURN_MIN, 45, 60, 90, 180, np.inf]
TURN_LABELS = ["30-44", "45-59", "60-89", "90-179", "180+"]
TURN_FILL = "45-59"   # most common bucket, used when the recorded sequence is irregular


def add_calendar_features(df):
    """holiday_window: 1 if the flight date is within 3 days of a major holiday."""
    window = pd.DatetimeIndex([
        day + pd.Timedelta(days=k)
        for day in MAJOR_HOLIDAYS
        for k in range(-HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS + 1)
    ])
    df["holiday_window"] = df["FL_DATE"].isin(window).astype("int8")
    return df


def add_congestion_features(df):
    """Scheduled traffic in the same hour, a proxy for congestion.

    origin_hour_deps: flights scheduled to depart the same origin in the same clock hour
    dest_hour_arrs:   flights scheduled to arrive at the same destination in the same hour
    Counts include flights that were later cancelled, because they were on the
    published schedule. Only the carriers that report to BTS are counted.
    """
    dep_slot = df["sched_dep"].dt.floor("h")
    arr_slot = df["sched_arr"].dt.floor("h")
    df["origin_hour_deps"] = (
        df.groupby([df["ORIGIN"], dep_slot], observed=True)["ORIGIN"].transform("size").astype("int16")
    )
    df["dest_hour_arrs"] = (
        df.groupby([df["DEST"], arr_slot], observed=True)["DEST"].transform("size").astype("int16")
    )
    return df


def add_route_keys(df):
    """Integer keys for high-cardinality combinations (cheaper than strings).

    route:          origin-destination pair
    carrier_origin: carrier at an origin airport (e.g. a carrier's own hub)
    """
    origin = df["ORIGIN"].cat.codes.astype("int32")
    dest = df["DEST"].cat.codes.astype("int32")
    carrier = df["OP_UNIQUE_CARRIER"].cat.codes.astype("int32")
    df["route"] = origin * 1000 + dest
    df["carrier_origin"] = carrier * 1000 + origin
    return df


def add_rotation_features(df, planned):
    """Features from the aircraft's planned sequence of flights that day.

    `planned` is build_rotations(...) over every flight that had a tail number
    (cancelled ones included, since they were part of the plan), ordered by
    scheduled departure.

    leg_bucket       position in the aircraft's day: "1" ... "5", "6+"
    turn_bucket      scheduled ground time before this leg, or "first leg"
    prev_landed_by_cutoff   1 if the previous leg had landed at this airport at
                     least two hours before this leg's scheduled departure
    prev_delay_at_cutoff    that previous leg's arrival delay (minutes, capped)
                     when it had landed by the cutoff, otherwise 0

    Guard against day-of information: when the recorded sequence is irregular
    (chain break, or a scheduled ground time under MIN_PLAUSIBLE_TURN_MIN), the
    tail assignment was most likely changed on the day in response to delays. For
    those legs the turnaround is replaced by the most common bucket, with no flag,
    so the model cannot single them out. The column `irregular_sequence` marks
    them; it is NOT a model feature, only used to keep them out of training.
    """
    rot = planned.reindex(df.index)
    has_tail = rot["leg_num"].notna()

    later_leg = rot["leg_num"].gt(1)
    df["irregular_sequence"] = (
        later_leg
        & (rot["chain_break"].eq(True)
           | rot["sched_turn_min"].isna()
           | rot["sched_turn_min"].lt(MIN_PLAUSIBLE_TURN_MIN))
    ).to_numpy()

    df["leg_bucket"] = pd.cut(
        rot["leg_num"].fillna(1), bins=[0, 1, 2, 3, 4, 5, np.inf],
        labels=["1", "2", "3", "4", "5", "6+"],
    )

    first_leg = rot["leg_num"].eq(1) | ~has_tail
    # Ground times under the plausible minimum, and chain breaks, come out of pd.cut
    # as missing; they get the most common bucket. First legs get their own label.
    turn = pd.cut(rot["sched_turn_min"], bins=TURN_BINS, labels=TURN_LABELS, right=False)
    turn = turn.cat.add_categories("first leg").fillna(TURN_FILL)
    turn[first_leg] = "first leg"
    df["turn_bucket"] = turn.cat.reorder_categories(["first leg"] + TURN_LABELS)

    cutoff = df["sched_dep"] - pd.Timedelta(minutes=CUTOFF_MIN)
    landed = (
        rot["prev_actual_arr"].notna()
        & ~rot["chain_break"].eq(True)
        & (rot["prev_actual_arr"] <= cutoff)
    )
    df["prev_landed_by_cutoff"] = landed.astype("int8")
    df["prev_delay_at_cutoff"] = (
        rot["prev_arr_delay"].clip(-60, 300).where(landed, 0).astype("float32")
    )
    return df


def make_target_encoder(seed):
    """Target encoder for a binary target with reproducible 5-fold cross-fitting.

    During fit, each training row is encoded using delay rates computed from the
    other four folds, so a row never sees its own label. scikit-learn 1.9 moved the
    shuffling seed from `random_state` to the `cv` splitter; this works either way.
    """
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    if version >= (1, 9):
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", random_state=seed)


def historical_rate(train, other, keys, target="DEP_DEL15", prior_weight=50):
    """Baseline score: the share of training flights with the same `keys` that
    departed 15+ minutes late, shrunk toward the overall rate for small groups.

    smoothed rate = (delayed + prior_weight * overall_rate) / (flights + prior_weight)

    Groups never seen in training get the overall training rate.
    """
    y = train[target].astype(float)
    overall = y.mean()
    stats = y.groupby([train[k] for k in keys], observed=True).agg(["sum", "count"])
    rate = (stats["sum"] + prior_weight * overall) / (stats["count"] + prior_weight)
    idx = pd.MultiIndex.from_arrays([other[k] for k in keys])
    return pd.Series(rate.reindex(idx).to_numpy(), index=other.index).fillna(overall)
