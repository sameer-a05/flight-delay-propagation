"""Rebuilding aircraft rotations: the ordered sequence of flights one aircraft
operates on one day.

A rotation is keyed on carrier + tail number + flight date (not tail + date alone,
so a leased aircraft flown by two carriers would form two rotations).

Two orderings are used:
  * actual departure time, for describing what each aircraft really flew (Q3).
    When airlines swap aircraft during the day, the scheduled times of the flights
    a tail ends up flying can overlap, so sorting by schedule would misorder them.
  * scheduled departure time, for model features (Q2), which may not use actual times.
"""

import pandas as pd

ROTATION_KEYS = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "FL_DATE"]


def build_rotations(df, mask, order_col="sched_dep"):
    """Order each aircraft's flights within a day and attach previous-leg information.

    Parameters
    ----------
    df : cleaned flight table (needs sched_dep, sched_arr from src.clean).
    mask : boolean Series choosing which flights to chain together, e.g. only
        flights that actually flew, or every flight that had a tail assigned.
    order_col : column used to order legs within an aircraft-day
        ("actual_dep" or "sched_dep").

    Returns
    -------
    DataFrame indexed like the selected rows of df, with columns
        leg_num          1 for the aircraft's first flight of the day, 2 for the next, ...
        n_legs           number of selected flights the aircraft has that day
        chain_break      True if the previous leg landed somewhere other than this
                         leg's origin (a data problem or an out-of-order sort)
        sched_turn_min   scheduled ground time between the previous leg's scheduled
                         arrival and this leg's scheduled departure (same airport, so
                         both times are on the same local clock); blank for first legs
                         and chain breaks
        prev_arr_delay   previous leg's arrival delay in minutes (blank if it was
                         cancelled or diverted)
        prev_actual_arr  when the previous leg actually landed (local time)
    """
    cols = list(dict.fromkeys(ROTATION_KEYS + [
        order_col, "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "sched_dep", "sched_arr", "ARR_DELAY",
    ]))
    sort_by = list(dict.fromkeys(ROTATION_KEYS + [order_col, "sched_dep", "OP_CARRIER_FL_NUM"]))
    r = df.loc[mask, cols].sort_values(sort_by, kind="stable")

    # A new aircraft-day starts wherever carrier, tail or date changes from the row above.
    keys = r[ROTATION_KEYS]
    new_day = (keys != keys.shift()).any(axis=1)
    day_id = new_day.cumsum()

    out = pd.DataFrame(index=r.index)
    out["leg_num"] = (r.groupby(day_id).cumcount() + 1).astype("int16")
    out["n_legs"] = r.groupby(day_id)["ORIGIN"].transform("size").astype("int16")

    # Previous leg = the row above, unless this row starts a new aircraft-day.
    prev = r[["DEST", "sched_arr", "ARR_DELAY"]].shift()
    prev.loc[new_day] = None
    has_prev = ~new_day

    out["chain_break"] = (has_prev & (prev["DEST"] != r["ORIGIN"])).to_numpy()
    turn = (r["sched_dep"] - prev["sched_arr"]).dt.total_seconds() / 60
    out["sched_turn_min"] = turn.where(has_prev & ~out["chain_break"]).astype("float32")
    out["prev_arr_delay"] = prev["ARR_DELAY"].astype("float32")
    out["prev_actual_arr"] = prev["sched_arr"] + pd.to_timedelta(prev["ARR_DELAY"], unit="min")
    return out


def rotation_summary(rot):
    """Headline numbers for a set of rotations returned by build_rotations."""
    first = rot["leg_num"] == 1
    return pd.Series({
        "flights in rotations": len(rot),
        "aircraft-days": int(first.sum()),
        "mean legs per aircraft-day": rot.loc[first, "n_legs"].mean().round(2),
        "max legs in one day": int(rot["n_legs"].max()),
        "legs that follow an earlier leg": int((~first).sum()),
        "chain breaks": int(rot["chain_break"].sum()),
        "chain-break rate (% of non-first legs)": round(
            100 * rot["chain_break"].sum() / max((~first).sum(), 1), 3
        ),
    })
