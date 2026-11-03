"""Shared plot styling so every figure in the project looks consistent.

Color rules used throughout:
  * one series -> one color (BLUE); highlight a single item with ACCENT and
    leave the rest GRAY instead of giving every bar its own hue
  * categories with no order (delay causes, model names) -> CATEGORICAL, always
    in the same fixed order, so a cause keeps its color in every chart
  * ordered bands (e.g. first-leg delay 0-15, 15-60, ...) -> ORDINAL light-to-dark blue
  * magnitudes on a grid (heatmaps) -> SEQUENTIAL single-hue blue ramp
  * signed values (correlations) -> DIVERGING blue/red with a gray midpoint
The categorical order was checked for color-vision-deficiency separation.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ACCENT = "#eb6834"
GRAY = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# Delay causes always use these colors, in this order.
CAUSE_COLORS = {
    "Carrier": CATEGORICAL[0],
    "Weather": CATEGORICAL[1],
    "NAS (air traffic system)": CATEGORICAL[2],
    "Security": CATEGORICAL[3],
    "Late-arriving aircraft": CATEGORICAL[4],
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32a2a"],
)


def set_style():
    """Quiet axes, hairline grid, readable default sizes."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "patch.edgecolor": SURFACE,
    })


def thousands(axis):
    """Format an axis as 0, 50k, 100k, ... (pass ax.xaxis or ax.yaxis)."""
    axis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k" if v else "0")
    )


def percent(axis, decimals=0):
    """Format an axis whose values are already in percent (e.g. 21.5 -> 21.5%)."""
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%"))


def save(fig, name, fig_dir):
    """Save a figure as PNG into figures/ for the written summary and slides."""
    path = Path(fig_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
