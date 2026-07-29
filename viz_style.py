"""Shared figure style for this repo.

One place for every styling decision: rcParams, the palette, layout geometry and
the save path. Notebooks should not set colors, grids or fonts themselves — they
call `new_figure()`, draw, then `finalize()` and `save()`.

Typical use:

    import viz_style as vz

    fig, ax = vz.new_figure()
    ax.scatter(X, y, **vz.SCATTER)
    vz.finalize(
        fig, ax,
        title="What the chart shows",
        subtitle="n = 100",
        xlabel="Feature x (unitless)",
        ylabel="Target y (unitless)",
        source="Synthetic sample generated in-notebook",
    )
    vz.save(fig, "01_my_figure")
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Assigned in this fixed order, never cycled. One series is one color.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL

# Sequential = one blue hue, light -> dark. For discrete ordered categories,
# start no lighter than index 1 (#86b6ef).
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]

# Status colors are reserved for status. Never use as a series color.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

DIVERGING_MID = "#f0efec"  # neutral gray, lands exactly on zero

sequential_cmap = LinearSegmentedColormap.from_list("repo_seq", SEQUENTIAL)
diverging_cmap = LinearSegmentedColormap.from_list(
    "repo_div", [SEQUENTIAL[-1], SEQUENTIAL[3], DIVERGING_MID, "#e07a78", "#d03b3b"]
)


def ordered_blues(n: int) -> list[str]:
    """`n` blues for genuinely ordered categories, light -> dark.

    Starts no lighter than #86b6ef so the pale end stays legible on #fcfcfb.
    """
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL[3]]
    usable = SEQUENTIAL[1:]
    step = (len(usable) - 1) / (n - 1)
    return [usable[round(i * step)] for i in range(n)]


def diverging_norm(vmin: float, vmax: float) -> TwoSlopeNorm:
    """Norm that pins the neutral gray midpoint to exactly zero."""
    return TwoSlopeNorm(vmin=min(vmin, -1e-9), vcenter=0.0, vmax=max(vmax, 1e-9))


# --------------------------------------------------------------------------
# Type scale
#
# Neither Liberation Sans nor DejaVu Sans ships a semibold face, so hierarchy
# comes from size and color only — never weight.
# --------------------------------------------------------------------------

SIZE_TITLE = 13.5
SIZE_SUBTITLE = 10.0
SIZE_LABEL = 10.5
SIZE_TICK = 9.5
SIZE_SOURCE = 8.5

FONT_STACK = ["Liberation Sans", "DejaVu Sans", "sans-serif"]

# --------------------------------------------------------------------------
# Layout
#
# Every figure is 8x5in at 200dpi (1600x1000) and shares one plot rectangle, so
# the set reads as one system: titles start at the same x, baselines align.
# --------------------------------------------------------------------------

FIGSIZE = (8, 5)
DPI = 200

# Left margin has to clear the widest y tick label plus the y axis label.
_AXES_RECT = (0.105, 0.150, 0.865, 0.635)  # left, bottom, width, height
_X_TITLE = _AXES_RECT[0]
_Y_TITLE = 0.945
_Y_SUBTITLE = 0.888
_Y_SOURCE = 0.032

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Common mark specs, so marks match across notebooks.
SCATTER = {"s": 26, "alpha": 0.75, "linewidths": 0, "color": BLUE}
LINE = {"linewidth": 2.0, "solid_capstyle": "round"}
BAR = {"linewidth": 0, "color": BLUE}  # bars carry no stroke
HIST = {"linewidth": 0, "color": BLUE}


def apply_style() -> None:
    """Install the rcParams. Called on import; safe to call again."""
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.bbox": None,  # fixed geometry; never bbox_inches="tight"
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": SIZE_TICK,
            "text.color": INK,
            "axes.titlesize": SIZE_TITLE,
            "axes.labelsize": SIZE_LABEL,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.grid.axis": "y",  # y only
            "axes.axisbelow": True,  # behind the data
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid hairlines, never dashed
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 4,
            "legend.frameon": False,
            "legend.fontsize": SIZE_LABEL,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.6,
            "lines.linewidth": 2.0,
        }
    )


apply_style()


def new_figure(figsize=FIGSIZE, left: float | None = None):
    """A figure and one axes on the shared plot rectangle.

    `left` widens the left margin for charts whose y tick labels are words
    rather than numbers (a confusion matrix, a category axis). The right edge
    stays put, so titles and the plot area still end where every other figure's
    do and the set stays aligned.
    """
    fig = plt.figure(figsize=figsize)
    rect = _AXES_RECT
    if left is not None:
        right = rect[0] + rect[2]
        rect = (left, rect[1], right - left, rect[3])
    ax = fig.add_axes(rect)
    return fig, ax


def finalize(
    fig,
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str | None = None,
    source: str | None = None,
    legend: bool = False,
    legend_ncol: int | None = None,
):
    """Apply title block, labels, legend and source line.

    `legend` should be True only for 2+ series; a single series is named by the
    title instead, so it gets no legend box.
    """
    fig.text(
        _X_TITLE, _Y_TITLE, title,
        fontsize=SIZE_TITLE, color=INK, ha="left", va="center",
    )
    if subtitle:
        fig.text(
            _X_TITLE, _Y_SUBTITLE, subtitle,
            fontsize=SIZE_SUBTITLE, color=INK_SECONDARY, ha="left", va="center",
        )
    if source:
        fig.text(
            _X_TITLE, _Y_SOURCE, source,
            fontsize=SIZE_SOURCE, color=MUTED, ha="left", va="center",
        )

    ax.set_xlabel(xlabel, fontsize=SIZE_LABEL, color=INK_SECONDARY, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=SIZE_LABEL, color=INK_SECONDARY, labelpad=7)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="lower left", bbox_to_anchor=(0.0, 1.015),
            ncol=legend_ncol or len(handles),
            frameon=False, fontsize=SIZE_LABEL, labelcolor=INK_SECONDARY,
            borderpad=0.0, handlelength=1.6,
        )
    return fig


def log_yaxis(ax, formatter: FuncFormatter | None = None) -> None:
    """Log y-scale with the minor tick labels suppressed.

    Use wherever a linear scale hides the data — losses spanning orders of
    magnitude, heavily skewed amounts, extreme class imbalance.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(formatter or significant())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def nice_ticks(ax, axis: str = "both", nbins: int = 6) -> None:
    """Force ticks onto round steps (1/2/5/10), never 0.25-style fractions.

    Without this a 0.25 step formatted to one decimal reads "0.0, 0.2, 0.5,
    0.8, 1.0", which looks like an irregular axis.
    """
    locator = lambda: mpl.ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(locator())
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(locator())


def annotate_point(ax, x, y, text, *, color=INK, dx=8, dy=8, ha="left"):
    """Direct-label a single mark — the max or the endpoint, not every point."""
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=SIZE_TICK, color=color, ha=ha, va="bottom",
    )


def matrix_heatmap(ax, values, *, xticklabels, yticklabels, cell_text=None):
    """A small annotated matrix on the sequential blue ramp.

    Used for confusion matrices. Counts are magnitudes, so the ramp is one hue —
    never a diverging or rainbow map. Cell labels flip to the surface color on
    the dark end of the ramp so they stay legible. Grid and spines are removed:
    the cells are the chart, so a gridline behind them reads as noise.
    """
    ax.imshow(values, cmap=sequential_cmap, aspect="auto",
              vmin=0, vmax=float(values.max()))

    ax.set_xticks(range(len(xticklabels)), xticklabels, color=INK_SECONDARY)
    ax.set_yticks(range(len(yticklabels)), yticklabels, color=INK_SECONDARY)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = float(values.max()) * 0.6
    for (row, col), value in _enumerate2d(values):
        label = cell_text[row][col] if cell_text is not None else f"{value:,.0f}"
        ax.text(
            col, row, label,
            ha="center", va="center", fontsize=SIZE_LABEL,
            color=SURFACE if value > threshold else INK,
        )
    return ax


def _enumerate2d(values):
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            yield (row, col), values[row][col]


def save(fig, name: str) -> str:
    """Write the figure to figures/<name>.png at 1600x1000.

    Saves the same Figure object the notebook renders — never re-plots.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    return path


# --------------------------------------------------------------------------
# Tick formatters
# --------------------------------------------------------------------------


def thousands(decimals: int = 0) -> FuncFormatter:
    """1,234 / 1,234.5"""
    return FuncFormatter(lambda v, _pos: f"{v:,.{decimals}f}")


def compact_currency(symbol: str = "$") -> FuncFormatter:
    """$12.4K, $1.2B — compact, sign preserved."""

    def _fmt(v, _pos):
        sign = "-" if v < 0 else ""
        v = abs(v)
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if v >= cutoff:
                return f"{sign}{symbol}{v / cutoff:.1f}{suffix}"
        return f"{sign}{symbol}{v:,.0f}"

    return FuncFormatter(_fmt)


def percent(decimals: int = 1, already_scaled: bool = True) -> FuncFormatter:
    """13.2% — set already_scaled=False for 0-1 fractions."""
    scale = 1.0 if already_scaled else 100.0
    return FuncFormatter(lambda v, _pos: f"{v * scale:.{decimals}f}%")


def decimals(n: int = 1) -> FuncFormatter:
    """Plain fixed-precision, with thousands separators."""
    return FuncFormatter(lambda v, _pos: f"{v:,.{n}f}")


def significant(n: int = 3) -> FuncFormatter:
    """Drop trailing zeros — 100 / 10 / 1 / 0.92 rather than 100.00 / 1.00."""
    return FuncFormatter(lambda v, _pos: f"{v:,.{n}g}")


def signed(value: float, n: int = 3) -> str:
    """Format a near-zero mean without the '-0.000' artifact."""
    if abs(value) < 0.5 * 10 ** (-n):
        value = 0.0
    return f"{value:,.{n}f}"
