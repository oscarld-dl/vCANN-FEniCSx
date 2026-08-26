# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Shared figure style for the paper and beamer plots: serif fonts + TikZ palette.

Call ``apply_style()`` before creating any figure, so the deck and the paper
share one font stack and one colour list, matching the \\colorlet names in the
LaTeX source.
"""

from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm


# Times-first serif, degrading through the metric-compatible clones. Naming a
# family that does not exist makes matplotlib fall back to DejaVu *Sans* (with a
# wall of findfont warnings) — i.e. silently sans-serif. Pick one that resolves.
_SERIF_CANDIDATES = [
    "Times New Roman",   # the real thing (Windows / msttcorefonts)
    "Liberation Serif",  # metric-compatible clone
    "Nimbus Roman",      # URW Times clone, present on most Linux boxes
    "STIXGeneral",       # Times-like, pairs exactly with the stix mathtext set
    "DejaVu Serif",      # always ships with matplotlib
]
_available = {f.name for f in fm.fontManager.ttflist}
SERIF = next((f for f in _SERIF_CANDIDATES if f in _available), "serif")


# --- beamer/TikZ palette -------------------------------------------------
# The \colorlet names from the slide deck, resolved to hex so the figures and
# the TikZ diagrams share one palette. xcolor semantics: "X!p" is p% X on white,
# "X!p!Y" is p% X with (100-p)% Y; base gray=0.5, orange=(1,.5,0).
#   outerfill  gray!8          (.96,.96,.96)
#   newtonfill blue!4          (.96,.96,1)
#   operatorfill blue!12       (.88,.88,1)
#   valuefill  yellow!20       (1,1,.8)      valueline   yellow!45!black (.45,.45,0)
#   tangentfill orange!20      (1,.9,.8)     tangentline blue!45!black   (0,0,.45)
#   commitfill green!20        (.8,1,.8)     commitline  green!40!black  (0,.4,0)
#   secondary  gray!40!black   (.2,.2,.2)    outer/newton/operatorline   black
C = {
    "outerfill":    "#F5F5F5",
    "outerline":    "#000000",
    "newtonfill":   "#F5F5FF",
    "newtonline":   "#000000",
    "operatorfill": "#E0E0FF",
    "operatorline": "#000000",
    "valuefill":    "#FFFFCC",
    "valueline":    "#737300",
    "tangentfill":  "#FFE6CC",
    "tangentline":  "#000073",
    "commitfill":   "#CCFFCC",
    "commitline":   "#006600",
    "secondary":    "#333333",
}


def apply_style():
    """Set the serif family + stix mathtext on the global rcParams."""
    matplotlib.rcParams["font.family"] = SERIF
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    return SERIF


def style_axes(ax, *, grid=True):
    """Palette-consistent spines, ticks, and grid for one axes."""
    ax.tick_params(axis="both", labelcolor=C["outerline"], color=C["secondary"])
    for spine in ax.spines.values():
        spine.set_color(C["secondary"])
    if grid:
        ax.grid(True, which="both", alpha=0.28, color=C["secondary"])


def style_legend(leg):
    """Palette-consistent legend frame (outerfill box, secondary edge)."""
    frame = leg.get_frame()
    frame.set_boxstyle("round,pad=0.4,rounding_size=0.5")
    frame.set_linewidth(0.8)
    frame.set_facecolor(C["outerfill"])
    frame.set_edgecolor(C["secondary"])
    return leg
