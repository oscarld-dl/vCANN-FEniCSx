# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plot the maximum first Piola-Kirchhoff stress over the load history.

The first 15 s are the displacement ramp; the rest is the hold/relaxation
phase. Writes a publication-quality PDF and a 200-dpi PNG.

    python plot_maxp.py [--output-dir ./figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Paper palette. Edit these hexadecimal values to recolor the complete plot.
C = {
    "outerline": "#2F3542",
    "secondary": "#7F8C8D",
    "outerfill": "#F7F8FA",
    "ramp": "#C0392B",
    "hold": "#1F3A93",
    "grid": "#B8BEC7",
}


P_KPA = np.array(
    [
        23.925, 38.822, 48.817, 55.886, 61.137, 65.229, 68.569,
        71.418, 73.948, 76.275, 78.478, 80.610, 82.706, 84.787,
        86.867, 84.184, 81.952, 80.060, 78.449, 77.070, 75.882,
        74.854, 73.956, 73.167, 72.467, 71.843, 71.280, 70.769,
        70.301, 69.869, 69.467, 69.091, 68.736, 68.399, 68.078,
        67.770, 67.474, 67.188, 66.911, 66.642, 66.380, 66.123,
        65.873, 65.628, 65.387, 65.151, 64.920, 64.692, 64.468,
        64.247, 64.030, 63.816, 63.605, 63.397, 63.192, 62.991,
        62.792, 62.595, 62.402, 62.211, 62.022, 61.837, 61.653,
        61.473, 61.294, 61.119, 60.945, 60.774, 60.605, 60.438,
        60.274, 60.112, 59.952, 59.794, 59.638, 59.484, 59.333,
        59.183, 59.036, 58.890, 58.747, 58.605, 58.465, 58.327,
        58.191, 58.057, 57.925, 57.794, 57.665, 57.538, 57.412,
        57.289, 57.166, 57.046, 56.927, 56.810, 56.694, 56.580,
        56.467, 56.356, 56.247, 56.139, 56.032, 55.927, 55.823,
        55.721, 55.620, 55.520, 55.422, 55.325, 55.229, 55.134,
        55.041, 54.949, 54.859,
    ],
    dtype=float,
)


def apply_style() -> None:
    """Apply the same compact paper style as the convergence figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    """Style spines, ticks, and grid consistently."""
    for spine in ax.spines.values():
        spine.set_color(C["outerline"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=C["outerline"], width=0.8, direction="out")
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        color=C["grid"],
        alpha=0.70,
    )


def make_plot(output_dir: Path) -> tuple[Path, Path]:
    """Create and save the load-ramp/hold stress-history figure."""
    time_s = np.arange(1.0, P_KPA.size + 1.0)
    ramp = time_s <= 15.0

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Continuous history beneath the phase-specific markers.
    ax.plot(
        time_s,
        P_KPA,
        color=C["hold"],
        linewidth=1.6,
        zorder=2,
    )
    ax.plot(
        time_s[ramp],
        P_KPA[ramp],
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor=C["ramp"],
        markeredgecolor=C["ramp"],
        markeredgewidth=0.9,
        label="Ramp",
        zorder=3,
    )
    ax.plot(
        time_s[~ramp],
        P_KPA[~ramp],
        linestyle="none",
        marker="s",
        markersize=3.8,
        markerfacecolor=C["hold"],
        markeredgecolor=C["hold"],
        markeredgewidth=0.9,
        label="Hold",
        zorder=3,
    )

    ax.axvline(
        15.0,
        linestyle="--",
        linewidth=0.9,
        color=C["secondary"],
        alpha=0.9,
        zorder=1,
    )

    ax.set_xlabel(r"Time $t$ [$s$]", color=C["outerline"])
    ax.set_ylabel(r"Maximum stress $\max\,|P|$ [$kPa$]", color=C["outerline"])
    ax.set_xlim(0.0, 116.0)
    ax.set_ylim(20.0, 90.0)
    style_axes(ax)

    # Manual legend placement in axes coordinates: x rightward, y upward.
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.82, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.55,
        edgecolor=C["secondary"],
        facecolor=C["outerfill"],
        labelcolor=C["outerline"],
    )
    legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.3")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "maxP_vs_time.png"
    pdf_path = output_dir / "maxP_vs_time.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    peak_index = int(np.argmax(P_KPA))
    print(
        f"peak = {P_KPA[peak_index]:.3f} kPa "
        f"at t = {time_s[peak_index]:.1f} s"
    )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="directory for maxP_vs_time.{png,pdf}",
    )
    args = parser.parse_args()

    apply_style()
    make_plot(args.output_dir)


if __name__ == "__main__":
    main()
