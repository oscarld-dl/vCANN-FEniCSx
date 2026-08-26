# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plot the Newton residual histories from ``newton_convergence_study.py``.

Reads ``fenicsx/results/newton_convergence_normalized.csv`` and writes
``fenicsx/results/convergence_normalized_guides.{png,pdf}``. The FD-tangent
curves are shown against two guides: a linear line at the frozen tangent's
measured average reduction factor, and the quadratic ``r_{k+1} = r_k^2``.

Flags: ``--results-dir``, ``--show-frozen``.

Run with the bridge environment (or any env with numpy + matplotlib)::

    .venv-bridge/bin/python fenicsx/models/plot_newton_convergence.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_RESULTS_DIR = _PROJECT_ROOT / "fenicsx" / "results"

from fenicsx.models.plotstyle import C, SERIF, apply_style, style_axes, style_legend

apply_style()


def load(path):
    """CSV -> {column name: float array}, blank cells become NaN (ragged curves)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    d = {h: [] for h in hdr}
    for r in rows[1:]:
        for h, v in zip(hdr, r):
            d[h].append(float(v) if v.strip() != "" else np.nan)
    return {h: np.array(d[h]) for h in hdr}


def make_plot(results_dir=_RESULTS_DIR, show_frozen=False):
    csv_path = results_dir / "newton_convergence_normalized.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found — run newton_convergence_study.py first."
        )
    norm = load(csv_path)

    # One palette pair per load level: stroke in the "line" colour, marker face
    # in the matching "fill". Only three of the palette's node pairs carry a
    # non-black stroke (commit/tangent/value), so 25 % takes the black
    # outerline stroke and is told apart by its operatorfill markers.
    colors = {"1%": C["commitline"], "10%": C["tangentline"],
              "20%": C["valueline"], "25%": C["operatorline"]}
    fills = {"1%": C["commitline"], "10%": C["tangentline"],
             "20%": C["valueline"], "25%": C["operatorline"]}
    markers = {"1%": "s", "10%": "D", "20%": "^", "25%": "P"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    it = norm["iter"]

    for lvl in ["1%", "10%", "20%", "25%"]:
        y = norm[lvl]
        m = ~np.isnan(y)
        ax.semilogy(it[m], y[m], marker=markers[lvl], color=colors[lvl],
                    mfc=fills[lvl], mec=colors[lvl], mew=1.1,
                    lw=1.6, ms=6.5, label=f"{lvl} strain", zorder=3)

    # --- reference guides ---
    kk = np.arange(0, 7)

    # LINEAR reference: r_{k+1} = rho * r_k  (straight line on semilog),
    # anchored to the frozen tangent's measured average reduction factor.
    fz = norm["frozen"]
    fz = fz[~np.isnan(fz)]
    rho = (fz[-1] / fz[0]) ** (1.0 / (len(fz) - 1))
    lin = fz[0] * rho ** kk
    ax.semilogy(kk, lin, "--", color=C["secondary"], lw=1.3, alpha=0.7, zorder=2,
                label=f"Linear ref. (frozen tangent)")

    if show_frozen:
        ax.semilogy(np.arange(len(fz)), fz, marker="o", color=C["secondary"],
                    mfc=C["outerfill"], mec=C["secondary"], mew=1.0,
                    lw=1.2, ms=4.5, alpha=0.85, zorder=2,
                    label="frozen tangent (measured)")

    # QUADRATIC reference: r_{k+1} = r_k^2  (curved on semilog).
    r0 = 3.5e-1
    quad = [r0]
    for _ in range(8):
        quad.append(quad[-1] ** 2)
    quad = np.array(quad)
    quad = quad[quad > 1e-16]
    ax.semilogy(np.arange(len(quad)), quad, ":", color=C["secondary"], lw=1.7, alpha=0.7,
                zorder=2, label=r"Quadratic ref.")

    ax.set_xlabel("Newton iteration", color=C["outerline"])
    ax.set_ylabel(r"Normalized residual  $\|b\|/\|b_0\|$", color=C["outerline"])
    # ax.set_title("FD-consistent tangent: quadratic convergence across strain levels",
                # color=C["outerline"], fontsize=11)
    ax.set_xlim(-0.1, 4.2)
    ax.set_ylim(1e-15, 3)
    ax.grid(True, which="both", alpha=0.28, color=C["secondary"])

    leg = ax.legend(
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.32),
        frameon=True,
        fancybox=True,
        framealpha=0.45,
        edgecolor=C["secondary"],
        facecolor=C["outerfill"],
        labelcolor=C["outerline"],
    )
    leg.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.3")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    png = results_dir / "convergence_normalized_guides.png"
    pdf = results_dir / "convergence_normalized_guides.pdf"
    fig.savefig(png, dpi=500)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"frozen linear factor rho = {rho:.3f}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR,
                        help="directory holding the study CSVs "
                             f"(default {_RESULTS_DIR})")
    parser.add_argument("--show-frozen", action="store_true",
                        help="also draw the measured frozen-tangent curve")
    args = parser.parse_args()
    make_plot(results_dir=args.results_dir, show_frozen=args.show_frozen)
