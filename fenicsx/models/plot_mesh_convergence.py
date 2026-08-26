# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Energy convergence plot, from ``convergence_study.py`` output.

Reads ``fenicsx/results/mesh_convergence.csv`` and writes
``mesh_convergence_energy.{png,pdf}`` (error vs h) and
``mesh_convergence_energy_values.{png,pdf}`` (E(h) itself).

Plots ``e(h) = |E(h) - E*|`` against h on log-log, with ``E*`` the
Richardson-extrapolated limit, so a rate ``e ~ C h^p`` is a line of slope ``p``.
The h^4 guide is the optimal P2 rate on a smooth domain; the gap to the fitted
slope is the accuracy lost to the hole-boundary regularity limit.

Flags: ``--results-dir``, ``--with-stress``, ``--vs-dofs``, ``--ref-orders``,
``--drop-finest``, ``--figures {both,error,values}``.

Run with the bridge environment (or any env with numpy + matplotlib)::

    .venv-bridge/bin/python fenicsx/models/plot_mesh_convergence.py
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
from matplotlib.ticker import FormatStrFormatter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_RESULTS_DIR = _PROJECT_ROOT / "fenicsx" / "results"

from fenicsx.models.plotstyle import C, apply_style, style_axes, style_legend

apply_style()


def load(path):
    """CSV -> {column: float array} (ints stay floats; blanks become NaN)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    cols = {h: [] for h in hdr}
    for r in rows[1:]:
        for h, v in zip(hdr, r):
            cols[h].append(float(v) if v.strip() != "" else np.nan)
    return {h: np.array(cols[h]) for h in hdr}


def fit_slope(x, y):
    """Least-squares slope of log y vs log x, over the finite entries only.

    Coarse meshes can sit outside the asymptotic range; the caller decides how
    many of the finest points to pass in.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def read_reference(results_dir, d, key="energy", value_col="energy",
                   err_col="err_energy"):
    """The Richardson-extrapolated limit behind the plotted errors.

    Preferred source is ``mesh_convergence_richardson.csv`` (last triple, i.e.
    the finest). If that file is absent, recover it from the CSV itself: the
    error column is ``|q(h) - q*|``, so ``q*`` is ``q_finest`` plus or minus the
    last error — pick whichever sign reproduces the errors of ALL the meshes.
    """
    rich = results_dir / "mesh_convergence_richardson.csv"
    if rich.exists():
        with open(rich, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["quantity"] == key]
        if rows:
            return float(rows[-1]["extrapolated"])

    q, e = d[value_col], d[err_col]
    best, best_miss = float("nan"), float("inf")
    for cand in (q[-1] - e[-1], q[-1] + e[-1]):
        miss = float(np.max(np.abs(np.abs(q - cand) - e)))
        if miss < best_miss:
            best, best_miss = cand, miss
    return best


def make_value_plot(results_dir=_RESULTS_DIR, vs_dofs=False, drop_finest=False):
    """The raw QoI: E(h) against mesh size, with the extrapolated limit E*.

    Companion to the error figure — same data before the reference is
    subtracted. Log abscissa (the mesh family is geometric) but a LINEAR
    ordinate: the whole family spans ~4e-4 in E, so a log axis would show a flat
    line. The horizontal E* asymptote is what the sequence is converging to, and
    the visible gap at each h is the quantity the error figure plots.
    """
    csv_path = results_dir / "mesh_convergence.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run convergence_study.py first.")
    d = load(csv_path)
    h = d["h"]
    x = d["ndofs"] if vs_dofs else h
    E = d["energy"]
    E_star = read_reference(results_dir, d)

    keep = np.ones_like(h, dtype=bool)
    if drop_finest:
        keep[-1] = False

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Data first, reference second — that is the order the legend lists them in
    # (zorder, not creation order, decides what is drawn on top).
    ax.semilogx(x[keep], E[keep], marker="s", color=C["commitline"],
                mfc=C["commitline"], mec=C["commitline"], mew=1.1,
                lw=1.6, ms=6.5, zorder=3, label=r"Energy $E(h)$")
    ax.axhline(E_star, ls="--", color="#FF0000", lw=1.3, alpha=0.85,
               zorder=2, label=rf"Richardson limit $E^\ast$")

    ax.set_xlabel(r"Number of dofs $N$" if vs_dofs else r"Mesh size $h$",
                  color=C["outerline"])
    ax.set_ylabel(r"Stored elastic energy  $E(h)$", color=C["outerline"])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    if not vs_dofs:
        ax.invert_xaxis()   # refinement runs left -> right
    # Breathing room around the data AND the asymptote, so the approach to E*
    # is visible instead of the limit sitting on the axis edge.
    lo, hi = min(E[keep].min(), E_star), max(E[keep].max(), E_star)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(0.23260, hi + pad)
    ax.set_xlim(x[keep].max() * 1.05, 0.01)
    style_axes(ax)
    style_legend(leg = ax.legend(
        fontsize=8.8,
        loc="upper left",
        bbox_to_anchor=(0.675, 0.975),
        frameon=True,
        fancybox=True,
        framealpha=0.55,
        edgecolor=C["secondary"],
        facecolor=C["outerfill"],
        labelcolor=C["outerline"],
    ))

    fig.tight_layout()
    png = results_dir / "mesh_convergence_energy_values.png"
    pdf = results_dir / "mesh_convergence_energy_values.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"E* = {E_star:.10e}   E(h) spans "
          f"[{E[keep].min():.10e}, {E[keep].max():.10e}]")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def make_plot(results_dir=_RESULTS_DIR, with_stress=False, vs_dofs=False,
              fit_last=4, drop_finest=False, ref_orders=(4.0,)):
    csv_path = results_dir / "mesh_convergence.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found — run convergence_study.py first."
        )
    d = load(csv_path)
    h, err_E = d["h"], d["err_energy"]
    x = d["ndofs"] if vs_dofs else h

    # The three finest meshes define E* through the Richardson triple, so the
    # last point necessarily sits on the fitted line — it is a consistency
    # check, not an independent measurement. Say so in the caption; --drop-finest
    # removes it if a referee prefers only independent points.
    keep = np.ones_like(h, dtype=bool)
    if drop_finest:
        keep[-1] = False

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.loglog(x[keep], err_E[keep], marker="s", color="#FF7F0E",
              mfc="#FF7F0E", mec="#FF7F0E", mew=1.1,
              lw=1.6, ms=6.5, zorder=3,
              label=r"Energy error")

    p_E = fit_slope(x[keep][-fit_last:], err_E[keep][-fit_last:])
    xs = np.array([x[keep].min(), x[keep].max()])
    anchor = err_E[keep][-1] / (x[keep][-1] ** p_E)
    ax.loglog(xs, anchor * xs ** p_E, "--", color=C["secondary"], lw=1.3,
              alpha=0.75, zorder=2,
              label=rf"Fit $\propto h^{{{p_E:.2f}}}$" if not vs_dofs
                    else rf"Fit$\propto N^{{{p_E:.2f}}}$")

    # Optimal-rate guides. h^4 is what P2 delivers on a smooth domain: the
    # energy-norm error is O(h^2), and the energy *functional* error converges
    # at twice that rate (pure-Dirichlet loading => the potential equals E and
    # is stationary at u, so the leading term is quadratic in the error).
    # Anchored at the COARSEST plotted point so both curves start together and
    # the widening vertical gap reads directly as the accuracy given up to the
    # hole-boundary regularity. Against #dofs an h^q rate becomes N^(-q/2) in 2D.
    for q in ref_orders:
        expo = q if not vs_dofs else -q / 2.0
        anchor_r = err_E[keep][0] / (x[keep][0] ** expo)
        sym = "h" if not vs_dofs else "N"
        exp_txt = f"{q:g}" if not vs_dofs else f"-{q / 2:g}"
        ax.loglog(xs, anchor_r * xs ** expo, ":", color="#1F77B4",
                  lw=2.25, alpha=1.0, zorder=2,
                  label=rf"Optimal$\propto {sym}^{{{exp_txt}}}$")

    if with_stress:
        err_S = d["err_P_point"]
        ax.loglog(x[keep], err_S[keep], marker="^", color=C["valueline"],
                  mfc=C["valuefill"], mec=C["valueline"], mew=1.1,
                  lw=1.6, ms=6.5, zorder=3,
                  label=r"local stress  $\,|\,|P|(h)-|P|^\ast|$")
        p_S = fit_slope(x[keep][-fit_last:], err_S[keep][-fit_last:])
        anchor_S = err_S[keep][-1] / (x[keep][-1] ** p_S)
        ax.loglog(xs, anchor_S * xs ** p_S, ":", color=C["tangentline"], lw=1.4,
                  alpha=0.8, zorder=2,
                  label=rf"fit  $\propto h^{{{p_S:.2f}}}$")
    else:
        p_S = float("nan")

    ax.set_xlabel(r"number of dofs $N$" if vs_dofs else r"Mesh size $h$",
                  color=C["outerline"])
    ax.set_ylabel(r"Error in stored energy  $|E(h)-E^{\ast}|$",
                  color=C["outerline"])
    #ax.set_title("Mesh convergence of the stored elastic energy",
                 #color=C["outerline"], fontsize=11)
    if not vs_dofs:
        ax.invert_xaxis()   # refinement runs left -> right
    style_axes(ax)
    style_legend(ax.legend(fontsize=9, loc="upper right",
            bbox_to_anchor=(0.975, 0.975), frameon=True, fancybox=True,
                           framealpha=0.95, labelcolor=C["outerline"]))

    fig.tight_layout()
    png = results_dir / "mesh_convergence_energy.png"
    pdf = results_dir / "mesh_convergence_energy.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"fitted energy rate p = {p_E:.3f}  "
          f"(=> energy-norm rate {p_E / 2:.3f})"
          + (f"   local |P| rate p = {p_S:.3f}" if with_stress else ""))
    for q in ref_orders:
        expo = q if not vs_dofs else -q / 2.0
        ideal_fine = err_E[keep][0] * (x[keep][-1] / x[keep][0]) ** expo
        print(f"  vs optimal h^{q:g}: at the finest mesh the error is "
              f"{err_E[keep][-1] / ideal_fine:.1f}x the optimal-rate value "
              f"({err_E[keep][-1]:.3e} vs {ideal_fine:.3e})")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    parser.add_argument("--with-stress", action="store_true",
                        help="also plot the local |P| QoI at the monitor point")
    parser.add_argument("--vs-dofs", action="store_true",
                        help="abscissa = number of dofs instead of h")
    parser.add_argument("--fit-last", type=int, default=4,
                        help="how many of the finest points enter the slope fit")
    parser.add_argument("--drop-finest", action="store_true",
                        help="omit the finest mesh (it defines the reference)")
    parser.add_argument("--ref-orders", default="4",
                        help="comma-separated optimal-rate guides to draw "
                             "(default '4' = P2 on a smooth domain; '2,4' adds h^2)")
    parser.add_argument("--figures", choices=("both", "error", "values"),
                        default="both",
                        help="which figures to write: the error convergence, "
                             "the raw E(h) values, or both (default)")
    args = parser.parse_args()
    orders = tuple(float(t) for t in args.ref_orders.split(",") if t.strip())
    if args.figures in ("both", "error"):
        make_plot(results_dir=args.results_dir, with_stress=args.with_stress,
                  vs_dofs=args.vs_dofs, fit_last=args.fit_last,
                  drop_finest=args.drop_finest, ref_orders=orders)
    if args.figures in ("both", "values"):
        make_value_plot(results_dir=args.results_dir, vs_dofs=args.vs_dofs,
                        drop_finest=args.drop_finest)
