# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Newton convergence study: frozen tangent vs. consistent (FD) tangent.

Compares ``embedded2d_vcann.py`` (vCANN residual, Neo-Hookean Jacobian, linear
at best) against ``embedded2d_vcann_extop.py`` at 1/10/20/25 % top displacement,
one load step each from the BC-satisfying seed. Same 4x4 mesh throughout, so the
curves are directly comparable.

Outputs (``fenicsx/results/``):
  ``newton_convergence.csv``  ||F||_2 at each accepted SNES iterate.
  ``newton_probes.csv``       ||F||_2 at every residual evaluation.
  ``newton_linesearch.csv``   residual evaluations per iteration.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/models/newton_convergence_study.py
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from petsc4py import PETSc
from dolfinx import fem

import fenicsx.models.embedded2d_vcann as frozen_mod
import fenicsx.models.embedded2d_vcann_extop as extop_mod


_RESULTS_DIR = _PROJECT_ROOT / "fenicsx" / "results"

N_CELLS = 4          # same mesh for every curve
DT = 1.0             # one increment of physical time
FROZEN_U_TOP = 0.01  # frozen tangent is only viable at the small load
EXTOP_LOADS = [0.01, 0.10, 0.20, 0.25]   # 1 %, 10 %, 20 %, 25 % of the height


class History:
    """Records ||F|| at every residual evaluation and at every accepted iterate."""

    def __init__(self, label):
        self.label = label
        self.iterates = []    # ||F|| at accepted SNES iterates (SNES monitor)
        self.probes = []      # ||F|| at every residual evaluation (line search too)
        self.evals_at_iter = []   # cumulative eval count when each iterate landed
        self.reason = None
        self.wall = float("nan")

    def wrap(self, problem, inner_fn):
        """Re-register the SNES function callback (``inner_fn``) behind a recorder
        so every probe is captured, and attach a monitor for accepted iterates."""
        snes = problem.solver

        def recording_fn(snes_, x, b):
            inner_fn(snes_, x, b)
            self.probes.append(b.norm())

        snes.setFunction(recording_fn, problem.b)
        snes.setMonitor(lambda s, its, norm: (self.iterates.append(float(norm)),
                                              self.evals_at_iter.append(len(self.probes))))

    def line_search_evals(self):
        """Residual evaluations consumed between successive accepted iterates."""
        return [b - a for a, b in zip(self.evals_at_iter, self.evals_at_iter[1:])]

    def orders(self):
        """Observed order p_k from three successive residual norms:
        p_k = ln(r_{k+1}/r_k) / ln(r_k/r_{k-1}).  ~1 linear, ~2 quadratic."""
        r = self.iterates
        out = []
        for k in range(1, len(r) - 1):
            if min(r[k - 1], r[k], r[k + 1]) <= 0.0:
                out.append(float("nan"))
                continue
            den = math.log(r[k] / r[k - 1])
            out.append(math.log(r[k + 1] / r[k]) / den if den != 0.0 else float("nan"))
        return out


def run_frozen(u_top=FROZEN_U_TOP, n_cells=N_CELLS):
    """embedded2d_vcann.py: vCANN residual + frozen Neo-Hookean tangent."""
    frozen_mod.N_CELLS = n_cells
    frozen_mod.U_TOP = u_top
    domain, u, P_qf, problem, top_disp = frozen_mod.build_problem()
    top_disp.value = PETSc.ScalarType(u_top)

    hist = History("frozen")
    hist.wrap(problem, problem._vcann_residual)
    print(f"[frozen] u_top={u_top}, N_QP={problem.N_QP} ...", flush=True)
    t0 = time.perf_counter()
    problem.solve()
    hist.wall = time.perf_counter() - t0
    hist.reason = problem.solver.getConvergedReason()
    print(f"[frozen] its={problem.solver.getIterationNumber()} "
          f"reason={hist.reason} wall={hist.wall:.1f}s "
          f"evals={len(hist.probes)}", flush=True)
    return hist


def run_extop(u_top, n_cells=N_CELLS, dt=DT):
    """embedded2d_vcann_extop.py: consistent FD tangent + Dirichlet lifting seed.

    One load step from the smooth, BC-satisfying seed u_y = y * u_top (the same
    seed solve_load_steps uses for its first increment), so the lifting term is
    exactly zero and the residual is a pure function of u.
    """
    extop_mod.N_CELLS = n_cells
    extop_mod.U_TOP = u_top
    domain, u, problem, top_disp, Q_P = extop_mod.build_problem(dt=dt)
    top_disp.value = PETSc.ScalarType(u_top)

    u.interpolate(lambda xx: np.vstack((np.zeros_like(xx[1]), u_top * xx[1])))
    fem.set_bc(u.x.petsc_vec, problem._bcs_list)
    u.x.scatter_forward()

    hist = History(f"{u_top * 100:g}%")
    hist.wrap(problem, problem._residual)
    print(f"[extop {hist.label}] N_QP={problem.operator.n_qp} ...", flush=True)
    t0 = time.perf_counter()
    problem.solve()
    hist.wall = time.perf_counter() - t0
    hist.reason = problem.solver.getConvergedReason()
    print(f"[extop {hist.label}] its={problem.solver.getIterationNumber()} "
          f"reason={hist.reason} wall={hist.wall:.1f}s "
          f"evals={len(hist.probes)}", flush=True)
    return hist


def _write_columns(path, index_name, columns, series):
    """CSV with `index_name` in column 0 and one ragged series per curve."""
    n = max((len(s) for s in series), default=0)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([index_name] + columns)
        for i in range(n):
            row = [i]
            for s in series:
                row.append(f"{s[i]:.6e}" if i < len(s) else "")
            w.writerow(row)
    print(f"  wrote {path}")


def _print_table(title, index_name, columns, series, fmt="{:.6e}"):
    n = max((len(s) for s in series), default=0)
    width = max(13, max(len(c) for c in columns) + 2)
    print(f"\n  {title}")
    print("  " + f"{index_name:>5}" + "".join(f"{c:>{width}}" for c in columns))
    for i in range(n):
        cells = "".join(
            f"{(fmt.format(s[i]) if i < len(s) else '—'):>{width}}" for s in series
        )
        print("  " + f"{i:>5}" + cells)


def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hists = [run_frozen()]
    for load in EXTOP_LOADS:
        hists.append(run_extop(load))

    labels = [h.label for h in hists]

    _print_table("||F||_2 at accepted Newton iterates", "iter", labels,
                 [h.iterates for h in hists])
    _print_table("||F||_2 at every residual evaluation (line-search probes included)",
                 "eval", labels, [h.probes for h in hists])
    _print_table("residual evaluations per accepted iterate (1 = no probing)",
                 "iter", labels, [h.line_search_evals() for h in hists], fmt="{:d}")
    _print_table("observed order p_k  (~1 linear, ~2 quadratic)", "k", labels,
                 [h.orders() for h in hists], fmt="{:.3f}")

    print("\n  summary")
    print(f"  {'curve':>8} {'iters':>6} {'evals':>6} {'reason':>7} {'wall[s]':>9} "
          f"{'r0':>12} {'r_final':>12}")
    for h in hists:
        r0 = h.iterates[0] if h.iterates else float("nan")
        rf = h.iterates[-1] if h.iterates else float("nan")
        print(f"  {h.label:>8} {len(h.iterates) - 1:6d} {len(h.probes):6d} "
              f"{h.reason:7d} {h.wall:9.2f} {r0:12.4e} {rf:12.4e}")

    _write_columns(_RESULTS_DIR / "newton_convergence.csv", "iter", labels,
                   [h.iterates for h in hists])
    _write_columns(_RESULTS_DIR / "newton_probes.csv", "eval", labels,
                   [h.probes for h in hists])
    _write_columns(_RESULTS_DIR / "newton_linesearch.csv", "iter", labels,
                   [[float(v) for v in h.line_search_evals()] for h in hists])


if __name__ == "__main__":
    main()
