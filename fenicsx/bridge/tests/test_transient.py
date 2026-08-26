# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Transient verification: single-QP replay of the full ramp+hold history.

The FE solve carries the Maxwell history Q(t) inside
VCANNStressOperator.state_committed. This runs a ramp-then-hold time loop, picks
the most viscoelastically active QP, replays that QP's F(t) through
NumpyVCANN.step from a fresh state, and asserts that P(t) and Q(t) match to
1e-12: the FE machinery commits exactly the right state and does not corrupt
history across Newton iterations or increments.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/tests/test_transient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from petsc4py import PETSc

from fenicsx.bridge.plane_stress import build_F3
from fenicsx.models.embedded2d_vcann_extop import build_problem

_N_RAMP = 3
_N_HOLD = 2
_DT = 1.0
_U_TOP = 0.05   # small displacement: stays well inside Newton basin
_TOL = 1e-12


def test_transient_verification():
    """Replay one QP's F(t) standalone and confirm P(t)/Q(t) match the FE history."""
    _, u, problem, top_disp, _ = build_problem(dt=_DT)
    op = problem.operator

    F_hist, P_hist, Q_hist = [], [], []

    n_steps = _N_RAMP + _N_HOLD
    for k in range(1, n_steps + 1):
        top_disp.value = PETSc.ScalarType(_U_TOP * min(k, _N_RAMP) / _N_RAMP)
        problem.solve()
        u.x.scatter_forward()
        reason = problem.solver.getConvergedReason()
        assert reason > 0, f"increment {k} did not converge (reason={reason})"

        problem.commit_state()
        F_hist.append(op.last_F.copy())                                    # (n_qp, 2, 2)
        P_hist.append(op.last_P.copy())                                    # (n_qp, 2, 2)
        Q_hist.append(np.stack([s.Q.copy() for s in op.state_committed]))  # (n_qp, M, 3, 3)

    # Most viscoelastically active QP — strongest signal, smallest relative error.
    Qs_final = Q_hist[-1]
    rep_qp = int(np.argmax(np.max(np.abs(Qs_final), axis=(1, 2, 3))))
    print(f"\n  representative QP: {rep_qp}  (max|Q_final|="
          f"{np.max(np.abs(Qs_final[rep_qp])):.3e})")
    print(f"  {'step':>4}  {'|err_P|_inf':>14}  {'|err_Q|_inf':>14}")

    # Replay from a fresh initial state using only the recorded F(t).
    state = op.npy.initial_state(t0=0.0)
    for k in range(n_steps):
        F2 = F_hist[k][rep_qp]
        F3 = build_F3(F2, 1.0 / np.linalg.det(F2))
        P3, state = op.npy.step(F3, _DT, state)

        err_P = float(np.max(np.abs(P3[:2, :2] - P_hist[k][rep_qp])))
        err_Q = float(np.max(np.abs(state.Q   - Q_hist[k][rep_qp])))
        print(f"  {k+1:4d}  {err_P:14.2e}  {err_Q:14.2e}")

        assert err_P < _TOL, f"step {k+1}: |err_P|_inf = {err_P:.2e} >= {_TOL:.0e}"
        assert err_Q < _TOL, f"step {k+1}: |err_Q|_inf = {err_Q:.2e} >= {_TOL:.0e}"

    print(f"\n  PASS — transient history verified at QP {rep_qp} "
          f"over {n_steps} steps (tol={_TOL:.0e})")


if __name__ == "__main__":
    test_transient_verification()
