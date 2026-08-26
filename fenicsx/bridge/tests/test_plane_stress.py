# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plane-stress condensation patch test.

Asserts that |P[2, 2]| < tol at the returned F33, and that the caller's
VCANNState is not mutated by condense_plane_stress — the clone discipline whose
failure would silently corrupt a time-stepping run.

Caveat: VCANNLocalStep applies an internal S[2,2]=0 closure before P = F @ S, so
with the block-diagonal F that build_F3 produces, P[2,2] = 0 identically and the
Newton converges in 0 iterations. This exercises the plumbing but does not prove
the Newton can find a nontrivial F33; test_plane_stress_toy.py covers that.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/tests/test_plane_stress.py
"""

from pathlib import Path
import sys
import copy

# tests/ -> bridge/ -> fenicsx/ -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from fenicsx.bridge.vcann_local_step import VCANNLocalStep
from fenicsx.bridge.plane_stress import condense_plane_stress


A_MATRIX = np.array([[0.1, 0.0],
                     [0.0, 0.0]], dtype=np.float64)
DT = 1.0
TOL_P33 = 1e-9


def _assert_state_unchanged(s_before, s_after, label: str) -> None:
    assert np.array_equal(s_before.Q,      s_after.Q),      f"{label}: Q mutated"
    assert np.array_equal(s_before.S_e,    s_after.S_e),    f"{label}: S_e mutated"
    assert np.array_equal(s_before.tau,    s_after.tau),    f"{label}: tau mutated"
    assert np.array_equal(s_before.g_visc, s_after.g_visc), f"{label}: g_visc mutated"
    assert s_before.t == s_after.t,                         f"{label}: t mutated"


def main() -> None:
    driver = VCANNLocalStep()
    state = driver.initial_state(t0=0.0)

    F2 = np.eye(2, dtype=np.float64) + A_MATRIX
    F33_incompressible = 1.0 / float(np.linalg.det(F2))

    # Snapshot for the state-purity check.
    state_snapshot = copy.deepcopy(state)

    P, new_state, F33_conv = condense_plane_stress(F2, driver, DT, state)

    print(f"F2 =\n{F2}")
    print(f"det(F2)                 = {np.linalg.det(F2):.10f}")
    print(f"F33 init (1/det F2)     = {F33_incompressible:.10f}")
    print(f"F33 converged           = {F33_conv:.10f}")
    print(f"|F33_conv - F33_init|   = {abs(F33_conv - F33_incompressible):.3e}")
    print(f"converged P[2, 2]       = {P[2, 2]:.3e}")
    print(f"max |P|                 = {np.max(np.abs(P)):.3e}")

    # --- assertion 1: plane-stress condition ---
    assert abs(P[2, 2]) < TOL_P33, (
        f"plane stress NOT satisfied: |P[2, 2]| = {abs(P[2, 2]):.3e} "
        f">= tol = {TOL_P33:.3e}"
    )

    # --- assertion 2: input state is unmutated ---
    _assert_state_unchanged(state_snapshot, state, "input-state purity")

    # --- assertion 3: returned new_state is real (not a phantom copy of input) ---
    # For F2 = I + A != I, the elastic 2nd-PK at this step is nonzero, so
    # new_state.S_e must differ from state.S_e (which is zero from initial_state).
    # If they were equal it would indicate the final eval never ran or got
    # clobbered.
    assert not np.array_equal(new_state.S_e, state.S_e), (
        "new_state.S_e equals input state.S_e — final converged eval "
        "appears not to have produced a real updated state"
    )

    print("plane-stress condensation test PASSED")


if __name__ == "__main__":
    main()
