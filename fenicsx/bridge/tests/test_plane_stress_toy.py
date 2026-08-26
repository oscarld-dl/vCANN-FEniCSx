# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plane-stress condensation against a compressible toy driver.

test_plane_stress.py runs the same condensation against the production VHB4910
driver, which is structurally incompressible and converges in zero iterations,
so it exercises plumbing only. Against the compressible Neo-Hookean toy,
P[2, 2] genuinely depends on F[2, 2], so this test can assert that the Newton
finds a nontrivial F33, that the forward-FD slope dP33/dF33 matches the analytic
derivative to ~1e-6 relative, that |P[2, 2]| < tol, and that the toy driver is
correct at the reference (P = 0 at F = I).

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/tests/test_plane_stress_toy.py
"""

from pathlib import Path
import sys

# tests/ -> bridge/ -> fenicsx/ -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from fenicsx.bridge.toy_compressible_step import ToyCompressibleStep
from fenicsx.bridge.plane_stress import build_F3, condense_plane_stress


A_MATRIX = np.array([[0.1, 0.0],
                     [0.0, 0.0]], dtype=np.float64)
DT = 1.0

TOL_P33 = 1e-9              # plane-stress residual
THRESH_F33_NONTRIVIAL = 1e-3 # F33 must move at least this far from 1/det(F2)
FD_STEP = 1e-8               # matches condense_plane_stress default
RTOL_FD_VS_ANALYTIC = 1e-5   # forward-FD on float64 with h=sqrt(eps)


def _driver_sanity(driver: ToyCompressibleStep) -> None:
    """At F = I, P must be zero (reference is stress-free)."""
    state = driver.initial_state(t0=0.0)
    P, _ = driver.step(np.eye(3), DT, state)
    assert np.max(np.abs(P)) < 1e-14, (
        f"toy driver wrong at reference: max|P(I)| = {np.max(np.abs(P)):.3e}"
    )


def main() -> None:
    driver = ToyCompressibleStep()  # mu = lam = 1
    _driver_sanity(driver)

    F2 = np.eye(2, dtype=np.float64) + A_MATRIX
    F33_init = 1.0 / float(np.linalg.det(F2))

    state = driver.initial_state(t0=0.0)
    P, _, F33_conv = condense_plane_stress(F2, driver, DT, state)

    print(f"F2 =\n{F2}")
    print(f"det(F2)                 = {np.linalg.det(F2):.10f}")
    print(f"F33 init (1/det F2)     = {F33_init:.10f}")
    print(f"F33 converged           = {F33_conv:.10f}")
    print(f"|F33_conv - F33_init|   = {abs(F33_conv - F33_init):.3e}")
    print(f"converged P[2, 2]       = {P[2, 2]:.3e}")
    print(f"max |P|                 = {np.max(np.abs(P)):.3e}")

    # --- 1) plane-stress residual is below tolerance ---
    assert abs(P[2, 2]) < TOL_P33, (
        f"plane stress NOT satisfied: |P[2, 2]| = {abs(P[2, 2]):.3e} "
        f">= tol = {TOL_P33:.3e}"
    )

    # --- 2) Newton did real work: F33 moved away from the incompressibility guess ---
    delta_F33 = abs(F33_conv - F33_init)
    assert delta_F33 > THRESH_F33_NONTRIVIAL, (
        f"Newton appears trivial: |F33_conv - 1/det(F2)| = {delta_F33:.3e} "
        f"<= threshold = {THRESH_F33_NONTRIVIAL:.3e}. Either the driver is "
        f"behaving incompressibly or the condensation never iterated."
    )

    # --- 3) FD tangent matches the closed-form derivative at F33_conv ---
    F3_at = build_F3(F2, F33_conv)
    F3_plus = build_F3(F2, F33_conv + FD_STEP)
    P_at, _ = driver.step(F3_at, DT, driver.initial_state(t0=0.0))
    P_plus, _ = driver.step(F3_plus, DT, driver.initial_state(t0=0.0))
    slope_fd = (P_plus[2, 2] - P_at[2, 2]) / FD_STEP
    slope_analytic = driver.analytic_dP33_dF33(F3_at)
    rel_err = abs(slope_fd - slope_analytic) / abs(slope_analytic)

    print(f"\n--- FD tangent check at F33 = F33_converged ---")
    print(f"slope FD (h={FD_STEP:g})    = {slope_fd:.10f}")
    print(f"slope analytic            = {slope_analytic:.10f}")
    print(f"relative error            = {rel_err:.3e}")

    assert rel_err < RTOL_FD_VS_ANALYTIC, (
        f"FD tangent disagrees with analytic by rel.err = {rel_err:.3e} "
        f">= {RTOL_FD_VS_ANALYTIC:.3e}"
    )

    # --- 4) Bonus: also check FD tangent at a probe well off-equilibrium ---
    # If the FD path were silently broken (e.g. wrong step direction, sign
    # error), it might still happen to match analytic at the special point
    # F33_converged. A second probe rules that out.
    F33_probe = 1.05 * F33_init
    F3_probe = build_F3(F2, F33_probe)
    F3_probe_plus = build_F3(F2, F33_probe + FD_STEP)
    P_probe, _ = driver.step(F3_probe, DT, driver.initial_state(t0=0.0))
    P_probe_plus, _ = driver.step(F3_probe_plus, DT, driver.initial_state(t0=0.0))
    slope_fd_2 = (P_probe_plus[2, 2] - P_probe[2, 2]) / FD_STEP
    slope_analytic_2 = driver.analytic_dP33_dF33(F3_probe)
    rel_err_2 = abs(slope_fd_2 - slope_analytic_2) / abs(slope_analytic_2)

    print(f"\n--- FD tangent check at F33 = 1.05 * F33_init ---")
    print(f"slope FD                  = {slope_fd_2:.10f}")
    print(f"slope analytic            = {slope_analytic_2:.10f}")
    print(f"relative error            = {rel_err_2:.3e}")

    assert rel_err_2 < RTOL_FD_VS_ANALYTIC, (
        f"FD tangent (off-equilibrium probe) disagrees with analytic by "
        f"rel.err = {rel_err_2:.3e} >= {RTOL_FD_VS_ANALYTIC:.3e}"
    )

    print("\nplane-stress condensation toy test PASSED")


if __name__ == "__main__":
    main()
