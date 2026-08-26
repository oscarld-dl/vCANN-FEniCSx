# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plane-stress condensation for the thin-film vCANN coupling.

A scalar Newton on the out-of-plane stretch F33 driving P[2, 2] -> 0 at fixed
in-plane F2. dP33/dF33 is a forward finite difference, the driver exposing no
analytic tangent. VCANNLocalStep.step is stateful, so every trial and FD probe
runs on a deepcopy and the caller's state is unchanged on return.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Imported only for static type checkers — runtime import would pull in
    # TensorFlow + the saved checkpoint, which the toy test does not need.
    # Annotations are strings under `from __future__ import annotations`.
    from .vcann_local_step import VCANNLocalStep, VCANNState


def build_F3(F2: np.ndarray, F33: float) -> np.ndarray:
    """Embed a 2x2 in-plane F into a 3x3 with an arbitrary thickness stretch.

    Numpy sibling of kinematics.embed_incompressible_2d_in_3d, which is UFL
    and chooses F33 = 1/det(F2). Here F33 is a free input — the plane-stress
    Newton owns the choice of F33.
    """
    F3 = np.zeros((3, 3), dtype=np.float64)
    F3[:2, :2] = F2
    F3[2, 2] = float(F33)
    return F3


def condense_plane_stress(
    F2: np.ndarray,
    driver: VCANNLocalStep,
    dt: float,
    state: VCANNState,
    *,
    F33_init: float | None = None,
    tol: float = 1e-10,
    fd_step: float = 1e-8,
    max_iter: int = 20,
):
    """Newton-solve F33 so that P[2, 2] = 0 at fixed in-plane F2.

    Parameters
    ----------
    F2 : (2, 2) ndarray
        In-plane deformation gradient. Held fixed across the Newton.
    driver : VCANNLocalStep
        Stateful constitutive driver. Used only via driver.step.
    dt : float
        Time increment passed through to driver.step.
    state : VCANNState
        Committed state at the start of the step. NOT mutated; all
        evaluations run on deepcopies.
    F33_init : float, optional
        Initial guess. Defaults to 1/det(F2) (incompressibility), which is
        a good warm start for near-incompressible materials. Time-stepping
        callers will typically want to pass the previous step's converged
        F33 instead.
    tol : float
        Convergence tolerance on |P[2, 2]|.
    fd_step : float
        Forward-FD step in F33 for the tangent. Defaults assume float64
        (vcann_local_step.py forces float64); loosen for a float32 driver.
    max_iter : int
        Maximum Newton iterations before raising.

    Returns
    -------
    P : (3, 3) ndarray
        First Piola-Kirchhoff stress at the converged F33.
    new_state : VCANNState
        Committed state at the end of this time step.
    F33 : float
        Converged thickness stretch.
    """
    if F33_init is None:
        F33_init = 1.0 / float(np.linalg.det(F2))
    F33 = float(F33_init)

    r = float("nan")
    converged = False
    for _ in range(max_iter):
        P_trial, _ = driver.step(build_F3(F2, F33), dt, copy.deepcopy(state))
        r = float(P_trial[2, 2])
        if abs(r) < tol:
            converged = True
            break
        P_plus, _ = driver.step(
            build_F3(F2, F33 + fd_step), dt, copy.deepcopy(state)
        )
        dr = (float(P_plus[2, 2]) - r) / fd_step
        if dr == 0.0:
            raise RuntimeError(
                "plane-stress Newton: dP33/dF33 vanished — cannot update"
            )
        F33 -= r / dr

    if not converged:
        raise RuntimeError(
            f"plane-stress Newton did not converge in {max_iter} iterations; "
            f"|P33| = {abs(r):.3e}"
        )

    # Only the final converged evaluation produces the returned new_state.
    P, new_state = driver.step(build_F3(F2, F33), dt, copy.deepcopy(state))
    return P, new_state, F33
