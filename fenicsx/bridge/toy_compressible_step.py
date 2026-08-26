# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Synthetic compressible Neo-Hookean local-step driver, used as a test fixture.

Mirrors the ``step(F, dt, state) -> (P, new_state)`` contract of
:class:`fenicsx.bridge.vcann_local_step.VCANNLocalStep`, so toy and production
swap in one line. Unlike the production driver it is genuinely compressible, so
``P[2, 2]`` really depends on ``F[2, 2]`` and the F33 Newton in
:mod:`fenicsx.bridge.plane_stress` has something to solve, and it is stateless.

    W(F) = (mu/2)(tr C - 3) - mu * ln J + (lambda/2)(ln J)^2,    J = det F
    S    = mu (I - C^{-1}) + lambda ln(J) C^{-1}
    P    = F S = mu (F - F^{-T}) + lambda ln(J) F^{-T}

For block-diagonal ``F = diag(F2, F33)`` the closed-form FD-test reference is

    dP33/dF33 = mu + (mu + lambda (1 - ln J)) / F33^2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyState:
    """Minimal state for the toy driver — no history, just a time stamp."""

    t: float


class ToyCompressibleStep:
    """Compressible Neo-Hookean numpy driver with the VCANNLocalStep contract."""

    def __init__(self, mu: float = 1.0, lam: float = 1.0) -> None:
        self.mu = float(mu)
        self.lam = float(lam)

    def initial_state(self, t0: float = 0.0) -> ToyState:
        return ToyState(t=float(t0))

    def step(self, F: np.ndarray, dt: float, state: ToyState):
        """Advance one step. Returns ``(P, new_state)`` with the same shape
        contract as :meth:`VCANNLocalStep.step` — ``P`` is a (3, 3) array
        and ``new_state`` is a fresh :class:`ToyState`.
        """
        F = np.asarray(F, dtype=np.float64)
        J = float(np.linalg.det(F))
        F_inv_T = np.linalg.inv(F).T
        P = self.mu * (F - F_inv_T) + self.lam * np.log(J) * F_inv_T
        return P, ToyState(t=state.t + dt)

    def analytic_dP33_dF33(self, F: np.ndarray) -> float:
        """Closed-form ``dP[2,2]/dF[2,2]`` for a block-diagonal F. Used by
        the FD-tangent correctness test."""
        F33 = float(F[2, 2])
        J = float(np.linalg.det(F))
        return self.mu + (self.mu + self.lam * (1.0 - np.log(J))) / (F33 * F33)
