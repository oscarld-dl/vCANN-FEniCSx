# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""The VHB4910 vCANN local step as a ``dolfinx_external_operator``.

Evaluated as a single external operator rather than split between UFL and an
operator: the elastic and viscoelastic responses are joined by ``S_dev`` and by
the plane-stress pressure closure, so the local step does not decompose.
``eval_P`` reads the committed state and writes the trial state; ``commit``
promotes it after a converged solve.

Contract:

* operand: in-plane ``F2`` (2x2), i.e. ``I + grad(u)`` at the quadrature points.
* embedding: incompressible, ``F33 = 1/det(F2)``; the closure inside ``step``
  enforces ``S33 = 0``.
* output ``(0,)``: in-plane ``P2 = P3[:2, :2]``, flattened over QPs.
* output ``(1,)``: tangent ``A[a, b, i, j] = dP2_ab / dF2_ij``, flattened over
  QPs in row-major (output indices then operand indices).
"""
from __future__ import annotations

import copy

import numpy as np

def _embed_batch(F2, F33):
    """Embed n in-plane (2×2) deformation gradients into (n, 3×3) incompressible form."""
    n = F2.shape[0]
    F3 = np.zeros((n, 3, 3))
    F3[:, :2, :2] = F2
    F3[:, 2, 2] = F33
    return F3


class VCANNStressOperator:
    """Stateful FD-tangent callback bundle for a :class:`FEMExternalOperator`.

    Bind :meth:`external_function` as the operator's ``external_function`` and
    call :meth:`commit` after each converged load/time step.

    Parameters
    ----------
    npy : NumpyVCANN
        TF-free local-step model (weights already extracted from the driver).
    dt : float
        Time increment passed to ``step`` each evaluation.
    n_qp : int
        Number of local quadrature points (length of the per-QP state lists).
    fd_step : float
        Relative forward-difference step for the tangent. Default 1e-8, i.e.
        ~sqrt(eps), the forward-difference optimum. Central differencing would
        need ~1e-6 (~eps^(1/3)) instead; at 1e-8 the two central evaluations
        agree to ~8 digits and differencing them is pure cancellation noise.
    """

    def __init__(self, npy, dt: float, n_qp: int, *, fd_step: float = 1e-8):
        self.npy = npy
        self.dt = float(dt)
        self.n_qp = int(n_qp)
        self.fd_step = float(fd_step)
        self.state_committed = [npy.initial_state(t0=0.0) for _ in range(n_qp)]
        self.state_trial = [copy.deepcopy(s) for s in self.state_committed]
        self.last_P = None  # (n_qp, 2, 2) from the most recent eval_P, for diagnostics
        self.last_F = None  # (n_qp, 2, 2) from the most recent eval_P, for diagnostics

    # ---- (0,): stress ----
    def eval_P(self, F_flat):
        # Evaluate from the committed state; store the unperturbed result as
        # the trial state.
        F2 = np.asarray(F_flat, dtype=np.float64).reshape(-1, 2, 2)
        assert F2.shape[0] == self.n_qp, (F2.shape[0], self.n_qp)
        F3 = _embed_batch(F2, 1.0 / np.linalg.det(F2))
        P3, self.state_trial = self.npy.batch_step(F3, self.dt, self.state_committed)
        P2 = P3[:, :2, :2].copy()
        self.last_P = P2
        self.last_F = F2.copy()
        return P2.reshape(-1)

    # ---- (1,): consistent (FD) tangent dP/dF ----
    def eval_dPdF(self, F_flat):
        F2 = np.asarray(F_flat, dtype=np.float64).reshape(-1, 2, 2)
        assert F2.shape[0] == self.n_qp, (F2.shape[0], self.n_qp)
        # Forward difference: one batch_step per component plus a shared base,
        # each already vectorised over all QPs. The probes must not modify the
        # trial or committed constitutive state, so every call reads
        # state_committed and discards the state it returns.
        F3 = _embed_batch(F2, 1.0 / np.linalg.det(F2))
        base3, _ = self.npy.batch_step(F3, self.dt, self.state_committed)
        base = base3[:, :2, :2]
        A = np.empty((self.n_qp, 2, 2, 2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                h = self.fd_step * np.maximum(1.0, np.abs(F2[:, i, j]))
                F2p = F2.copy()
                F2p[:, i, j] += h
                Pp3, _ = self.npy.batch_step(
                    _embed_batch(F2p, 1.0 / np.linalg.det(F2p)), self.dt, self.state_committed
                )
                A[:, :, :, i, j] = (Pp3[:, :2, :2] - base) / h[:, None, None]
        return A.reshape(-1)

    # ---- dispatch by derivative multi-index ----
    def external_function(self, derivatives):
        if derivatives == (0,):
            return self.eval_P
        if derivatives == (1,):
            return self.eval_dPdF
        raise NotImplementedError(
            f"VCANNStressOperator: derivative {derivatives} not implemented "
            "(only (0,) stress and (1,) tangent)."
        )

    def commit(self):
        """Promote the converged trial state to the committed state."""
        self.state_committed = [copy.deepcopy(s) for s in self.state_trial]

    def committed_Q_stats(self):
        """Magnitude of the committed history variables ``Q`` across all QPs.

        Returns ``(max_abs, per_branch_max)``, where
        ``per_branch_max[i] = max_qp |Q_i|`` for Maxwell branch ``i``. Under a
        constant-strain hold these decay as ``Q_n ~ exp(-dt/tau_i) Q_old``.
        """
        Qs = np.stack([s.Q for s in self.state_committed])
        per_branch = np.max(np.abs(Qs), axis=(0, 2, 3))
        return float(np.max(np.abs(Qs))), per_branch

    def committed_time(self):
        """Physical time carried in the committed state (uniform across QPs)."""
        return float(self.state_committed[0].t)

    def committed_prony(self):
        """Per-QP Prony parameters carried in the committed state.

        Returns ``(tau, g_visc)``, each ``(n_qp, N_MAXWELL)``. These are the
        trained sub-ANN (``Tau_subANN`` / ``G_subANN``) outputs evaluated at each
        QP's current deformation inside ``NumpyVCANN.step``, so they vary across
        QPs and over a ramp, and hold steady under a fixed-strain hold.
        """
        tau = np.stack([s.tau for s in self.state_committed])
        g_visc = np.stack([s.g_visc for s in self.state_committed])
        return tau, g_visc
