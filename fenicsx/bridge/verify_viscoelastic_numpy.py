# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""The full VHB4910 local step in pure NumPy, verified against the TF driver.

Energy, Prony networks and the generalized-Maxwell recurrence, with the weights
lifted off the TF layers once at construction. Carrying the history variables
``Q_j`` between steps is the only stateful part, and is exactly the
quadrature-point state an FE coupling stores.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/verify_viscoelastic_numpy.py
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf

from fenicsx.bridge.vcann_local_step import (
    VCANNLocalStep,
    VCANNState,
    N_MAXWELL,
    TAU_SCALE,
)
from fenicsx.bridge.histories import make_uniaxial_history


def _softplus(x):
    return np.logaddexp(0.0, x)          # ln(1 + e^x), numerically stable


def _sigmoid(x):
    # 0.5*(1+tanh(x/2)) == 1/(1+e^-x) exactly, but tanh saturates instead of
    # overflowing for large |x|, and is a single branch.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _dense(x, W, b, act):
    z = x @ W + b
    return act(z)


class NumpyVCANN:
    """Mirror of :class:`VCANNLocalStep` with no TF at inference time.

    Weights are extracted once from a built TF driver; thereafter the step is
    pure NumPy, including the analytic energy gradient.
    """

    def __init__(self, driver: VCANNLocalStep):
        # Energy network: 8-unit softplus plus linear read-out.
        Wp1, bp1 = driver.model_psi.get_layer("Psi_1_1").get_weights()
        Wp2 = driver.model_psi.get_layer("Psi_1_2").get_weights()[0]
        self.Wp1 = np.asarray(Wp1)          # (2, 8)
        self.bp1 = np.asarray(bp1)          # (8,)
        self.a = np.asarray(Wp2).ravel()    # (8,)  output weights
        self.alpha = float(driver.alpha)
        self.beta = float(driver.beta)
        self.psi_ref = float(driver.psi_ref)

        # Prony g sub-networks: 6 branches, Dense(4,softplus)+Dense(1,softplus).
        self.g_layers = []
        for jj in range(N_MAXWELL + 1):
            W1, b1 = driver.model_g.get_layer(f"G_1_{jj}_1").get_weights()
            W2, b2 = driver.model_g.get_layer(f"G_1_{jj}_2").get_weights()
            self.g_layers.append((np.asarray(W1), np.asarray(b1),
                                  np.asarray(W2), np.asarray(b2)))

        # Prony tau sub-networks: 5 branches, Dense(4,softplus)+Dense(1,softplus).
        self.tau_layers = []
        for jj in range(N_MAXWELL):
            W1, b1 = driver.model_tau.get_layer(f"Tau_1_{jj+1}_1").get_weights()
            W2, b2 = driver.model_tau.get_layer(f"Tau_1_{jj+1}_2").get_weights()
            self.tau_layers.append((np.asarray(W1), np.asarray(b1),
                                    np.asarray(W2), np.asarray(b2)))

        self.tau_scale = TAU_SCALE.numpy().astype(np.float64)   # (5,)

    @staticmethod
    def _invariants(C):
        j = np.linalg.det(C)
        C_bar = j ** (-1.0 / 3.0) * C
        I = np.trace(C_bar) / 3.0
        J = np.trace(np.linalg.inv(C_bar)) / 3.0
        return I, J

    # Analytic 2nd-PK elastic stress S_e = 2 dpsi/dC.
    def _elastic_S_e_dev(self, F):
        C = F.T @ F
        j = np.linalg.det(C)
        Cinv = np.linalg.inv(C)
        trC = np.trace(C)
        trCinv = np.trace(Cinv)
        I = (1.0 / 3.0) * j ** (-1.0 / 3.0) * trC
        J = (1.0 / 3.0) * j ** (1.0 / 3.0) * trCinv

        # dPsi/dI, dPsi/dJ from the closed-form energy (+ Linden offset).
        z = self.Wp1[0, :] * I + self.Wp1[1, :] * J + self.bp1     # (8,)
        sig = _sigmoid(z)
        dPsi_dI = np.sum(self.a * self.Wp1[0, :] * sig) + self.alpha
        dPsi_dJ = np.sum(self.a * self.Wp1[1, :] * sig) + self.beta

        # dI/dC and dJ/dC (matrix calculus on isochoric invariants).
        eye = np.eye(3)
        jm13 = j ** (-1.0 / 3.0)
        j13 = j ** (1.0 / 3.0)
        dIdC = (1.0 / 3.0) * jm13 * (eye - (1.0 / 3.0) * trC * Cinv)
        dJdC = (1.0 / 3.0) * j13 * ((1.0 / 3.0) * trCinv * Cinv - Cinv @ Cinv)

        dPsi_dC = dPsi_dI * dIdC + dPsi_dJ * dJdC
        return 2.0 * dPsi_dC, (I, J)

    def _prony_at(self, I, J):
        x = np.array([I, J], dtype=np.float64)
        g_raw = np.array([_dense(_dense(x, W1, b1, _softplus), W2, b2, _softplus)[0]
                          for (W1, b1, W2, b2) in self.g_layers])    # (6,)
        tau_raw = np.array([_dense(_dense(x, W1, b1, _softplus), W2, b2, _softplus)[0]
                            for (W1, b1, W2, b2) in self.tau_layers])  # (5,)
        g_norm = g_raw / g_raw.sum()
        g_inf = float(g_norm[0])
        g_visc = g_norm[1:].astype(np.float64)
        tau_phys = (tau_raw * self.tau_scale).astype(np.float64)
        return g_inf, g_visc, tau_phys

    def initial_state(self, t0=0.0):
        _, g_visc_0, tau_0 = self._prony_at(1.0, 1.0)
        return VCANNState(
            Q=np.zeros((N_MAXWELL, 3, 3)),
            S_e=np.zeros((3, 3)),
            tau=tau_0,
            g_visc=g_visc_0,
            t=float(t0),
        )

    # Mirrors VCANNLocalStep.step.
    def step(self, F_n, dt, state: VCANNState):
        S_e_dev, (I, J) = self._elastic_S_e_dev(F_n)
        g_inf_n, g_visc_n, tau_n = self._prony_at(I, J)

        tau_bar = 0.5 * (tau_n + state.tau)
        g_bar = 0.5 * (g_visc_n + state.g_visc)
        decay = np.exp(-dt / tau_bar)
        drive = np.exp(-dt / (2.0 * tau_bar)) * g_bar
        delta_S_e = S_e_dev - state.S_e
        Q_n = decay[:, None, None] * state.Q + drive[:, None, None] * delta_S_e[None, :, :]

        S_dev = g_inf_n * S_e_dev + Q_n.sum(axis=0)

        C = F_n.T @ F_n
        Cinv = np.linalg.inv(C)
        p = S_dev[2, 2] / Cinv[2, 2]
        S = S_dev - p * Cinv
        P = F_n @ S

        new_state = VCANNState(Q=Q_n, S_e=S_e_dev, tau=tau_n, g_visc=g_visc_n, t=state.t + dt)
        return P, new_state

    # Batched variants, evaluating n_qp quadrature points simultaneously.
    def _elastic_S_e_dev_batch(self, F):
        """Vectorised elastic stress for F of shape (n, 3, 3) → (n, 3, 3), (n,), (n,)."""
        C = np.einsum('...ki,...kj->...ij', F, F)
        j = np.linalg.det(C)
        Cinv = np.linalg.inv(C)
        trC = np.einsum('...ii', C)
        trCinv = np.einsum('...ii', Cinv)
        I = (1.0 / 3.0) * j ** (-1.0 / 3.0) * trC
        J = (1.0 / 3.0) * j ** (1.0 / 3.0) * trCinv

        x = np.stack([I, J], axis=-1)
        z = x @ self.Wp1 + self.bp1
        sig = _sigmoid(z)
        dPsi_dI = sig @ (self.a * self.Wp1[0, :]) + self.alpha
        dPsi_dJ = sig @ (self.a * self.Wp1[1, :]) + self.beta

        eye = np.eye(3)
        jm13 = j ** (-1.0 / 3.0)
        j13  = j ** (1.0 / 3.0)
        dIdC = ((1.0 / 3.0) * jm13[:, None, None]
                * (eye - (1.0 / 3.0) * trC[:, None, None] * Cinv))
        Cinv2 = np.einsum('...ij,...jk->...ik', Cinv, Cinv)
        dJdC = ((1.0 / 3.0) * j13[:, None, None]
                * ((1.0 / 3.0) * trCinv[:, None, None] * Cinv - Cinv2))

        dPsi_dC = dPsi_dI[:, None, None] * dIdC + dPsi_dJ[:, None, None] * dJdC
        return 2.0 * dPsi_dC, (I, J)

    def _prony_at_batch(self, I, J):
        """Prony parameters for I, J of shape (n,) → (n,), (n, M), (n, M)."""
        x = np.stack([I, J], axis=-1)
        g_raw = np.stack([
            (_softplus((_softplus(x @ W1 + b1)) @ W2 + b2))[:, 0]
            for (W1, b1, W2, b2) in self.g_layers
        ], axis=-1)
        tau_raw = np.stack([
            (_softplus((_softplus(x @ W1 + b1)) @ W2 + b2))[:, 0]
            for (W1, b1, W2, b2) in self.tau_layers
        ], axis=-1)
        g_norm = g_raw / g_raw.sum(axis=-1, keepdims=True)
        g_inf  = g_norm[:, 0]
        g_visc = g_norm[:, 1:]
        tau_phys = tau_raw * self.tau_scale[None, :]
        return g_inf, g_visc, tau_phys

    def batch_step(self, F_batch, dt, states):
        """Vectorised step for n_qp QPs simultaneously.

        Parameters
        ----------
        F_batch : (n_qp, 3, 3)
        dt : float
        states : list of n_qp VCANNState

        Returns
        -------
        P_batch : (n_qp, 3, 3)
        new_states : list of n_qp VCANNState
        """
        n = F_batch.shape[0]
        Q_old    = np.stack([s.Q      for s in states])
        S_e_old  = np.stack([s.S_e    for s in states])
        tau_old  = np.stack([s.tau    for s in states])
        g_old    = np.stack([s.g_visc for s in states])
        t        = states[0].t

        S_e_dev, (I, J) = self._elastic_S_e_dev_batch(F_batch)
        g_inf_n, g_visc_n, tau_n = self._prony_at_batch(I, J)

        tau_bar = 0.5 * (tau_n + tau_old)
        g_bar   = 0.5 * (g_visc_n + g_old)
        decay   = np.exp(-dt / tau_bar)
        drive   = np.exp(-dt / (2.0 * tau_bar)) * g_bar
        delta_S_e = S_e_dev - S_e_old
        Q_n = (decay[:, :, None, None] * Q_old
               + drive[:, :, None, None] * delta_S_e[:, None, :, :])

        S_dev = g_inf_n[:, None, None] * S_e_dev + Q_n.sum(axis=1)

        # Incompressibility closure: enforce S[2, 2] = 0
        C    = np.einsum('...ki,...kj->...ij', F_batch, F_batch)
        Cinv = np.linalg.inv(C)
        p    = S_dev[:, 2, 2] / Cinv[:, 2, 2]
        S    = S_dev - p[:, None, None] * Cinv
        P    = np.einsum('...ij,...jk->...ik', F_batch, S)

        t_new = t + dt
        new_states = [
            VCANNState(Q=Q_n[q], S_e=S_e_dev[q], tau=tau_n[q], g_visc=g_visc_n[q], t=t_new)
            for q in range(n)
        ]
        return P, new_states


# Reference: the driver's elastic S_e_dev via TF autodiff.
def tf_elastic_S_e_dev(driver, F):
    F_tf = tf.constant(F, dtype="float64")
    C = tf.linalg.matmul(F_tf, F_tf, transpose_a=True)
    with tf.GradientTape() as tape:
        tape.watch(C)
        detC = tf.linalg.det(C)
        C_bar = tf.pow(detC, -1.0 / 3.0) * C
        I = tf.linalg.trace(C_bar) / 3.0
        J = tf.linalg.trace(tf.linalg.inv(C_bar)) / 3.0
        invars = tf.reshape(tf.stack([I, J]), (1, 1, 2))
        psi = driver.model_psi(invars)[0, 0, 0] - driver.psi_ref \
            + driver.alpha * (I - 1.0) + driver.beta * (J - 1.0)
    return (2.0 * tape.gradient(psi, C)).numpy()


def main():
    print("Building TF driver and extracting weights into NumPy model ...", flush=True)
    driver = VCANNLocalStep()
    npy = NumpyVCANN(driver)

    # Stage 1: analytic energy gradient vs TF autodiff.
    print("\n--- Stage 1: analytic dpsi/dC (NumPy) vs autodiff (TF) ---")
    print(f"{'lam':>7} {'max abs err':>14} {'rel err':>11}")
    worst = 0.0
    for lam in [1.0025, 1.005, 1.01, 1.02, 1.05, 1.1]:
        F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])
        S_np, _ = npy._elastic_S_e_dev(F)
        S_tf = tf_elastic_S_e_dev(driver, F)
        err = np.max(np.abs(S_np - S_tf))
        rel = err / max(np.max(np.abs(S_tf)), 1e-30)
        worst = max(worst, rel)
        print(f"{lam:7.4f} {err:14.3e} {rel:11.2e}")
    grad_ok = worst < 1e-9
    print(f"  worst rel err: {worst:.2e}  ->  {'PASS' if grad_ok else 'FAIL'} (tol 1e-9)")

    # Stage 2: full viscoelastic step over a uniaxial history.
    print("\n--- Stage 2: full step over uniaxial history (NumPy vs TF driver) ---")
    n_steps = 200
    F_hist, t_hist = make_uniaxial_history(n_steps=n_steps, lambda_max=1.2)

    st_tf = driver.initial_state(t0=0.0)
    st_np = npy.initial_state(t0=0.0)
    prev_t = 0.0
    max_abs = 0.0
    max_rel = 0.0
    P_tf_axial = np.zeros(n_steps)
    P_np_axial = np.zeros(n_steps)
    for n in range(n_steps):
        dt = float(t_hist[n] - prev_t)
        P_tf, st_tf = driver.step(F_hist[n], dt=dt, state=st_tf)
        P_np, st_np = npy.step(F_hist[n], dt=dt, state=copy.deepcopy(st_np))
        prev_t = float(t_hist[n])
        P_tf_axial[n] = P_tf[0, 0]
        P_np_axial[n] = P_np[0, 0]
        abs_err = np.max(np.abs(P_tf - P_np))
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, abs_err / max(np.max(np.abs(P_tf)), 1e-30))

    print(f"  {'step':>5} {'lambda':>8} {'P11_tf':>13} {'P11_np':>13} {'abs err':>11}")
    for n in [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]:
        print(f"  {n:5d} {F_hist[n,0,0]:8.4f} {P_tf_axial[n]:13.6e} "
              f"{P_np_axial[n]:13.6e} {abs(P_tf_axial[n]-P_np_axial[n]):11.2e}")
    print(f"\n  max abs error over all {n_steps} steps, all components: {max_abs:.3e}")
    print(f"  max rel error: {max_rel:.2e}")
    step_ok = max_rel < 1e-9
    print(f"  FULL-STEP MATCH: {'PASS' if step_ok else 'FAIL'} (tol 1e-9)")

    print("\n" + "=" * 62)
    print(f"RESULT: energy-gradient {'PASS' if grad_ok else 'FAIL'}, "
          f"full-step {'PASS' if step_ok else 'FAIL'}")
    print("The entire VHB4910 local step runs with no TensorFlow at inference;")
    print("only the history variables Q_j need carrying between steps.")
    return 0 if (grad_ok and step_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
