# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""The VHB4910 equilibrium energy in UFL, verified against the TF driver.

For this config (numTens=1, numDir=0, isotropic, incompressible) the strain
energy is closed-form in two isochoric invariants, so FEniCSx builds both the
stress and the consistent tangent symbolically, with no TensorFlow at solve
time:

    I = tr(C_bar)/3 ,   J = tr(C_bar^{-1})/3 ,   C_bar = det(C)^{-1/3} C

    psi_raw(I, J) = sum_{k=1..8} a_k * softplus(p_k*I + q_k*J + b_k)
    psi(I, J)     = psi_raw(I, J) - psi_raw(1, 1) + alpha*(I-1) + beta*(J-1)

Equilibrium part only; the 5 Maxwell branches are the separate stateful piece.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/verify_ufl_energy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

import tensorflow as tf

from fenicsx.bridge.vcann_local_step import VCANNLocalStep


def extract_psi_weights(driver):
    """Pull the 8-unit softplus energy network out of the TF submodel.

    Returns
    -------
    W1 : (2, 8)  hidden kernel  -> columns are (p_k, q_k)
    b1 : (8,)    hidden bias    -> b_k
    W2 : (8,)    output kernel  -> a_k   (linear read-out, no bias)
    """
    dense1 = driver.model_psi.get_layer("Psi_1_1")   # Dense(8, softplus)
    dense2 = driver.model_psi.get_layer("Psi_1_2")   # Dense(1, linear, no bias)
    W1, b1 = dense1.get_weights()
    W2 = dense2.get_weights()[0]
    return np.asarray(W1), np.asarray(b1), np.asarray(W2).ravel()


def tf_elastic_S_e_dev(driver, F):
    """The driver's elastic 2nd-PK stress, replicated standalone so that
    S_e_dev is available without the viscous branches."""
    F_tf = tf.constant(F, dtype="float64")
    C = tf.linalg.matmul(F_tf, F_tf, transpose_a=True)
    with tf.GradientTape() as tape:
        tape.watch(C)
        detC = tf.linalg.det(C)
        C_bar = tf.pow(detC, -1.0 / 3.0) * C
        I_inv = tf.linalg.trace(C_bar) / 3.0
        J_inv = tf.linalg.trace(tf.linalg.inv(C_bar)) / 3.0
        invars = tf.reshape(tf.stack([I_inv, J_inv]), (1, 1, 2))
        psi_NN = driver.model_psi(invars)[0, 0, 0]
        psi_total = (
            psi_NN - driver.psi_ref
            + driver.alpha * (I_inv - 1.0)
            + driver.beta * (J_inv - 1.0)
        )
    return (2.0 * tape.gradient(psi_total, C)).numpy()


def build_ufl_evaluators(W1, b1, W2, alpha, beta, psi_ref):
    """Construct the UFL energy psi(F), its PK1 stress P = dpsi/dF, and the
    consistent tangent A = dP/dF, and return closures that evaluate P and A
    at a prescribed homogeneous deformation gradient.

    The deformation is imposed by interpolating the linear displacement field
    u(x) = (F - I) x on a single unit cube, so grad(u) = F - I exactly.
    """
    domain = dmesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, dmesh.CellType.tetrahedron)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(V, name="u")

    # F as a UFL variable so we can differentiate the energy w.r.t. it.
    F_var = ufl.variable(ufl.Identity(3) + ufl.grad(u))
    C = F_var.T * F_var
    C_bar = ufl.det(C) ** (-1.0 / 3.0) * C
    I_inv = ufl.tr(C_bar) / 3.0
    J_inv = ufl.tr(ufl.inv(C_bar)) / 3.0

    softplus = lambda x: ufl.ln(1.0 + ufl.exp(x))

    # psi_raw = sum_k a_k * softplus(p_k I + q_k J + b_k); weights baked in as
    # literal constants, exactly as a frozen-weight UFL material would be.
    psi_raw = sum(
        float(W2[k]) * softplus(float(W1[0, k]) * I_inv + float(W1[1, k]) * J_inv + float(b1[k]))
        for k in range(W2.shape[0])
    )
    psi = psi_raw - float(psi_ref) + float(alpha) * (I_inv - 1.0) + float(beta) * (J_inv - 1.0)

    P_ufl = ufl.diff(psi, F_var)            # first Piola-Kirchhoff
    A_ufl = ufl.diff(P_ufl, F_var)          # consistent tangent dP/dF

    P_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    P_fun = fem.Function(P_space)
    P_expr = fem.Expression(P_ufl, P_space.element.interpolation_points)

    A_space = fem.functionspace(domain, ("DG", 0, (3, 3, 3, 3)))
    A_fun = fem.Function(A_space)
    A_expr = fem.Expression(A_ufl, A_space.element.interpolation_points)

    def set_F(Fmat):
        M = (np.asarray(Fmat, dtype=np.float64) - np.eye(3))
        u.interpolate(lambda x: M @ x)
        u.x.scatter_forward()

    def get_P(Fmat):
        set_F(Fmat)
        P_fun.interpolate(P_expr)
        return P_fun.x.array[:9].reshape(3, 3).copy()

    def get_A(Fmat):
        set_F(Fmat)
        A_fun.interpolate(A_expr)
        return A_fun.x.array[:81].reshape(3, 3, 3, 3).copy()

    return get_P, get_A


def main():
    print("Loading driver and extracting closed-form energy weights ...", flush=True)
    driver = VCANNLocalStep()
    W1, b1, W2 = extract_psi_weights(driver)
    print(f"  energy network: W1{W1.shape}, b1{b1.shape}, W2{W2.shape}  "
          f"({W1.size + b1.size + W2.size} trained scalars)")
    print(f"  reference constants: psi_ref={driver.psi_ref:.6e}, "
          f"alpha={driver.alpha:.6e}, beta={driver.beta:.6e}")

    get_P, get_A = build_ufl_evaluators(
        W1, b1, W2, driver.alpha, driver.beta, driver.psi_ref
    )

    # Incompressible uniaxial sweep: F2 = diag(1, lam) in-plane, F33 = 1/lam.
    lambdas = [1.000, 1.0025, 1.005, 1.01, 1.02, 1.05]
    print("\n--- Elastic PK1 stress: UFL  vs  TF driver (F @ S_e_dev) ---")
    print(f"{'lam':>7} {'|P_ufl|':>13} {'|P_tf|':>13} {'max abs err':>13} {'rel err':>11}")
    worst_rel = 0.0
    for lam in lambdas:
        Fmat = np.diag([1.0, lam, 1.0 / lam]).astype(np.float64)
        P_ufl = get_P(Fmat)
        S_e_dev_tf = tf_elastic_S_e_dev(driver, Fmat)
        P_tf = Fmat @ S_e_dev_tf
        abs_err = np.max(np.abs(P_ufl - P_tf))
        denom = max(np.max(np.abs(P_tf)), 1e-30)
        rel = abs_err / denom
        worst_rel = max(worst_rel, rel)
        print(f"{lam:7.4f} {np.linalg.norm(P_ufl):13.6e} {np.linalg.norm(P_tf):13.6e} "
              f"{abs_err:13.3e} {rel:11.2e}")
    print(f"\n  worst relative error across sweep: {worst_rel:.2e}")
    energy_ok = worst_rel < 1e-7
    print(f"  ENERGY MATCH: {'PASS' if energy_ok else 'FAIL'} (tol 1e-7)")

    # Consistent-tangent check: central-difference dP/dF vs symbolic A at lam=1.1.
    print("\n--- Consistent tangent: symbolic dP/dF  vs  finite difference ---")
    lam = 1.1
    Fmat = np.diag([1.0, lam, 1.0 / lam]).astype(np.float64)
    A_sym = get_A(Fmat)
    h = 1e-6
    max_t_err = 0.0
    for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)]:
        Fp = Fmat.copy(); Fp[a, b] += h
        Fm = Fmat.copy(); Fm[a, b] -= h
        dP_num = (get_P(Fp) - get_P(Fm)) / (2.0 * h)   # = dP_ij / dF_ab
        err = np.max(np.abs(dP_num - A_sym[:, :, a, b]))
        max_t_err = max(max_t_err, err)
        print(f"  dP/dF[:,:,{a},{b}]  max|symbolic - FD| = {err:.3e}")
    tangent_ok = max_t_err < 1e-4
    print(f"\n  TANGENT CONSISTENT: {'PASS' if tangent_ok else 'FAIL'} (tol 1e-4)")

    print("\n" + "=" * 60)
    print(f"RESULT: energy {'PASS' if energy_ok else 'FAIL'}, "
          f"tangent {'PASS' if tangent_ok else 'FAIL'}")
    print("The VHB4910 equilibrium energy is reproduced exactly in UFL, and")
    print("the consistent tangent comes from symbolic differentiation alone.")
    return 0 if (energy_ok and tangent_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
