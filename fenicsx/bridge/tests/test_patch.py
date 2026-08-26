# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Single-element patch test for the vCANN <-> FEniCSx coupling.

With u = A*x for constant A, F = I + A is constant across all quadrature points,
so filling P_qf from VCANNLocalStep.step must assemble to the same residual as a
fem.Constant stress:

    inner(P_qf, grad(v)) * dx  ==  inner(P_const, grad(v)) * dx

This validates quadrature-element construction, matching-degree wiring,
fem.Expression interpolation and the per-QP indexing layout. Newton and time
evolution are not exercised.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/bridge/tests/test_patch.py
"""

from pathlib import Path
import sys

# Make `fenicsx.bridge.*` and `fenicsx.models.*` importable when this file is
# executed directly (not as a package). The project root is three levels up:
# tests/ -> bridge/ -> fenicsx/ -> <root>.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh

from fenicsx.bridge.vcann_local_step import VCANNLocalStep
from fenicsx.models.kinematics import (
    embed_incompressible_2d_in_3d,
    plane_deformation_gradient,
)

K_QUAD = 4                                  # quadrature degree
DT = 1.0                                    # any positive dt; not exercised here
A_MATRIX = np.array([[0.1, 0.0],            # u(x) = A_MATRIX @ x  -> F2 = I + A
                     [0.0, 0.0]], dtype=np.float64)


def make_quadrature_function(domain, value_shape, degree):
    """
    Allocate a DOLFINx Function on a Quadrature element.

    The Quadrature family stores one DOF per quadrature point of the cell's
    integration rule, with no shape functions defined off the DOFs. This makes
    it transparent storage for QP-local data (stress, deformation gradient,
    history variables) — what you write at QP q is what gets read at QP q.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh the Quadrature element will live on. Needed because the
        quadrature scheme depends on the cell topology (triangle vs tet, etc.).
    value_shape : tuple[int, ...]
        Shape of the value stored at each QP. Use (3, 3) for a 3D tensor like
        F or P; () for a scalar; (3,) for a vector.
    degree : int
        Quadrature degree. MUST match the `quadrature_degree` of the dx
        measure that this Function will appear in — otherwise assembly will
        ask for values at QP locations that aren't this element's DOFs.

    Returns
    -------
    qf_function : dolfinx.fem.Function
        The allocated Function. Per-QP values live in `qf_function.x.array`.
    qf_element : basix.ufl._ElementBase
        The element spec — needed downstream for fem.Expression construction
        (qf_element.interpolation_points gives the QP coordinates in the
        reference cell).
    """
    qf_element = basix.ufl.quadrature_element(
        cell=domain.basix_cell(),
        value_shape=value_shape,
        scheme="default",
        degree=degree,
    )
    qf_space = fem.functionspace(domain, qf_element)
    qf_function = fem.Function(qf_space)
    return qf_function, qf_element


def fill_P_qf_from_F_qf(F_qf, P_qf, driver, dt, state):
    """
    Fill the P_qf Function by calling driver.step at each quadrature point.

    Parameters
    ----------
    F_qf : dolfinx.fem.Function
        Quadrature-function holding the deformation gradient F at each QP.
        Shape should be (N_QP, 3, 3) when reshaped, where N_QP is the number of
        quadrature points in the cell.
    P_qf : dolfinx.fem.Function
        Quadrature-function to be filled with the resulting stress P at each QP.
        Should have the same shape and number of DOFs as F_qf.
    driver : VCANNLocalStep
        The vCANN local step driver, which has a .step(state, F, dt) method that
        returns the updated state and the computed P for a given input F.
    dt : float
        Time step size to pass to driver.step. Not exercised in this patch test,
        so can be any positive value.

    state : list[dict]
        List of length N_QP, where each entry is the state dict for the
        corresponding quadrature point. This state will be updated in-place by
        driver.step.
    
    Returns
    -------
    None
        The P_qf Function is filled in-place.
    """
    F_arr = F_qf.x.array.reshape(-1, 3, 3)  # shape (N_QP, 3, 3)
    P_arr = P_qf.x.array.reshape(-1, 3, 3)  # same shape as F_arr

    for q in range(F_arr.shape[0]):
        F_q = F_arr[q]
        P_q, state[q] = driver.step(F_q, dt, state[q])
        P_arr[q] = P_q
    
    return None


def main() -> None:
    
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [1, 1],
        cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    u.interpolate(lambda x: A_MATRIX @ x[:2, :])  # ignore z-coord if present
    v = ufl.TestFunction(V)

    F_qf, qf_element = make_quadrature_function(domain, value_shape=(3, 3), degree=K_QUAD)
    P_qf, _          = make_quadrature_function(domain, value_shape=(3, 3), degree=K_QUAD) 

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": K_QUAD})

    N_QP = P_qf.x.array.size // 9   # 9 = product of (3, 3) value shape
    local_step = VCANNLocalStep()
    state_committed = [local_step.initial_state(t0=0.0) for _ in range(N_QP)]

    # --- merge point: UFL F expression -> Quadrature buffer ---
    F2 = plane_deformation_gradient(u)
    F3 = embed_incompressible_2d_in_3d(F2)
    F_expr = fem.Expression(F3, F_qf.function_space.element.interpolation_points)
    F_qf.interpolate(F_expr)

    F_arr = F_qf.x.array.reshape(N_QP, 3, 3)
    print(f"N_QP = {N_QP}")
    print(f"F at QP 0:\n{F_arr[0]}")
    print(f"max |F[q] - F[0]| across QPs = {np.abs(F_arr - F_arr[0]).max():.3e}")

    F_expected = np.eye(3)
    F_expected[:2, :2] += A_MATRIX
    F_expected[2, 2] = 1.0 / np.linalg.det(F_expected[:2, :2])  # incompressible: F33 = 1/det(F2)

    err_F = np.abs(F_arr - F_expected).max()
    print(f"max |F[q] - F_expected| = {err_F:.3e}")
    assert err_F < 1e-12, f"F interpolation wrong: {err_F:.3e}"

    fill_P_qf_from_F_qf(F_qf, P_qf, local_step, dt=DT, state=state_committed)
    P_arr = P_qf.x.array.reshape(N_QP, 3, 3)
    print(f"P at QP 0:\n{P_arr[0]}")
    print(f"max |P[q] - P[0]| across QPs = {np.abs(P_arr - P_arr[0]).max():.3e}")

    # --- Stage 1 patch test: form_A (via P_qf) vs form_B (via fem.Constant) ---
    # P is 3x3 but grad(v) is 2x2 (mesh is 2D, u is 2-component). Slice P to its
    # in-plane 2x2 block; the out-of-plane row/column do not couple to in-plane v.
    P_qf_2 = ufl.as_matrix([[P_qf[0, 0], P_qf[0, 1]],
                            [P_qf[1, 0], P_qf[1, 1]]])

    # Reference: a single constant P value (all QPs hold the same one in Stage 1).
    P_const_value = P_arr[0]
    P_const = fem.Constant(domain, P_const_value)
    P_const_2 = ufl.as_matrix([[P_const[0, 0], P_const[0, 1]],
                               [P_const[1, 0], P_const[1, 1]]])

    form_A = ufl.inner(P_qf_2,    ufl.grad(v)) * dx
    form_B = ufl.inner(P_const_2, ufl.grad(v)) * dx

    vec_A = fem.assemble_vector(fem.form(form_A))
    vec_B = fem.assemble_vector(fem.form(form_B))

    diff = float(np.abs(vec_A.array - vec_B.array).max())
    print(f"\nmax |vec_A - vec_B| = {diff:.3e}")
    assert diff < 1e-12, (
        f"patch test FAILED: form_A and form_B disagree by {diff:.3e} "
        f"(should be <1e-12)"
    )
    print("patch test PASSED")




if __name__ == "__main__":
    main()
