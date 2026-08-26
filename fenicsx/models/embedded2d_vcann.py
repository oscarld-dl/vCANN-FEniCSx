# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""vCANN-driven NonlinearProblem on the embedded-2D mesh.

Residual ``inner(P_qf, grad(v)) * dx`` with ``P_qf`` filled per quadrature point
from the VHB4910 driver, but the Jacobian left as the Neo-Hookean energy second
derivative: a frozen tangent, deliberately inconsistent with the residual,
which converges linearly at best. ``embedded2d_vcann_extop.py`` is the
consistent-tangent successor and the one to read first.

DOLFINx 0.10 exposes no ``F()`` method to override, so the SNES residual is
registered through ``solver.setFunction``.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/models/embedded2d_vcann.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.la.petsc as la_petsc
from dolfinx import fem
from dolfinx.fem.petsc import (
    LinearProblem,
    NonlinearProblem,
    apply_lifting,
    assemble_vector,
    assign,
    set_bc,
)
from dolfinx import mesh as dmesh
from dolfinx.io import XDMFFile

from fenicsx.bridge.plane_stress import condense_plane_stress
from fenicsx.bridge.vcann_local_step import VCANNLocalStep
from fenicsx.models.kinematics import (
    embed_incompressible_2d_in_3d,
    plane_deformation_gradient,
)
from fenicsx.models.materials import neo_hookean_embedded_energy


_FENICSX_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _FENICSX_ROOT / "results"

K_QUAD = 4      # quadrature degree, matches test_patch.py
DT = 1.0        # one quasi-static step; harmless positive dt for the viscoelastic update
U_TOP = 0.01    # small top displacement — step 2 only proves the wiring
N_CELLS = 4     # NxN grid → 2*N*N triangles. Step 2 wants the cheapest mesh that
                # still exercises Newton on a multi-cell problem; the full
                # unitsquare_hole.msh has ~3800 elements and one residual eval
                # would take minutes through TF. Step 3 / step 4 can scale up.


class VCANNProblem(NonlinearProblem):
    """NonlinearProblem with a vCANN per-QP residual update.

    The SNES residual callback is replaced with one that:

    1. updates ``u`` from the current SNES iterate,
    2. interpolates ``F_qf`` from ``grad(u)``,
    3. calls ``condense_plane_stress`` per QP to fill ``P_qf``,
    4. then performs the standard residual assembly on the fresh ``P_qf``.

    State discipline:

    - ``self.state_committed`` is read-only inside the SNES callback.
    - ``self.state_trial`` is overwritten on every call.
    - The outer load loop calls :meth:`commit_state` after a successful
      :meth:`solve` to promote ``state_committed ← state_trial``.

    The condensation deepcopies state internally on every probe, so the
    committed state cannot be corrupted by failed Newton iterates.
    """

    def __init__(
        self,
        F,
        u,
        *,
        P_qf,
        F_qf,
        F_expr,
        driver,
        state_committed,
        bcs=None,
        J=None,
        **kwargs,
    ):
        bcs = list(bcs or [])
        super().__init__(F, u, bcs=bcs, J=J, **kwargs)
        self.P_qf = P_qf
        self.F_qf = F_qf
        self.F_expr = F_expr
        self.driver = driver
        self.state_committed = list(state_committed)
        self.state_trial = [copy.deepcopy(s) for s in state_committed]
        self.dt = DT
        self.N_QP = len(state_committed)
        self._bcs = bcs
        self._eval_count = 0
        # Replace the parent's SNES residual callback with our QP-aware one.
        self.solver.setFunction(self._vcann_residual, self.b)

    def _vcann_residual(self, snes, x, b):
        eval_start = time.perf_counter()
        self._eval_count += 1
        # 1. Pull SNES iterate into the displacement Function.
        la_petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        assign(x, self._u)

        # 2. Fresh F at every QP from the current u.
        self.F_qf.interpolate(self.F_expr)
        F_arr = self.F_qf.x.array.reshape(self.N_QP, 2, 2)
        P_arr = self.P_qf.x.array.reshape(self.N_QP, 2, 2)

        # 3. Per-QP plane-stress condensation. condense_plane_stress treats
        #    state_committed[q] as read-only (deepcopies internally on every
        #    probe), so failed line-search iterates cannot corrupt history.
        for q in range(self.N_QP):
            P_3d, self.state_trial[q], _ = condense_plane_stress(
                F_arr[q], self.driver, self.dt, self.state_committed[q]
            )
            P_arr[q] = P_3d[:2, :2]
        qp_time = time.perf_counter() - eval_start

        # 4. Standard residual assembly — same shape as dolfinx.fem.petsc
        #    .assemble_residual, just minus the duplicate u-update we did
        #    in step 1.
        la_petsc._zero_vector(b)
        assemble_vector(b, self._F)
        apply_lifting(b, [self._J], bcs=[self._bcs], x0=[x], alpha=-1.0)
        la_petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(b, self._bcs, x0=x, alpha=-1.0)
        la_petsc._ghost_update(b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        b_norm = b.norm()
        total_time = time.perf_counter() - eval_start
        print(
            f"  residual #{self._eval_count:2d}: |b| = {b_norm:.3e}  "
            f"(qp-loop {qp_time:.2f}s, total {total_time:.2f}s)",
            flush=True,
        )

    def commit_state(self) -> None:
        """Promote ``state_trial`` → ``state_committed``. Call this after a
        successful :meth:`solve`. If solve fails, do *not* call this — the
        committed state stays as it was and the step can be retried."""
        self.state_committed = [copy.deepcopy(s) for s in self.state_trial]


def build_problem():
    """Construct mesh, function spaces, residual + Jacobian forms, driver,
    and the :class:`VCANNProblem` instance ready to solve.

    Uses a small inline ``N_CELLS x N_CELLS`` mesh — step 2 is about
    wiring + Newton-converges-at-all, not full-resolution results.
    The full ``unitsquare_hole.msh`` is left for step 3+.
    """
    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N_CELLS, N_CELLS],
        cell_type=dmesh.CellType.triangle,
    )

    # Lagrange-2 displacement, matching embedded2d_hyperelastic.py.
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Dirichlet: bottom y=0 clamped both directions, top y=1 displaces in y.
    fdim = domain.topology.dim - 1
    bottom_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    top_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 1.0)
    )
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    top_dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)

    u_zero = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    top_displacement = fem.Constant(domain, PETSc.ScalarType(0.0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, V)
    bc_top_y = fem.dirichletbc(top_displacement, top_dofs_y, V.sub(1))
    bcs = [bc_bottom, bc_top_y]

    # Quadrature functions for F (input to driver) and P (output, in the form).
    # 2x2 in-plane: the 3D embedding lives entirely inside condense_plane_stress.
    qf_element = basix.ufl.quadrature_element(
        cell=domain.basix_cell(),
        value_shape=(2, 2),
        scheme="default",
        degree=K_QUAD,
    )
    F_qf = fem.Function(fem.functionspace(domain, qf_element), name="F_qf")
    P_qf = fem.Function(fem.functionspace(domain, qf_element), name="P_qf")

    F2_ufl = plane_deformation_gradient(u)
    F_expr = fem.Expression(F2_ufl, F_qf.function_space.element.interpolation_points)

    dx = ufl.Measure(
        "dx",
        domain=domain,
        metadata={"quadrature_degree": K_QUAD},
    )

    # New residual: P_qf is filled per QP inside the SNES callback.
    residual = ufl.inner(P_qf, ufl.grad(v)) * dx

    # Frozen tangent: Neo-Hookean energy second derivative on the 3D-embedded F.
    # Intentionally inconsistent with the residual — proving the wiring is the
    # goal here, not asymptotic quadratic Newton.
    F3_ufl = embed_incompressible_2d_in_3d(F2_ufl)
    mu = fem.Constant(domain, PETSc.ScalarType(1.0))
    psi = neo_hookean_embedded_energy(F3_ufl, mu)
    total_potential = psi * dx
    residual_NH = ufl.derivative(total_potential, u, v)
    jacobian = ufl.derivative(residual_NH, u, du)

    # Driver + per-QP committed state at the reference configuration.
    driver = VCANNLocalStep()
    N_QP = P_qf.x.array.size // 4   # 4 = product of (2, 2) value shape
    state_committed = [driver.initial_state(t0=0.0) for _ in range(N_QP)]

    problem = VCANNProblem(
        residual,
        u,
        P_qf=P_qf,
        F_qf=F_qf,
        F_expr=F_expr,
        driver=driver,
        state_committed=state_committed,
        bcs=bcs,
        J=jacobian,
        petsc_options_prefix="embedded2d_vcann_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_max_it": 50,    # frozen tangent may need extra iters
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    return domain, u, P_qf, problem, top_displacement


def project_qf_to_nodes(qf, degree, *, name=None):
    """L2-project a quadrature-element ``Function`` onto a nodal Lagrange-1 space.

    Quadrature-element functions (``F_qf``, ``P_qf``) live at integration
    points and have no nodal representation, so XDMF / ParaView cannot write
    them directly — only the interpolated ``u`` ever reached disk before.
    This solves the mass-matrix projection

        ``inner(P, w) * dx == inner(qf, w) * dx``

    onto a Lagrange-1 space of the *same value shape* as ``qf`` (so a 2x2
    quadrature tensor projects to a 2x2 nodal tensor), giving smoothed nodal
    values suitable for output and post-processing.

    The projection ``dx`` quadrature degree MUST match the quadrature
    element's degree — same matching-degree discipline as the residual form
    (see test_patch.py) — otherwise the RHS samples ``qf`` at the wrong
    points and the projection is silently wrong.

    Reusable for any future simulation: pass any quadrature ``Function`` and
    its degree, get back a nodal ``Function`` to write or inspect.
    """
    dom = qf.function_space.mesh
    V_nodal = fem.functionspace(dom, ("Lagrange", 1, qf.ufl_shape))
    w = ufl.TestFunction(V_nodal)
    P_trial = ufl.TrialFunction(V_nodal)
    dx_proj = ufl.Measure("dx", domain=dom, metadata={"quadrature_degree": degree})
    problem = LinearProblem(
        ufl.inner(P_trial, w) * dx_proj,
        ufl.inner(qf, w) * dx_proj,
        petsc_options_prefix="qf_projection_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-12,
        },
    )
    P_nodal = problem.solve()
    if name is not None:
        P_nodal.name = name
    return P_nodal


def solve_one_step() -> None:
    """Drive the problem through one quasi-static load step and report Newton
    behavior. Goal: confirm the callback fires and Newton converges *at all*."""
    domain, u, P_qf, problem, top_displacement = build_problem()

    top_displacement.value = PETSc.ScalarType(U_TOP)
    print(
        f"Solving one quasi-static step: "
        f"u_top = {U_TOP}, dt = {problem.dt}, N_QP = {problem.N_QP}",
        flush=True,
    )

    t0 = time.perf_counter()
    problem.solve()
    u.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    iterations = problem.solver.getIterationNumber()
    reason = problem.solver.getConvergedReason()
    converged = reason > 0

    print(f"\nSNES iterations  = {iterations}")
    print(f"residual evals   = {problem._eval_count}")
    print(f"solve wall time  = {solve_time:.1f}s")
    print(f"converged_reason = {reason}  ({'CONVERGED' if converged else 'FAILED'})")

    if not converged:
        raise RuntimeError(
            f"Newton did not converge: iterations={iterations}, reason={reason}. "
            "With the Neo-Hookean frozen tangent, total stall at this load level "
            "suggests upgrading to the FD tangent."
        )

    problem.commit_state()
    print("state_committed promoted from state_trial.")

    # Diagnostic dump.
    dof_values = u.x.array.reshape(-1, 2)
    print(
        f"u_x range = [{dof_values[:, 0].min():.3e}, {dof_values[:, 0].max():.3e}]"
    )
    print(
        f"u_y range = [{dof_values[:, 1].min():.3e}, {dof_values[:, 1].max():.3e}]"
    )

    P_arr = P_qf.x.array.reshape(problem.N_QP, 2, 2)
    print(f"max |P_qf|        = {np.max(np.abs(P_arr)):.3e}")

    # XDMF for visual sanity check: displacement plus the projected stress.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RESULTS_DIR / "embedded2d_vcann_one_step.xdmf"
    V_out = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u_out = fem.Function(V_out, name="u")
    u_out.interpolate(u)

    # P_qf lives at quadrature points — project to nodes so it can be written.
    P_nodal = project_qf_to_nodes(P_qf, K_QUAD, name="P")

    with XDMFFile(domain.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_out, float(U_TOP))
        xdmf.write_function(P_nodal, float(U_TOP))
    print(f"Wrote {output_path} (u + projected P)")


if __name__ == "__main__":
    solve_one_step()
