# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Plane-stress FEniCSx example using the vCANN external operator.

UFL holds ``inner(P, grad(v)) * dx`` with ``P = P(F)`` an external operator, so
``ufl.derivative`` produces a tangent consistent with the residual; see
``fenicsx/bridge/vcann_external_operator.py``. No TensorFlow at solve time.
``snes_monitor`` prints the per-iteration residual history.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/models/embedded2d_vcann_extop.py
"""

from __future__ import annotations

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

from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import (
    NonlinearProblem,
    assemble_jacobian,
    assemble_residual,
    assign,
)
import gmsh as gmsh_api
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

from fenicsx.bridge.vcann_external_operator import VCANNStressOperator
from fenicsx.bridge.vcann_local_step import VCANNLocalStep, N_MAXWELL
from fenicsx.bridge.verify_viscoelastic_numpy import NumpyVCANN
from fenicsx.models.embedded2d_vcann import project_qf_to_nodes
from fenicsx.models.kinematics import plane_deformation_gradient


_FENICSX_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _FENICSX_ROOT / "results"


K_QUAD = 4      # quadrature degree, matches test_patch.py / embedded2d_vcann.py
DT = 1.0        # s per load increment; tau is ~[0.01, 10] s, so dt ~ O(1)
                # relaxes the mid and slow Maxwell branches visibly
U_TOP = 0.1     # final top displacement, reached at the end of the ramp
N_CELLS = 4     # NxN grid -> 2*N*N triangles
N_RAMP = 5      # increments ramping u_top 0 -> U_TOP, small enough to stay in basin
N_HOLD = 5      # constant-strain increments after the ramp: F frozen, dt keeps
                # advancing, so Q decays


class VCANNExternalOperatorProblem(NonlinearProblem):
    """NonlinearProblem whose residual/Jacobian assembly first evaluates the
    vCANN external operators.

    SNES calls the residual then the Jacobian each iteration. In both we sync
    ``u`` from the iterate ``x`` *before* evaluating the operands (the operand
    is ``F = I + grad(u)``, so ``u`` must be current), evaluate the operator(s)
    needed by that form, then delegate to the standard
    ``assemble_residual`` / ``assemble_jacobian``.
    """

    def __init__(self, F, u, *, J, F_ops, J_ops, operator, bcs=None, **kwargs):
        bcs = list(bcs or [])
        super().__init__(F, u, bcs=bcs, J=J, **kwargs)
        self._u_fn = u
        self._bcs_list = bcs
        self.F_ops = F_ops
        self.J_ops = J_ops
        self.operator = operator
        self.eval_count = 0
        # Replace the parent's callbacks with operator-aware ones.
        self.solver.setFunction(self._residual, self.b)
        self.solver.setJacobian(self._jacobian, self.A, self.P_mat)

    def _sync(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assign(x, self._u_fn)

    def _residual(self, snes, x, b):
        # Evaluates only F_ops (the (0,) stress), not the expensive (1,) tangent.
        # assemble_residual's Dirichlet lifting (b -= K*(g - u) on constrained
        # dofs) reads the Jacobian form self.J, so the tangent would have to be
        # current here unless the iterate already satisfies the Dirichlet BC.
        # solve_load_steps seeds it so that it does: (g - u) = 0 on constrained
        # dofs, the lifting term vanishes however stale the tangent coefficient
        # is, and the residual is a tangent-free function of u. Newton corrections
        # are zero on constrained dofs, so every iterate and line-search trial
        # stays BC-satisfying.
        #
        # CONTRACT: correct only while the iterate satisfies the Dirichlet BC.
        # Driving this problem without the BC seed requires evaluating
        # self.F_ops + self.J_ops here instead.
        self.eval_count += 1
        self._sync(x)
        operands = evaluate_operands(self.F_ops)
        evaluate_external_operators(self.F_ops, operands)
        assemble_residual(self._u_fn, self.F, self.J, self._bcs_list, snes, x, b)

    def _jacobian(self, snes, x, A, P):
        # The tangent (1,) operator lives here: the residual no longer evaluates
        # it (see _residual / the BC-seed contract), so refresh it before assembly.
        self._sync(x)
        operands = evaluate_operands(self.J_ops)
        evaluate_external_operators(self.J_ops, operands)
        assemble_jacobian(
            self._u_fn, self.J, self.preconditioner, self._bcs_list, snes, x, A, P
        )

    def commit_state(self):
        self.operator.commit()


def build_problem(dt=DT, snes_monitor=False, mesh_path=None):
    """Mesh, spaces, external-operator forms, and the problem instance.

    ``mesh_path`` — optional path to a Gmsh ``.msh`` file (e.g.
    ``unitsquare_hole.msh``). The file must define physical groups named
    ``"bottom"``, ``"top"``, and ``"domain"`` (same convention as
    ``embedded2d_hyperelastic.py``). When ``None`` a plain N_CELLS×N_CELLS
    triangle rectangle is generated in memory.
    """
    if mesh_path is not None:
        # Meshes from fenicsx/meshes/unitsquare_hole.py already carry named
        # physical groups ("bottom", "top", "domain", ...), tagged by coordinate
        # classification at generation time, so read those directly rather than
        # re-adding groups by curve ID: those IDs are not stable across
        # regenerations.
        gmsh_api.initialize()
        gmsh_api.option.setNumber("General.Terminal", 0)
        gmsh_api.open(str(mesh_path))
        mesh_data = model_to_mesh(gmsh_api.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh_api.finalize()
        domain = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
        physical = mesh_data.physical_groups
    else:
        domain = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
            [N_CELLS, N_CELLS],
            cell_type=dmesh.CellType.triangle,
        )
        facet_tags = None
        physical = None

    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    fdim = domain.topology.dim - 1
    if facet_tags is not None:
        # Gmsh mesh: BCs identified by physical-group tags.
        bottom_dofs = fem.locate_dofs_topological(
            V, fdim, facet_tags.find(physical["bottom"].tag)
        )
        top_dofs_y = fem.locate_dofs_topological(
            V.sub(1), fdim, facet_tags.find(physical["top"].tag)
        )
    else:
        # Rectangle mesh: BCs identified by coordinate predicate.
        bottom = dmesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], 0.0)
        )
        top = dmesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], 1.0)
        )
        bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom)
        top_dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, top)

    u_zero = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    top_disp = fem.Constant(domain, PETSc.ScalarType(0.0))
    bcs = [
        fem.dirichletbc(u_zero, bottom_dofs, V),
        fem.dirichletbc(top_disp, top_dofs_y, V.sub(1)),
    ]

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": K_QUAD})

    # External-operator output space: 2x2 in-plane PK1, on the quadrature element
    # whose degree matches dx (same matching-degree discipline as test_patch.py).
    Qe = basix.ufl.quadrature_element(
        cell=domain.basix_cell(), value_shape=(2, 2), scheme="default", degree=K_QUAD
    )
    Q_P = fem.functionspace(domain, Qe)
    n_qp = fem.Function(Q_P).x.array.size // 4   # local quadrature-point count

    # TF-free step (weights lifted off the driver once, here at build time).
    driver = VCANNLocalStep()
    npy = NumpyVCANN(driver)
    operator = VCANNStressOperator(npy, dt, n_qp)

    # The external operator P(F), with F2 = I + grad(u) as its operand.
    F2 = plane_deformation_gradient(u)
    P = FEMExternalOperator(
        F2, function_space=Q_P, external_function=operator.external_function
    )

    residual = ufl.inner(P, ufl.grad(v)) * dx
    jacobian = ufl.derivative(residual, u, du)   # consistent tangent via dP operator

    # Swap the symbolic external operators for their coefficient stand-ins, and
    # collect the operator lists each form needs evaluated before assembly.
    F_replaced, F_ops = replace_external_operators(residual)
    J_replaced, J_ops = replace_external_operators(jacobian)

    petsc_options = {
        "snes_type": "newtonls",
        # Armijo backtracking. Viable because the N_RAMP load increments keep
        # every step inside the quadratic basin; applying u_top in one shot from
        # u=0 does not, and "bt" then rejects the correct first full step because
        # it raises the residual before it falls. Use "basic" (full Newton step)
        # to isolate the FD tangent's own per-increment convergence behaviour.
        "snes_linesearch_type": "bt",
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if snes_monitor:
        petsc_options["snes_monitor"] = None   # per-iteration residual history
        petsc_options["snes_linesearch_monitor"] = None  # why bt accepts/rejects

    problem = VCANNExternalOperatorProblem(
        F_replaced,
        u,
        J=J_replaced,
        F_ops=F_ops,
        J_ops=J_ops,
        operator=operator,
        bcs=bcs,
        petsc_options_prefix="embedded2d_vcann_extop_",
        petsc_options=petsc_options,
    )
    return domain, u, problem, top_disp, Q_P


def solve_load_steps(n_ramp=N_RAMP, n_hold=N_HOLD, dt=DT, snes_monitor=False,
                     mesh_path=None):
    """Quasi-static load-stepping with a carried viscoelastic history.

    Two phases over a single :class:`VCANNExternalOperatorProblem`:

    * **ramp** (``n_ramp`` increments): ``u_top`` rises linearly 0 -> ``U_TOP``.
      Each increment advances physical time by ``dt`` and is committed, so the
      Maxwell history ``Q`` accumulates and, from the second increment on, the
      recurrence reads a nonzero ``Q_old``. Small increments also keep Newton
      inside its basin.
    * **hold** (``n_hold`` increments): ``u_top`` is frozen at ``U_TOP`` while
      ``dt`` keeps advancing. With ``F`` ~ constant, ``delta_S_e -> 0`` and
      ``Q_n ~ exp(-dt/tau)*Q_old``, i.e. pure decay, so the stress relaxes at
      fixed strain. This only happens if ``Q_old`` is carried across steps.

    ``n_ramp=1, n_hold=0`` recovers a single-shot solve.
    """
    domain, u, problem, top_disp, Q_P = build_problem(
        dt=dt, snes_monitor=snes_monitor, mesh_path=mesh_path
    )
    op = problem.operator
    n_steps = n_ramp + n_hold
    print(
        f"Option A load-stepping: U_TOP={U_TOP}, dt={dt}, "
        f"n_ramp={n_ramp}, n_hold={n_hold}, N_QP={op.n_qp}\n",
        flush=True,
    )
    print(f"  {'k':>3} {'phase':>5} {'t':>7} {'u_top':>8} {'its':>4} "
          f"{'evals':>6} {'max|P|':>11} {'max|Q|':>11}")

    # Time-series XDMF: open once, write the fixed mesh once, then append the
    # committed state of every increment keyed by physical time t_now. P lives at
    # quadrature points (Q_P) and is L2-projected to nodal Lagrange-1 each step,
    # the only vector/tensor representation XDMF can hold. The field names
    # ("u", "P") stay fixed so ParaView reads them as one time-varying field.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RESULTS_DIR / "embedded2d_vcann_extop.xdmf"
    V_out = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u_out = fem.Function(V_out, name="u")
    P_qf = fem.Function(Q_P, name="P_qf")
    xdmf = XDMFFile(domain.comm, str(output_path), "w")
    xdmf.write_mesh(domain)

    t0 = time.perf_counter()
    prev_evals = 0
    prev_target = 0.0   # u_top of the previous increment, for the seed scaling
    history = []
    prony_log = []   # (tau, g_visc) per increment, each (n_qp, N_MAXWELL)
    for k in range(1, n_steps + 1):
        if k <= n_ramp:
            phase = "ramp"
            u_target = U_TOP * k / n_ramp
        else:
            phase = "hold"
            u_target = U_TOP
        top_disp.value = PETSc.ScalarType(u_target)

        # Seed the iterate with a smooth, BC-satisfying initial guess. This is
        # what lets _residual skip the tangent: with u already satisfying the
        # Dirichlet BC the lifting term K*(g - u) vanishes (see _residual's
        # contract). Smoothness matters too — set_bc alone on a zero interior
        # puts the whole jump in one element layer (~210% strain) and cripples
        # convergence.
        #   * first loaded increment: linear ramp u_y = y * u_target (uniform
        #     strain, exactly satisfies bottom u=0 / top u=u_target);
        #   * later increments: scale the previous converged field by the load
        #     ratio (stays smooth and near-equilibrium; ratio=1 on hold).
        # set_bc then pins the constrained dofs to g exactly, removing any
        # roundoff in the gap so the lifting is exactly zero.
        if prev_target == 0.0:
            u.interpolate(lambda xx: np.vstack((np.zeros_like(xx[1]),
                                                u_target * xx[1])))
        elif u_target != prev_target:
            u.x.array[:] *= (u_target / prev_target)
        fem.set_bc(u.x.petsc_vec, problem._bcs_list)
        u.x.scatter_forward()
        prev_target = u_target

        problem.solve()
        u.x.scatter_forward()
        reason = problem.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"increment {k} ({phase}, u_top={u_target:.4g}) did not converge "
                f"(reason={reason}). Re-run with snes_monitor=True: a stall above "
                "tol points at the FD tangent floor (central diff / smaller dt or "
                "increment), divergence at a wiring bug."
            )

        # Diagnostics from the converged but not yet committed trial state, then
        # promote trial -> committed so the next increment reads this Q. State is
        # committed only after Newton has converged.
        max_P = float(np.max(np.abs(op.last_P)))
        iters = problem.solver.getIterationNumber()
        d_evals = problem.eval_count - prev_evals
        prev_evals = problem.eval_count
        problem.commit_state()
        max_Q, per_branch = op.committed_Q_stats()
        t_now = op.committed_time()
        history.append((k, phase, t_now, u_target, iters, max_P, max_Q, per_branch))
        prony_log.append(op.committed_prony())
        print(f"  {k:3d} {phase:>5} {t_now:7.2f} {u_target:8.4f} {iters:4d} "
              f"{d_evals:6d} {max_P:11.4e} {max_Q:11.4e}")

        # Append this increment to the time series at its physical time. The flat
        # order of op.last_P matches Q_P's dofs.
        u_out.interpolate(u)
        P_qf.x.array[:] = op.last_P.reshape(-1)
        P_qf.x.scatter_forward()
        P_nodal = project_qf_to_nodes(P_qf, K_QUAD, name="P")
        xdmf.write_function(u_out, t_now)
        xdmf.write_function(P_nodal, t_now)

    wall = time.perf_counter() - t0
    print(f"\n  total wall {wall:.2f}s, {problem.eval_count} residual evals over "
          f"{n_steps} increments")

    _report_recurrence(history, n_ramp, n_hold)

    # Representative QP: the most viscoelastically active point, i.e. the largest
    # carried |Q| at the final state.
    Qs = np.stack([s.Q for s in op.state_committed])
    rep_qp = int(np.argmax(np.max(np.abs(Qs), axis=(1, 2, 3))))
    _report_prony(history, prony_log, rep_qp)

    xdmf.close()

    uv = u.x.array.reshape(-1, 2)
    print(f"\n  final u_x range = [{uv[:,0].min():.3e}, {uv[:,0].max():.3e}]")
    print(f"  final u_y range = [{uv[:,1].min():.3e}, {uv[:,1].max():.3e}]")
    print(f"\n  Wrote {output_path} ({n_steps}-step time series of u + projected P)")
    return history


def _report_recurrence(history, n_ramp, n_hold):
    """Turn the per-increment record into an explicit pass/fail on the Q test."""
    print("\n  --- Q-recurrence check ---")
    if n_ramp >= 2:
        q_ramp = [h[6] for h in history if h[1] == "ramp"]
        grew = all(b > a for a, b in zip(q_ramp, q_ramp[1:]))
        print(f"  ramp: max|Q| {q_ramp[0]:.3e} -> {q_ramp[-1]:.3e} "
              f"({'monotone build-up' if grew else 'NON-monotone (inspect)'})")
    else:
        print("  ramp: <2 increments, recurrence memory term not exercised")

    if n_hold >= 2:
        hold = [h for h in history if h[1] == "hold"]
        q_hold = [h[6] for h in hold]
        p_hold = [h[5] for h in hold]
        q_decayed = all(b < a for a, b in zip(q_hold, q_hold[1:]))
        p_relaxed = all(b < a for a, b in zip(p_hold, p_hold[1:]))
        print(f"  hold: max|Q| {q_hold[0]:.3e} -> {q_hold[-1]:.3e} "
              f"({'decays' if q_decayed else 'does NOT decay'})")
        print(f"  hold: max|P| {p_hold[0]:.3e} -> {p_hold[-1]:.3e} "
              f"({'relaxes' if p_relaxed else 'does NOT relax'})")
        ok = q_decayed and p_relaxed
        print(f"  RESULT: stress relaxation at fixed strain "
              f"{'OBSERVED — recurrence carries state' if ok else 'NOT observed (FAIL)'}")
        # per-branch decay vs the closed-form exp(-dt/tau) expectation
        last_branch = hold[-1][7]
        first_branch = hold[0][7]
        print("  per-branch max|Q| (hold start -> end):")
        for i, (a, b) in enumerate(zip(first_branch, last_branch)):
            print(f"    branch {i}: {a:.3e} -> {b:.3e}")
    else:
        print("  hold: <2 increments, relaxation not exercised")


def _report_prony(history, prony_log, rep_qp):
    """Log the per-increment sub-ANN Prony outputs (tau_n, g_visc_n) at one QP.

    These are recomputed from the trained Tau_subANN / G_subANN at each step
    against the current invariants, so they track the deformation during the ramp
    and go flat during the hold, where frozen strain means frozen invariants and
    identical sub-ANN inputs.
    """
    cols = " ".join(f"b{i:>8d}" for i in range(N_MAXWELL))
    print(f"\n  --- sub-ANN Prony outputs at representative QP {rep_qp} ---")

    print(f"\n  tau_n [s] per Maxwell branch (relaxation times):")
    print(f"  {'k':>3} {'phase':>5}  {cols}")
    for (k, phase, *_), pr in zip(history, prony_log):
        vals = " ".join(f"{v:9.4e}" for v in pr[0][rep_qp])
        print(f"  {k:3d} {phase:>5}  {vals}")

    print(f"\n  g_visc_n per Maxwell branch (viscous coefficients):")
    print(f"  {'k':>3} {'phase':>5}  {cols}")
    for (k, phase, *_), pr in zip(history, prony_log):
        vals = " ".join(f"{v:9.4e}" for v in pr[1][rep_qp])
        print(f"  {k:3d} {phase:>5}  {vals}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Option A vCANN external-operator solve with viscoelastic "
                    "load-stepping (ramp + constant-strain hold)."
    )
    parser.add_argument("--u-top", type=float, default=U_TOP,
                        help=f"final top-edge y-displacement (default {U_TOP})")
    parser.add_argument("--n-cells", type=int, default=N_CELLS,
                        help=f"N for the NxN triangle mesh (default {N_CELLS})")
    parser.add_argument("--n-ramp", type=int, default=N_RAMP,
                        help=f"ramp increments 0->u_top (default {N_RAMP})")
    parser.add_argument("--n-hold", type=int, default=N_HOLD,
                        help=f"constant-strain hold increments (default {N_HOLD})")
    parser.add_argument("--dt", type=float, default=DT,
                        help=f"physical time per increment (default {DT})")
    parser.add_argument("--snes-monitor", action="store_true",
                        help="print the per-iteration SNES residual history")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="path to a Gmsh .msh file (overrides the in-memory "
                             "rectangle; must define physical groups 'bottom', "
                             "'top', 'domain'). Example: "
                             "fenicsx/meshes/unitsquare_hole.msh")
    args = parser.parse_args()

    # Override the module-level constants build_problem()/solve_load_steps() read.
    U_TOP = args.u_top
    N_CELLS = args.n_cells
    solve_load_steps(n_ramp=args.n_ramp, n_hold=args.n_hold, dt=args.dt,
                     snes_monitor=args.snes_monitor, mesh_path=args.mesh)
