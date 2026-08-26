# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Mesh-refinement convergence study for the external-operator solve.

Two quantities of interest: the global stored elastic energy
``E(h) = ∫ ψ_e(F(u)) dx``, and the local stress magnitude ``|P|`` at a fixed
point on the largest hole, where the concentration makes convergence slower.
No analytical solution exists, so the reference is a 3-level Richardson
extrapolation over a constant-ratio mesh family.

Run with the bridge environment::

    .venv-bridge/bin/python fenicsx/models/convergence_study.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import (
    bb_tree,
    compute_collisions_points,
    compute_colliding_cells,
)

from fenicsx.meshes.unitsquare_hole import build_mesh
from fenicsx.bridge.vcann_local_step import VCANNLocalStep
from fenicsx.bridge.verify_ufl_energy import extract_psi_weights
from fenicsx.models.embedded2d_vcann_extop import build_problem, K_QUAD


# --- study parameters ------------------------------------------------------
# Constant-ratio (~sqrt(2)) family over the same geometry, coarse to fine.
# Richardson requires a roughly constant h-ratio.
MESH_SIZES = [0.06, 0.0424, 0.03, 0.0212, 0.015, 0.0106]
U_TOP = 0.05         # single load step (n_ramp=1): spatial study, fixed dt/load
N_RAMP = 1
DT = 1.0
# Monitor point for the local stress QoI: just outside the largest hole
# (centre (0.70, 0.45), r=0.125) at its horizontal "equator", where vertical
# tension concentrates the hoop stress. Offset 0.015 into the material so the
# point is inside the faceted domain at every resolution.
MONITOR_PT = np.array([[0.84, 0.45, 0.0]])

_MESH_DIR = _PROJECT_ROOT / "fenicsx" / "meshes" / "_convergence"
_RESULTS_DIR = _PROJECT_ROOT / "fenicsx" / "results"


def plot_mesh(domain, h, out_path):
    """Save a matplotlib wireframe of the mesh.

    The mesh is P2, but a wireframe needs only the corner vertices. The geometry
    dofmap lists vertices before edge midpoints in the basix ordering, so the
    first 3 nodes of each triangle cell are its corners.
    """
    import matplotlib
    matplotlib.use("Agg")               # headless: write a file, never block
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    x = domain.geometry.x
    cells = domain.geometry.dofmap[:, :3]
    triang = Triangulation(x[:, 0], x[:, 1], cells)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(triang, color="0.25", lw=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"h = {h:.4f}   ({cells.shape[0]} cells, {x.shape[0]} nodes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    mesh plot -> {out_path}", flush=True)


def strain_energy_form(u, W1, b1, W2, alpha, beta, psi_ref):
    """∫ ψ_e dx — the verified VHB4910 elastic energy density, plane-strain
    incompressible embedding (F33 = 1/det F2). Mirrors verify_ufl_energy."""
    F2 = ufl.Identity(2) + ufl.grad(u)
    det2 = F2[0, 0] * F2[1, 1] - F2[0, 1] * F2[1, 0]
    F = ufl.as_matrix([
        [F2[0, 0], F2[0, 1], 0.0],
        [F2[1, 0], F2[1, 1], 0.0],
        [0.0,      0.0,      1.0 / det2],
    ])
    C = F.T * F
    C_bar = ufl.det(C) ** (-1.0 / 3.0) * C
    I_inv = ufl.tr(C_bar) / 3.0
    J_inv = ufl.tr(ufl.inv(C_bar)) / 3.0
    softplus = lambda x: ufl.ln(1.0 + ufl.exp(x))
    psi_raw = sum(
        float(W2[k]) * softplus(float(W1[0, k]) * I_inv
                                + float(W1[1, k]) * J_inv + float(b1[k]))
        for k in range(W2.shape[0])
    )
    psi = (psi_raw - float(psi_ref)
           + float(alpha) * (I_inv - 1.0) + float(beta) * (J_inv - 1.0))
    return psi * ufl.dx


def solve_state(mesh_path):
    """Build and solve to the final load state on one mesh, returning the pieces
    needed for the QoIs. Mirrors solve_load_steps' seed and commit, minus XDMF."""
    domain, u, problem, top_disp, Q_P = build_problem(dt=DT, mesh_path=str(mesh_path))
    prev = 0.0
    for k in range(1, N_RAMP + 1):
        tgt = U_TOP * k / N_RAMP
        top_disp.value = PETSc.ScalarType(tgt)
        # smooth, BC-satisfying seed so the Dirichlet lifting term vanishes
        if prev == 0.0:
            u.interpolate(lambda xx: np.vstack((np.zeros_like(xx[1]), tgt * xx[1])))
        elif tgt != prev:
            u.x.array[:] *= (tgt / prev)
        fem.set_bc(u.x.petsc_vec, problem._bcs_list)
        u.x.scatter_forward()
        prev = tgt
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError(
                f"mesh {mesh_path.name}: increment {k} did not converge "
                f"(reason={problem.solver.getConvergedReason()})"
            )
        problem.commit_state()
    return domain, u, problem, Q_P


def point_stress_magnitude(domain, problem, Q_P):
    """|P| (Frobenius of the in-plane PK1) at MONITOR_PT.

    The QP stress is recovered in DG2 before the point evaluation. Lagrange-1,
    as used by project_qf_to_nodes, smears a steep stress concentration onto
    linear nodes and jitters the point value across meshes, which pollutes the
    local Richardson rate; the local per-element L2 projection onto DG2 resolves
    the concentration faithfully."""
    P_qf = fem.Function(Q_P)
    P_qf.x.array[:] = problem.operator.last_P.reshape(-1)
    P_qf.x.scatter_forward()

    V_dg = fem.functionspace(domain, ("DG", 2, (2, 2)))
    w = ufl.TestFunction(V_dg)
    tr = ufl.TrialFunction(V_dg)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": K_QUAD})
    proj = LinearProblem(
        ufl.inner(tr, w) * dx, ufl.inner(P_qf, w) * dx,
        petsc_options_prefix="conv_proj_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    P_dg = proj.solve()

    tree = bb_tree(domain, domain.topology.dim)
    cand = compute_collisions_points(tree, MONITOR_PT)
    coll = compute_colliding_cells(domain, cand, MONITOR_PT)
    cells = coll.links(0)
    if len(cells) == 0:
        raise RuntimeError(f"MONITOR_PT {MONITOR_PT[0]} not inside the mesh")
    val = P_dg.eval(MONITOR_PT, cells[:1])   # flattened 2x2
    return float(np.sqrt(np.sum(np.asarray(val) ** 2)))


def richardson(hs, qs):
    """3-level Richardson over consecutive triples (assumes ~constant h-ratio).
    Returns list of (h_fine, observed_order_p, extrapolated_q).

    With the error model ``q(h) = Q + C h^p`` on a triple h1 > h2 > h3 of
    constant ratio ``r = h1/h2 = h2/h3``::

        d12 = q1 - q2 = C h3^p r^p (r^p - 1)
        d23 = q2 - q3 = C h3^p     (r^p - 1)
        =>  p = ln(d12 / d23) / ln r        and     C h3^p = d23 / (r^p - 1)
        =>  Q = q3 - d23 / (r^p - 1)

    Note the minus sign: the extrapolated limit lies beyond the finest value, on
    the far side from the coarser ones.
    """
    out = []
    for i in range(len(qs) - 2):
        r = hs[i] / hs[i + 1]
        d12, d23 = qs[i] - qs[i + 1], qs[i + 1] - qs[i + 2]
        if d23 == 0.0 or d12 / d23 <= 0.0:
            out.append((hs[i + 2], float("nan"), qs[i + 2]))
            continue
        p = math.log(d12 / d23) / math.log(r)
        q_ext = qs[i + 2] - d23 / (r ** p - 1.0)
        out.append((hs[i + 2], p, q_ext))
    return out


def main():
    _MESH_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting closed-form energy weights from the driver ...", flush=True)
    driver = VCANNLocalStep()
    W1, b1, W2 = extract_psi_weights(driver)
    alpha, beta, psi_ref = driver.alpha, driver.beta, driver.psi_ref

    print(f"\nConvergence study: U_TOP={U_TOP}, n_ramp={N_RAMP}, dt={DT}, "
          f"monitor={MONITOR_PT[0,:2]}")
    print(f"\n  {'h':>8} {'ndofs':>8} {'n_qp':>7} "
          f"{'energy E':>16} {'|P|(pt)':>14}")
    hs, Es, Ss = [], [], []
    ndofs_all, nqp_all = [], []
    for h in MESH_SIZES:
        mesh_path = _MESH_DIR / f"mesh_h{h:.4f}.msh"
        build_mesh(output_path=mesh_path, mesh_size=h, element_order=2)
        domain, u, problem, Q_P = solve_state(mesh_path)
        plot_mesh(domain, h, _MESH_DIR / f"mesh_h{h:.4f}.png")

        E_form = fem.form(strain_energy_form(u, W1, b1, W2, alpha, beta, psi_ref))
        E = domain.comm.allreduce(fem.assemble_scalar(E_form), op=MPI.SUM)
        S = point_stress_magnitude(domain, problem, Q_P)
        ndofs = u.x.array.size
        n_qp = problem.operator.n_qp
        hs.append(h); Es.append(E); Ss.append(S)
        ndofs_all.append(ndofs); nqp_all.append(n_qp)
        print(f"  {h:8.4f} {ndofs:8d} {n_qp:7d} {E:16.10e} {S:14.6e}", flush=True)

    print("\n  --- Richardson extrapolation (observed order p) ---")
    print(f"\n  GLOBAL: energy E = ∫ψ dx")
    for h_f, p, q in richardson(hs, Es):
        print(f"    h={h_f:.4f}:  p = {p:6.3f}   E* = {q:.10e}")
    print(f"\n  LOCAL: |P| at {MONITOR_PT[0,:2]}")
    for h_f, p, q in richardson(hs, Ss):
        print(f"    h={h_f:.4f}:  p = {p:6.3f}   |P|* = {q:.6e}")
    print("\n  (expect global p near the optimal smooth rate; local p lower, "
          "limited by the hole stress concentration.)")

    _write_csvs(hs, ndofs_all, nqp_all, Es, Ss)


def _write_csvs(hs, ndofs_all, nqp_all, Es, Ss):
    """Per-mesh QoIs and the Richardson tables, for plotting.

    ``mesh_convergence.csv`` also carries the errors used in the log-log plot,
    ``|E(h) - E*|`` and ``||P|(h) - |P|*|``, where the reference is the
    Richardson-extrapolated value from the finest triple, the best estimate of
    the h->0 limit available without an analytical solution.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rich_E = richardson(hs, Es)
    rich_S = richardson(hs, Ss)
    E_star = rich_E[-1][2] if rich_E else float("nan")
    S_star = rich_S[-1][2] if rich_S else float("nan")

    path = _RESULTS_DIR / "mesh_convergence.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["h", "ndofs", "n_qp", "energy", "err_energy",
                    "P_point", "err_P_point"])
        for h, nd, nq, E, S in zip(hs, ndofs_all, nqp_all, Es, Ss):
            w.writerow([f"{h:.6g}", nd, nq, f"{E:.12e}", f"{abs(E - E_star):.12e}",
                        f"{S:.12e}", f"{abs(S - S_star):.12e}"])
    print(f"\n  wrote {path}  (E* = {E_star:.10e}, |P|* = {S_star:.6e})")

    path = _RESULTS_DIR / "mesh_convergence_richardson.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "h_fine", "p_observed", "extrapolated"])
        for h_f, pp, q in rich_E:
            w.writerow(["energy", f"{h_f:.6g}", f"{pp:.6f}", f"{q:.12e}"])
        for h_f, pp, q in rich_S:
            w.writerow(["P_point", f"{h_f:.6g}", f"{pp:.6f}", f"{q:.12e}"])
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
