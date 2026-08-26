# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Small DOLFINx prototype for a 2D mesh with 3D embedded kinematics."""

from pathlib import Path
import sys

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FENICSX_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_VENV = (_PROJECT_ROOT / ".venv").resolve()
_MESH_PATH = _FENICSX_ROOT / "meshes" / "unitsquare_hole.msh"
_RESULTS_DIR = _FENICSX_ROOT / "results"

if Path(sys.prefix).resolve() == _PROJECT_VENV:
    raise SystemExit(
        "Run this prototype with Ubuntu's system Python, not the project .venv.\n"
        "Use: /usr/bin/python3 -m fenicsx.models.embedded2d_hyperelastic"
    )

import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile, gmsh

try:
    from .kinematics import embed_incompressible_2d_in_3d, plane_deformation_gradient
    from .materials import first_piola_stress_from_energy, neo_hookean_embedded_energy
except ImportError:
    from kinematics import embed_incompressible_2d_in_3d, plane_deformation_gradient
    from materials import first_piola_stress_from_energy, neo_hookean_embedded_energy


def build_problem():
    """Create a small uniaxial-extension problem on a 2D rectangle with holes."""
    mesh_data = gmsh.read_from_msh(_MESH_PATH, MPI.COMM_WORLD, 0, gdim=2)

    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    physical = mesh_data.physical_groups

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    fdim = domain.topology.dim - 1

    bottom_facets = facet_tags.find(physical["bottom"].tag)
    top_facets = facet_tags.find(physical["top"].tag)

    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    top_dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)

    u_zero = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    top_displacement = fem.Constant(domain, PETSc.ScalarType(0.0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, V)
    bc_top_y = fem.dirichletbc(top_displacement, top_dofs_y, V.sub(1))
    bcs = [bc_bottom, bc_top_y]

    F2 = plane_deformation_gradient(u)
    F3 = embed_incompressible_2d_in_3d(F2)
    J = ufl.det(F3)

    mu = fem.Constant(domain, PETSc.ScalarType(1.0))
    psi = neo_hookean_embedded_energy(F3, mu)
    P = first_piola_stress_from_energy(psi, F3)
    sigma = (1.0 / J) * P * F3.T
    s = sigma - (1.0 / 3.0) * ufl.tr(sigma) * ufl.Identity(3)
    sigma_vm = ufl.sqrt(1.5 * ufl.inner(s, s))

    # Implicit weak form through the energy: residual and jacobian for Newton's method
    # Check the Modern Simulation Development notebook in Inkodo for more details on the Newton Solver and the variational forms.

    total_potential = psi * dx(physical["domain"].tag) # here we integrate the energy over the whole domain. 
    residual = ufl.derivative(total_potential, u, v) # the residual is the first variation of the total potential energy with respect to the displacement field u, tested against v.
    jacobian = ufl.derivative(residual, u, du) # the jacobian is the second variation of the total potential energy, which is the derivative of the residual with respect to u, in the direction of du.


    return domain, u, P, sigma_vm, residual, jacobian, bcs, top_displacement


def solve_problem():
    """Solve the prototype incrementally and write a load-step series to XDMF."""
    target_vertical_displacement = 1.0
    num_load_steps = 100

    domain, u, P, sigma_vm, residual, jacobian, bcs, top_displacement = build_problem()

    problem = NonlinearProblem(
        residual,
        u,
        petsc_options_prefix="embedded2d_",
        bcs=bcs,
        J=jacobian,
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RESULTS_DIR / "embedded2d_hyperelastic_VM.xdmf"
    V_out = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u_out = fem.Function(V_out, name="u")
    P_out_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
    P_out = fem.Function(P_out_space, name="P")
    P_expr = fem.Expression(P, P_out_space.element.interpolation_points)
    sigma_vm_out_space = fem.functionspace(domain, ("DG", 0))
    sigma_vm_out = fem.Function(sigma_vm_out_space, name="sigma_vm")
    sigma_vm_expr = fem.Expression(sigma_vm, sigma_vm_out_space.element.interpolation_points)
    load_values = np.round(
        np.linspace(0.0, target_vertical_displacement, num_load_steps + 1), 6
    )

    with XDMFFile(domain.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for step, load_value in enumerate(load_values):
            top_displacement.value = PETSc.ScalarType(load_value)

            problem.solve()
            u.x.scatter_forward()

            iterations = problem.solver.getIterationNumber()
            converged_reason = problem.solver.getConvergedReason()
            converged = converged_reason > 0

            if not converged:
                raise RuntimeError(
                    "Newton solver did not converge at "
                    f"load step {step}/{num_load_steps} "
                    f"(u_top={load_value:.6f}) after {iterations} iterations. "
                    f"Reason={converged_reason}."
                )

            u_out.interpolate(u)
            P_out.interpolate(P_expr)
            sigma_vm_out.interpolate(sigma_vm_expr)
            xdmf.write_function(u_out, float(load_value))
            xdmf.write_function(P_out, float(load_value))
            xdmf.write_function(sigma_vm_out, float(load_value))

            if domain.comm.rank == 0:
                dof_values = u.x.array.reshape(-1, 2)
                ux_max = float(np.max(dof_values[:, 0]))
                uy_min = float(np.min(dof_values[:, 1]))
                uy_max = float(np.max(dof_values[:, 1]))
                print(
                    f"Step {step:02d}/{num_load_steps}: "
                    f"u_top={load_value:.6f}, iterations={iterations}, "
                    f"u_x,max={ux_max:.6f}, "
                    f"u_y range=[{uy_min:.6f}, {uy_max:.6f}]"
                )

    if domain.comm.rank == 0:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    solve_problem()
