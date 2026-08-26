# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Kinematics helpers for 2D finite elements embedded in 3D."""

import ufl


def plane_deformation_gradient(u):
    """Return the in-plane deformation gradient F2 = I + grad(u)."""
    return ufl.variable(ufl.Identity(2) + ufl.grad(u))


def embed_incompressible_2d_in_3d(F2):
    """
    Embed an in-plane 2x2 deformation gradient into a 3x3 tensor.

    The thickness stretch is chosen algebraically from incompressibility:
        lambda_3 = 1 / det(F2)
    """
    lam3 = 1.0 / ufl.det(F2)
    return ufl.variable(
        ufl.as_matrix(
            (
                (F2[0, 0], F2[0, 1], 0.0),
                (F2[1, 0], F2[1, 1], 0.0),
                (0.0, 0.0, lam3),
            )
        )
    )
