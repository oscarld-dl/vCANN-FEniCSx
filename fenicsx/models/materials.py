# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Placeholder constitutive laws for early DOLFINx tests."""

import ufl


def neo_hookean_embedded_energy(F3, mu):
    """
    Simple isochoric 3D neo-Hookean energy for embedded 2D tests.

    With the embedded incompressibility closure det(F3) = 1, so a minimal
    isochoric energy is enough for the first prototype.
    """
    C3 = F3.T * F3
    I1 = ufl.tr(C3)
    return 0.5 * mu * (I1 - 3.0)


def first_piola_stress_from_energy(psi, F):
    """Return the first Piola-Kirchhoff stress P = dpsi/dF."""
    return ufl.diff(psi, F)
