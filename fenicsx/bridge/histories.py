# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

import numpy as np

def make_uniaxial_history(n_steps=100, lambda_max=1.2):
    """Generate an incompressible uniaxial deformation history over 1 s."""
    time = np.linspace(0.0, 1.0, n_steps)
    lambdas = np.linspace(1.0, lambda_max, n_steps)

    F_history = np.zeros((n_steps, 3, 3))

    for i, lam in enumerate(lambdas):
        F_history[i] = np.eye(3)
        F_history[i, 0, 0] = lam
        F_history[i, 1, 1] = 1.0 / np.sqrt(lam)
        F_history[i, 2, 2] = 1.0 / np.sqrt(lam)
    return F_history, time
