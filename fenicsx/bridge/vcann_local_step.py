# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

import sys
from pathlib import Path 
from dataclasses import dataclass 

import numpy as np 
import tensorflow as tf 

tf.keras.backend.set_floatx('float64') # Training and checkpoint weights use float64.
TF_FLOAT = "float64"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent           
CODE_DIR = PROJECT_ROOT / "vcann"
sys.path.insert(0, str(CODE_DIR))

import subANNs  # noqa: E402

# Checkpoint from the saved VHB4910 run.

CKPT_DIR = PROJECT_ROOT / "vcann" / "Results_VHB4910" / "20260422-191716" / "ckpt"
CKPT_PREFIX = str(CKPT_DIR / "ckpt") 


# Configuration of the trained VHB4910 model (must match vCANN_main.py)
N_MAXWELL = 5
NUM_TENS = 1
NUM_DIR = 0
NUM_EXTRA = 0

LAYER_SIZE_PSI = [8]
ACTIVATIONS_PSI = ["softplus"]

LAYER_SIZE_TAU = [4]        # one MLP per Maxwell branch
ACTIVATIONS_TAU = ["softplus"]

LAYER_SIZE_G = [4]          # one per branch, plus one for the equilibrium spring
ACTIVATIONS_G = ["softplus"]

# tau_i lives in [10^T_MIN, 10^T_MAX] seconds
T_MIN, T_MAX = -2, 1

# Logarithmic branch scaling used by the original ScaleLayer.
TAU_SCALE = tf.constant(
    np.logspace(T_MIN, T_MAX, num=N_MAXWELL, base=10.0, dtype=np.float64),
    dtype=TF_FLOAT,
)


# Submodel builders 
def _build_psi_submodel() -> tf.keras.Model:
    """Reconstruct the strain-energy submodel as build.py would."""
    inputs = tf.keras.Input(shape=(None, 2), name="Invars_in", dtype=TF_FLOAT)
    outputs = subANNs.Psi_subANN(inputs, LAYER_SIZE_PSI, ACTIVATIONS_PSI, suffix="1")
    return tf.keras.Model(inputs, outputs, name="model_Psi")


def _build_tau_submodel() -> tf.keras.Model:
    """Reconstruct the relaxation-time submodel.

    The ScaleLayer multiplier carries no trainable weights and is applied
    later in step(), so this submodel mirrors the checkpoint variables 1:1.
    """
    inputs = tf.keras.Input(shape=(None, 2), name="Invars_in", dtype=TF_FLOAT)
    branches = [
        subANNs.Tau_subANN(inputs, LAYER_SIZE_TAU, ACTIVATIONS_TAU, suffix=f"1_{jj+1}")
        for jj in range(N_MAXWELL)
    ]
    outputs = tf.keras.layers.concatenate(branches, axis=-1, name="concat_tau_0")
    return tf.keras.Model(inputs, outputs, name="model_tau_0")


def _build_g_submodel() -> tf.keras.Model:
    """Reconstruct the relaxation-coefficient submodel.

    g[..., 0] is the equilibrium coefficient g_inf; g[..., 1:] are the
    Maxwell-branch coefficients. The sum-to-1 normalization is applied
    later in step(), so the submodel mirrors the checkpoint exactly.
    """
    inputs = tf.keras.Input(shape=(None, 2), name="Invars_in", dtype=TF_FLOAT)
    branches = [
        subANNs.G_subANN(inputs, LAYER_SIZE_G, ACTIVATIONS_G, suffix=f"1_{jj}")
        for jj in range(N_MAXWELL + 1)
    ]
    g_concat = tf.keras.layers.concatenate(branches, axis=-1, name="concat_g_0")
    # Per-channel learned rescale. lambda_prony=0.0 in the saved run left this
    # layer untrainable, so its kernel is all-ones; the saved value is still
    # loaded in case a future run uses lambda_prony > 0.
    g_rescaled = tf.keras.layers.DepthwiseConv1D(
        kernel_size=1,
        depthwise_initializer="ones",
        use_bias=False,
        trainable=False,
        name="regularization_layer_g_0",
    )(g_concat)
    return tf.keras.Model(inputs, g_rescaled, name="model_g_0")


# The kinematics helpers below take (3, 3) tensors, with no batch/time dims.

def _C_from_F(F: tf.Tensor) -> tf.Tensor:
    """Right Cauchy-Green tensor: C = F^T F."""
    return tf.linalg.matmul(F, F, transpose_a=True)


def _C_bar_from_C(C: tf.Tensor) -> tf.Tensor:
    """Isochoric right Cauchy-Green: C_bar = det(C)^(-1/3) * C."""
    detC = tf.linalg.det(C)
    return tf.pow(detC, -1.0 / 3.0) * C


def _invariants_from_C_bar(C_bar: tf.Tensor) -> tf.Tensor:
    """
    Generalized invariants for the isotropic single-tensor case (H = I/3):
        I = tr(C_bar @ I/3)        = tr(C_bar) / 3
        J = tr(C_bar^{-T} @ I/3)   = tr(C_bar^{-1}) / 3
    Returns a (1, 1, 2) tensor laid out as the MLPs expect.
    """
    I_inv = tf.linalg.trace(C_bar) / 3.0
    J_inv = tf.linalg.trace(tf.linalg.inv(C_bar)) / 3.0
    invs = tf.stack([I_inv, J_inv])
    return tf.reshape(invs, (1, 1, 2))


def _compute_reference_constants(model_psi: tf.keras.Model):
    """
    Evaluate the energy and its (I, J) gradients at the reference state
    F = I (so I = J = 1) and return the scalars used by the stress-free
    reference correction (Linden et al. 2023, build.py:192-199):

        delta = dPsi/dI - dPsi/dJ   (at the reference)
        alpha = ReLU(-delta)
        beta  = ReLU( delta)
        Psi_ref = Psi(1, 1)

    Constants for this config (isotropic, no extra inputs).
    """
    invars = tf.constant([[[1.0, 1.0]]], dtype=TF_FLOAT)
    with tf.GradientTape() as tape:
        tape.watch(invars)
        psi = model_psi(invars)
    grad = tape.gradient(psi, invars)

    psi_ref = float(psi.numpy().squeeze())
    dPsi_dI = float(grad[0, 0, 0].numpy())
    dPsi_dJ = float(grad[0, 0, 1].numpy())
    delta = dPsi_dI - dPsi_dJ
    alpha = max(-delta, 0.0)
    beta = max(delta, 0.0)

    return psi_ref, alpha, beta


@dataclass
class VCANNState:
    """
    Constitutive state carried at one quadrature point between time steps.

    All arrays are float64, with no batch/time leading dims. The fields are
    the previous-step values needed by the Prony midpoint recurrence:

      Q       (M, 3, 3)  per-Maxwell-branch overstresses
      S_e     (3, 3)     deviatoric elastic 2nd PK at the last step
      tau     (M,)       physical relaxation times at the last step
      g_visc  (M,)       normalized viscous coefficients at the last step
      t       float      absolute time at the last step
    """
    Q: np.ndarray
    S_e: np.ndarray
    tau: np.ndarray
    g_visc: np.ndarray
    t: float


class VCANNLocalStep:
    """
    Stateful one-step constitutive update for the VHB4910 vCANN.

    Built once at FE startup (loads the three sub-MLPs from the saved
    checkpoint and pre-computes the Linden et al. reference constants),
    then called once per quadrature point per Newton iteration.

    dPsi/dC is evaluated eagerly via tf.GradientTape; the rest of the
    per-step math is plain NumPy.
    """

    def __init__(self):
        self.model_psi, self.model_tau, self.model_g = build_models()
        self.psi_ref, self.alpha, self.beta = _compute_reference_constants(self.model_psi)

    def _prony_at(self, invars: tf.Tensor):
        """
        Evaluate the Prony MLPs at the (1, 1, 2) invariants tensor.

        Returns
        -------
        g_inf : float
            Normalized equilibrium coefficient.
        g_visc : np.ndarray of shape (M,)
            Normalized viscous coefficients (one per Maxwell branch).
        tau_phys : np.ndarray of shape (M,)
            Physical relaxation times (post-ScaleLayer).
        """
        tau_raw = self.model_tau(invars).numpy().ravel()
        g_raw = self.model_g(invars).numpy().ravel()
        g_norm = g_raw / g_raw.sum()
        g_inf = float(g_norm[0])
        g_visc = g_norm[1:].astype(np.float64)
        tau_phys = (tau_raw * TAU_SCALE.numpy()).astype(np.float64)
        return g_inf, g_visc, tau_phys

    def initial_state(self, t0: float = 0.0) -> VCANNState:
        """
        Initial state at F = I, t = t0. Previous-step Prony parameters are
        evaluated at the reference invariants (I, J) = (1, 1), so the first
        step has a sensible previous step to average against.
        """
        invars = tf.constant([[[1.0, 1.0]]], dtype=TF_FLOAT)
        _, g_visc_0, tau_0 = self._prony_at(invars)
        return VCANNState(
            Q=np.zeros((N_MAXWELL, 3, 3), dtype=np.float64),
            S_e=np.zeros((3, 3), dtype=np.float64),
            tau=tau_0,
            g_visc=g_visc_0,
            t=float(t0),
        )

    def step(self, F_n: np.ndarray, dt: float, state: VCANNState):
        """
        Advance one time step at one quadrature point.

        Parameters
        ----------
        F_n   : np.ndarray of shape (3, 3)  - current deformation gradient
        dt    : float                       - time increment t_n - t_{n-1}
        state : VCANNState                  - committed state at step n-1

        Returns
        -------
        P_n       : np.ndarray of shape (3, 3)  - first Piola-Kirchhoff stress
        state_new : VCANNState                  - state to commit at step n
        """
        F_n_tf = tf.constant(F_n, dtype=TF_FLOAT)

        C_n = _C_from_F(F_n_tf)
        with tf.GradientTape() as tape:
            tape.watch(C_n)
            C_bar_n = _C_bar_from_C(C_n)
            invars = _invariants_from_C_bar(C_bar_n)
            I_n = invars[0, 0, 0]
            J_n = invars[0, 0, 1]
            psi_NN = self.model_psi(invars)[0, 0, 0]
            # Total energy with stress-free reference correction.
            psi_total = (
                psi_NN
                - self.psi_ref
                + self.alpha * (I_n - 1.0)
                + self.beta * (J_n - 1.0)
            )
        # S_e = 2 dPsi/dC
        S_e_dev = (2.0 * tape.gradient(psi_total, C_n)).numpy()

        # Prony parameters at the current step
        g_inf_n, g_visc_n, tau_n = self._prony_at(invars)

        # Maxwell midpoint update
        tau_bar = 0.5 * (tau_n + state.tau)
        g_bar = 0.5 * (g_visc_n + state.g_visc)
        decay = np.exp(-dt / tau_bar)
        drive = np.exp(-dt / (2.0 * tau_bar)) * g_bar
        delta_S_e = S_e_dev - state.S_e
        Q_n = (
            decay[:, None, None] * state.Q
            + drive[:, None, None] * delta_S_e[None, :, :]
        )

        # Total deviatoric viscoelastic 2nd PK
        S_dev = g_inf_n * S_e_dev + Q_n.sum(axis=0)

        # Incompressibility closure: enforce S[2, 2] = 0
        C_np = C_n.numpy()
        C_inv = np.linalg.inv(C_np)
        p = S_dev[2, 2] / C_inv[2, 2]
        S = S_dev - p * C_inv

        # First Piola-Kirchhoff stress
        P = F_n @ S

        new_state = VCANNState(
            Q=Q_n,
            S_e=S_e_dev,
            tau=tau_n,
            g_visc=g_visc_n,
            t=state.t + dt,
        )
        return P, new_state


def _load_submodel_weights(model: tf.keras.Model, root_idx: int, reader) -> None:
    """
    Restore variables from 'layer_with_weights-{root_idx}/...' in the
    checkpoint into the weight-bearing layers of `model`, in declaration
    order.

    Assumes the rebuilt submodel's weighted layers appear in the same order
    as at training time, which holds as long as the builders above mirror
    build.py's construction order.
    """
    weighted_layers = [layer for layer in model.layers if len(layer.weights) > 0]
    for j, layer in enumerate(weighted_layers):
        layer_prefix = f"layer_with_weights-{root_idx}/layer_with_weights-{j}"
        new_weights = []
        for var in layer.weights:
            short_name = var.name.split("/")[-1].split(":")[0]
            # Keras 3 renamed DepthwiseConv1D's weight to "kernel";
            # the checkpoint was written by Keras 2 as "depthwise_kernel".
            if isinstance(layer, tf.keras.layers.DepthwiseConv1D) and short_name == "kernel":
                short_name = "depthwise_kernel"
            ckpt_name = f"{layer_prefix}/{short_name}/.ATTRIBUTES/VARIABLE_VALUE"
            new_weights.append(reader.get_tensor(ckpt_name))
        layer.set_weights(new_weights)


def build_models():
    """
    Build the three submodels and load their weights from the saved
    VHB4910 checkpoint.

    Returns
    -------
    (model_Psi, model_tau_0, model_g_0)
    """
    model_psi = _build_psi_submodel()
    model_tau = _build_tau_submodel()
    model_g = _build_g_submodel()

    reader = tf.train.load_checkpoint(CKPT_PREFIX)
    _load_submodel_weights(model_psi, root_idx=0, reader=reader)
    _load_submodel_weights(model_tau, root_idx=1, reader=reader)
    _load_submodel_weights(model_g,   root_idx=2, reader=reader)

    return model_psi, model_tau, model_g


def _smoke_test_loaded() -> None:
    """
    Sanity check after weight loading: forward (I, J) = (1, 1) through each
    submodel. No output should be NaN or zero, except possibly Psi at the
    reference.
    """
    model_psi, model_tau, model_g = build_models()

    # All-ones expected: lambda_prony=0 left this layer untrainable.
    dw_kernel = model_g.get_layer("regularization_layer_g_0").get_weights()[0]
    max_dev = np.max(np.abs(dw_kernel - 1.0))
    print(f"depthwise kernel shape={dw_kernel.shape}, |kernel-1|_inf = {max_dev:.3e}")

    # Forward at the reference: F = I  =>  I = J = 1.
    invars = tf.constant([[[1.0, 1.0]]], dtype=TF_FLOAT)
    psi_val = model_psi(invars).numpy().squeeze()
    tau_val = model_tau(invars).numpy().squeeze()
    g_val = model_g(invars).numpy().squeeze()

    print(f"Psi(1,1) = {psi_val:.6e}")
    print(f"tau (raw, pre-ScaleLayer) = {tau_val}")
    print(f"g (rescaled, pre-normalization) = {g_val}")
    print(f"  sum(g) = {g_val.sum():.6f}  (will be normalized to 1 in step)")

    # Reference-config constants
    psi_ref, alpha, beta = _compute_reference_constants(model_psi)
    print()
    print(f"Psi_ref = {psi_ref:.6e}   (must equal Psi(1,1) above)")
    print(f"alpha = {alpha:.6e}")
    print(f"beta  = {beta:.6e}")
    print(f"  (exactly one of alpha, beta is zero by construction)")


def _verify_uniaxial_history(n_steps: int = 600, lambda_max: float = 1.2) -> None:
    """
    Drive the local step over a uniaxial loading ramp, print the axial stress
    at a few trajectory points, and save a PNG of P[0,0] against lambda.
    """
    from histories import make_uniaxial_history
    import matplotlib.pyplot as plt

    print(f"---- {n_steps}-step uniaxial verification ----")
    F_history, t_history = make_uniaxial_history(n_steps=n_steps, lambda_max=lambda_max)

    driver = VCANNLocalStep()
    state = driver.initial_state(t0=0.0)

    P_history = np.zeros((n_steps, 3, 3), dtype=np.float64)
    prev_t = 0.0
    for n in range(n_steps):
        dt = float(t_history[n] - prev_t)
        P_n, state = driver.step(F_history[n], dt=dt, state=state)
        P_history[n] = P_n
        prev_t = float(t_history[n])

    P_axial = P_history[:, 0, 0]
    lam_axial = F_history[:, 0, 0]

    quartiles = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    print(f"  {'step':>4}  {'t':>8}  {'lambda':>8}  {'P[0,0]':>14}")
    for n in quartiles:
        print(f"  {n:>4d}  {t_history[n]:>8.4f}  {lam_axial[n]:>8.4f}  {P_axial[n]:>14.6e}")

    print(f"\n  P[0,0] range: {P_axial.min():.3e}  to  {P_axial.max():.3e}")
    diffs = np.diff(P_axial)
    print(f"  monotonically non-decreasing? {bool(np.all(diffs >= -1e-9))}")
    if not bool(np.all(diffs >= -1e-9)):
        bad = np.argmin(diffs)
        print(f"    first decrease at step {bad}: P[0,0] went from "
              f"{P_axial[bad]:.6e} to {P_axial[bad + 1]:.6e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lam_axial, P_axial, lw=1.6)
    ax.set_xlabel(r"$\lambda$ (axial stretch)")
    ax.set_ylabel(r"$P_{11}$  (presumed kPa)")
    ax.set_title(f"vCANN local-step, uniaxial ramp ({n_steps} steps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = HERE / "verify_uniaxial.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot to {out_path}")


def _smoke_test_step() -> None:
    """
    Take a couple of steps from the initial state and check that the stresses
    are sensible.
    """
    print("---- step smoke test ----")
    driver = VCANNLocalStep()
    state = driver.initial_state(t0=0.0)

    # At the reference state the stress should vanish to FP precision.
    F_id = np.eye(3, dtype=np.float64)
    P_id, state = driver.step(F_id, dt=1.0, state=state)
    print(f"Step at F = I:  max|P| = {np.max(np.abs(P_id)):.3e}   (should be ~0)")

    # Small incompressible uniaxial stretch: diag(lam, 1/sqrt(lam), 1/sqrt(lam))
    # preserves det F = 1.
    state = driver.initial_state(t0=0.0)   # restart so this step is independent
    lam = 1.1
    F_uni = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)]).astype(np.float64)
    P_uni, state = driver.step(F_uni, dt=0.1, state=state)
    print(f"Step at F = diag({lam}, 1/sqrt({lam}), 1/sqrt({lam})):")
    print(f"  P =\n{P_uni}")
    print(f"  P[0, 0] (axial) = {P_uni[0, 0]:.6e}")
    off_diag_max = np.max(np.abs(P_uni - np.diag(np.diag(P_uni))))
    print(f"  max off-diagonal |P| = {off_diag_max:.3e}   (should be ~0 in principal frame)")


def list_checkpoint_variables():
    variables = tf.train.list_variables(CKPT_PREFIX)
    for name, shape in variables:
        print(f"{name}: {shape}")


if __name__ == "__main__":
  print("---- weight-loading smoke test ----")
  _smoke_test_loaded()
  print()
  _smoke_test_step()
  print()
  _verify_uniaxial_history(n_steps=600, lambda_max=1.2)