# FEniCSx Prototype

This package contains the FEniCSx-side prototype work for the vCANN bridge.

## Layout

- `models/`: simulation entrypoints and constitutive helpers
- `meshes/`: mesh generators and `.msh` assets
- `bridge/`: the vCANN↔FEniCSx coupling code (`vcann_local_step.py`, the external operator, verification scripts, tests)
- `results/`: generated `.xdmf` and `.h5` outputs
- `paraview/`: ParaView launch and automation helpers
- `docs/`: troubleshooting notes and modeling notes

## Recommended Commands

Run everything here with `.venv-bridge` (see the root [`README.md`](../README.md)
for how to set it up — it inherits the system `dolfinx`/`gmsh` install and
adds TensorFlow on top):

```bash
.venv-bridge/bin/python -m fenicsx.meshes.unitsquare_hole
.venv-bridge/bin/python -m fenicsx.models.embedded2d_hyperelastic
.venv-bridge/bin/python -m fenicsx.models.embedded2d_vcann_extop
```

Do not use the repository `.venv` for these commands — that environment is
the vCANN training stack (`vcann/`) and has no `dolfinx`. Bare
`/usr/bin/python3` also won't work here: it has `dolfinx`/`gmsh` but not
TensorFlow, which the bridge (`fenicsx/bridge/vcann_local_step.py`) needs
at import time.

## ParaView

Open the newest result in Linux ParaView:

```bash
./fenicsx/paraview/open_in_paraview.sh
```

Open a specific result:

```bash
./fenicsx/paraview/open_in_paraview.sh fenicsx/results/embedded2d_hyperelastic_VM.xdmf
```

For Linux-side automation, use the installed ParaView CLI tools:

```bash
pvpython your_script.py
pvbatch your_script.py
```

## Current Prototype

The current model:

- solves a 2D in-plane displacement field on a perforated square mesh
- builds `F2 = I + grad(u)`
- embeds `F2` into a `3x3` incompressible deformation gradient
- evaluates a placeholder 3D hyperelastic energy
- writes displacement, first Piola stress, and von Mises stress to `results/`
