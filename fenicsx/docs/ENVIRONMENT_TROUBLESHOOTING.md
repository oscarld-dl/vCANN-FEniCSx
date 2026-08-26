# FEniCSx Environment Troubleshooting

## Problem Encountered

Running

```bash
python3 -m fenicsx.models.embedded2d_hyperelastic
```

failed with:

```text
ImportError: cannot import name 'AbstractFiniteElement' from 'ufl.finiteelement'
```

## What Was Checked

The active interpreter was:

```bash
python3 -c "import sys; print(sys.executable); print(sys.prefix)"
```

Result:

```text
<repo>/.venv/bin/python3
<repo>/.venv
```

The Python path also showed cross-contamination from another local Python tree:

```text
<repo>/.python310/...
```

Installed FEniCS-related packages in `.venv`:

```text
UFL            2017.1.0
fenics-ufl     2025.3.0
fenics-basix   0.10.0
fenics-ffcx    0.10.1.post0
mpi4py         4.1.1
petsc4py       3.25.0
```

`dolfinx` was **not** installed in that interpreter.

## Diagnosis

This is a mixed and incompatible environment:

- legacy `UFL==2017.1.0` is installed
- modern `fenics-ufl==2025.3.0` is also installed
- both provide the `ufl` import namespace
- Python is importing the wrong `ufl` implementation for the newer FEniCSx stack
- `dolfinx` itself is missing, so even fixing `ufl` alone would not be enough

## Recommended Fix

Do **not** continue using the current `.venv` for FEniCSx.

Use Ubuntu's system FEniCSx installation instead of the project virtualenv:

```bash
cd <repo-root>
/usr/bin/python3 -c "import sys, dolfinx, ufl; print(sys.executable); print(dolfinx.__version__); print(ufl.__file__)"
/usr/bin/python3 -m fenicsx.models.embedded2d_hyperelastic
```

Verified on April 24, 2026:

```text
/usr/bin/python3
0.10.0.post2
/usr/lib/python3/dist-packages/ufl/__init__.py
```

Ubuntu packages already installed on this machine include:

- `fenicsx`
- `python3-dolfinx`
- `python3-dolfinx-real`
- `python3-petsc4py`
- `python3-mpi4py`

This means a separate conda environment is **not required** for the current
machine, although it would still be a reasonable isolation strategy on a
different host.

## Current Status

Running the prototype with the correct interpreter moved the failure past the
environment issue and exposed a real code/API mismatch:

- DOLFINx `0.10.x` required `petsc_options_prefix` in `NonlinearProblem`
- `XDMFFile.write_function` required interpolation to degree-1 output space
- both issues were fixed in `fenicsx/models/embedded2d_hyperelastic.py`

Verified end-to-end on April 24, 2026:

```text
Converged in 4 Newton iterations.
u_x,max = 0.200000
u_y range = [-0.008754, 0.008852]
Wrote <repo>/fenicsx/results/embedded2d_hyperelastic_VM.xdmf
```

## Why This Is The Safer Path

- keeps TensorFlow/vCANN dependencies isolated from FEniCSx
- avoids the current `.venv` package collision
- uses the already-installed and internally consistent Ubuntu FEniCSx packages

## If You Insist On Repairing The Current `.venv`

This was the first cleanup step identified:

```bash
python3 -m pip uninstall -y UFL ufl
```

But this was **not** recommended, because:

- `dolfinx` is still missing there
- the environment is already contaminated
- it is easy to break the vCANN setup while trying to repair it

## Files Relevant To This Issue

- [embedded2d_hyperelastic.py](../models/embedded2d_hyperelastic.py)
- [requirements-bridge.txt](../../requirements-bridge.txt)
- [requirements.txt](../../requirements.txt)

## Resume Point

As of April 24, 2026:

1. Run FEniCSx code with `/usr/bin/python3`, not the project `.venv`.
2. Keep the `.venv` cleanup optional unless you specifically want to reduce confusion.
3. Use the generated `results/embedded2d_hyperelastic_VM.xdmf` for ParaView or downstream checks.
4. Extend the prototype toward the vCANN material update from this now-working baseline.
