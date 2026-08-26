# Upstream provenance

The code in this directory is the vCANN (Viscoelastic Constitutive Artificial
Neural Network) reference implementation from:

https://github.com/ConstitutiveANN/vCANN

Vendored unmodified (a single trailing-newline diff in `vCANN_main.py` aside)
under the upstream MIT license in `LICENSE` — the exact text from the
upstream repository, copyright Kian Abdolazizi.

## Citation

If you use this code, cite the associated paper:

```bibtex
@article{Abdolazizi2024,
  title   = {Viscoelastic constitutive artificial neural networks (vCANNs) -
             A framework for data-driven anisotropic nonlinear finite
             viscoelasticity},
  journal = {Journal of Computational Physics},
  volume  = {499},
  pages   = {112704},
  year    = {2024},
  doi     = {https://doi.org/10.1016/j.jcp.2023.112704},
  author  = {Kian P. Abdolazizi and Kevin Linka and Christian J. Cyron}
}
```

## Scope in this repo

This directory trains the vCANN constitutive model on VHB4910 uniaxial data
(see `../data/VHB4910/`). The rest of this repository (`fenicsx/`) is
original work that couples the trained model to a FEniCSx finite-element
solver and is not part of the upstream project.
