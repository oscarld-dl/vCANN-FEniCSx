# Oscar Ludeña Navarro
# DLR Institute of Lightweight Systems, September 2026

"""Generate a unit-square mesh with circular holes using Gmsh."""

from __future__ import annotations

from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_VENV = (_PROJECT_ROOT / ".venv").resolve()
_OUTPUT_PATH = Path(__file__).with_suffix(".msh")

_SQUARE_WIDTH = 1.0
_SQUARE_HEIGHT = 1.0
_DEFAULT_MESH_SIZE = 0.02
_HOLES = (
    (0.20, 0.20, 0.05),
    (0.35, 0.75, 0.1),
    (0.70, 0.45, 0.125)
)

_DOMAIN_PHYSICAL_TAG = 1
_BOTTOM_BOUNDARY_PHYSICAL_TAG = 2
_TOP_BOUNDARY_PHYSICAL_TAG = 3
_LEFT_BOUNDARY_PHYSICAL_TAG = 4
_RIGHT_BOUNDARY_PHYSICAL_TAG = 5
_HOLE_BOUNDARY_PHYSICAL_TAG = 6


def _ensure_supported_python() -> None:
    """Fail fast when the repository venv shadows Ubuntu's Gmsh install."""
    if Path(sys.prefix).resolve() == _PROJECT_VENV:
        raise SystemExit(
            "Run this mesh generator with Ubuntu's system Python, not the project .venv.\n"
            "Use: /usr/bin/python3 -m fenicsx.meshes.unitsquare_hole"
        )


def _import_gmsh():
    """Import Gmsh lazily so we can show a better interpreter hint first."""
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The gmsh Python module is not available in this interpreter.\n"
            "Use: /usr/bin/python3 -m fenicsx.meshes.unitsquare_hole"
        ) from exc

    return gmsh


def _classify_boundary_curves(gmsh, surface_tag: int) -> dict[str, list[int]]:
    """Split boundary curves into named square edges and hole boundaries."""
    edge_boundaries = {"bottom": [], "top": [], "left": [], "right": []}
    hole_boundaries: list[int] = []
    tolerance = 1.0e-8

    for dim, curve_tag in gmsh.model.getBoundary([(2, surface_tag)], oriented=False):
        if dim != 1:
            continue

        curve_type = gmsh.model.getType(dim, curve_tag)
        if curve_type == "Line":
            x_center, y_center, _ = gmsh.model.occ.getCenterOfMass(dim, curve_tag)
            if abs(y_center) <= tolerance:
                edge_boundaries["bottom"].append(curve_tag)
            elif abs(y_center - _SQUARE_HEIGHT) <= tolerance:
                edge_boundaries["top"].append(curve_tag)
            elif abs(x_center) <= tolerance:
                edge_boundaries["left"].append(curve_tag)
            elif abs(x_center - _SQUARE_WIDTH) <= tolerance:
                edge_boundaries["right"].append(curve_tag)
            else:
                raise RuntimeError(
                    "Failed to classify outer boundary curve "
                    f"{curve_tag} with center of mass ({x_center}, {y_center})."
                )
        else:
            hole_boundaries.append(curve_tag)

    for edge_name, edge_curves in edge_boundaries.items():
        if len(edge_curves) != 1:
            raise RuntimeError(
                f"Expected exactly one curve for the {edge_name} boundary, got {edge_curves}."
            )
    if not hole_boundaries:
        raise RuntimeError("Failed to identify any hole boundaries.")

    edge_boundaries["hole_boundary"] = hole_boundaries
    return edge_boundaries


def build_mesh(
    output_path: Path = _OUTPUT_PATH,
    mesh_size: float = _DEFAULT_MESH_SIZE,
    element_order: int = 1,
) -> Path:
    """Build and write the mesh next to this script.

    ``element_order=2`` produces curved (isoparametric) second-order triangles
    whose boundary nodes are projected onto the OCC geometry, so the circular
    holes are represented to 2nd order instead of faceted with straight chords.
    This removes the leading geometry error on the curved boundaries — important
    for a convergence study, where re-faceting at each ``mesh_size`` otherwise
    perturbs the domain and pollutes especially the local (near-hole) quantities.
    """
    if mesh_size <= 0.0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}.")
    if element_order not in (1, 2):
        raise ValueError(f"element_order must be 1 or 2, got {element_order}.")

    _ensure_supported_python()
    gmsh = _import_gmsh()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("unitsquare_hole")

        square_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, _SQUARE_WIDTH, _SQUARE_HEIGHT)
        hole_tags = [
            gmsh.model.occ.addDisk(x_center, y_center, 0.0, radius, radius)
            for x_center, y_center, radius in _HOLES
        ]

        cut_surfaces, _ = gmsh.model.occ.cut(
            [(2, square_tag)],
            [(2, hole_tag) for hole_tag in hole_tags],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        if len(cut_surfaces) != 1:
            raise RuntimeError(
                "Expected one surface after subtracting holes from the square, "
                f"got {cut_surfaces}."
            )

        domain_surface_tag = cut_surfaces[0][1]
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        boundaries = _classify_boundary_curves(gmsh, domain_surface_tag)

        gmsh.model.addPhysicalGroup(2, [domain_surface_tag], _DOMAIN_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(2, _DOMAIN_PHYSICAL_TAG, "domain")

        gmsh.model.addPhysicalGroup(1, boundaries["bottom"], _BOTTOM_BOUNDARY_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(1, _BOTTOM_BOUNDARY_PHYSICAL_TAG, "bottom")

        gmsh.model.addPhysicalGroup(1, boundaries["top"], _TOP_BOUNDARY_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(1, _TOP_BOUNDARY_PHYSICAL_TAG, "top")

        gmsh.model.addPhysicalGroup(1, boundaries["left"], _LEFT_BOUNDARY_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(1, _LEFT_BOUNDARY_PHYSICAL_TAG, "left")

        gmsh.model.addPhysicalGroup(1, boundaries["right"], _RIGHT_BOUNDARY_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(1, _RIGHT_BOUNDARY_PHYSICAL_TAG, "right")

        gmsh.model.addPhysicalGroup(1, boundaries["hole_boundary"], _HOLE_BOUNDARY_PHYSICAL_TAG)
        gmsh.model.setPhysicalName(1, _HOLE_BOUNDARY_PHYSICAL_TAG, "hole_boundary")

        gmsh.model.mesh.generate(2)
        if element_order == 2:
            # Raise to 2nd order and curve boundary mid-nodes onto the CAD curves
            # (HighOrderOptimize improves the resulting curved-element quality).
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

    return output_path


def main() -> None:
    output_path = build_mesh()
    print(f"Mesh generation complete. Wrote {output_path}.")


if __name__ == "__main__":
    main()
