"""Interactive 3D visualization of molecular geometries.

The geometry and its coordinate unit come from :mod:`vqe_cudaq.molecules`.
Bond connectivity is inferred for display only; it is not used by the VQE
calculation.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import py3Dmol
from IPython.display import HTML, display
from rdkit import Chem


BOHR_TO_ANGSTROM = 0.529177210903

# Approximate single-bond covalent radii in angstrom.
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "Cl": 1.02,
}


def create_molecule(
    coords: Sequence[tuple[str, Sequence[float]]],
    *,
    unit: str = "angstrom",
    bond_factor: float = 1.25,
) -> Chem.Mol:
    """Create an RDKit molecule with inferred bonds and a 3D conformer."""
    normalized_unit = unit.lower()
    if normalized_unit not in {"angstrom", "bohr"}:
        raise ValueError("unit must be either 'angstrom' or 'bohr'")
    if not coords:
        raise ValueError("The coordinate list is empty.")

    symbols = [symbol for symbol, _ in coords]
    positions = np.asarray([position for _, position in coords], dtype=float)
    if positions.shape != (len(symbols), 3):
        raise ValueError("Each atom must have exactly three coordinates (x, y, z).")

    if normalized_unit == "bohr":
        positions *= BOHR_TO_ANGSTROM

    editable_mol = Chem.RWMol()
    for symbol in symbols:
        try:
            editable_mol.AddAtom(Chem.Atom(symbol))
        except Exception as error:
            raise ValueError(f"Invalid chemical element: {symbol!r}") from error

    # Infer display bonds from distances and covalent radii.
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            radius_i = COVALENT_RADII.get(symbols[i], 0.7)
            radius_j = COVALENT_RADII.get(symbols[j], 0.7)
            cutoff = bond_factor * (radius_i + radius_j)
            if 0.4 < distance <= cutoff:
                editable_mol.AddBond(i, j, Chem.BondType.SINGLE)

    molecule = editable_mol.GetMol()
    conformer = Chem.Conformer(len(symbols))
    for atom_index, (x, y, z) in enumerate(positions):
        conformer.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def visualize_molecule(
    name: str,
    data: Mapping[str, Any],
    *,
    width: int = 600,
    height: int = 450,
):
    """Build, but do not display, a py3Dmol view for one molecule."""
    if "geometry" not in data:
        raise KeyError(f"The data for {name!r} does not contain 'geometry'.")
    if "coordinate_unit" not in data:
        raise KeyError(f"The data for {name!r} does not contain 'coordinate_unit'.")

    unit = str(data["coordinate_unit"]).lower()
    molecule = create_molecule(data["geometry"], unit=unit)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(Chem.MolToMolBlock(molecule), "mol")
    view.setStyle(
        {
            "stick": {"radius": 0.15},
            "sphere": {"scale": 0.25},
        }
    )
    view.setBackgroundColor("white")
    view.zoomTo()

    display(
        HTML(
            f"<h3>{name}</h3>"
            f"<p>Input coordinate unit: {unit}</p>"
        )
    )
    return view


def visualize_one(
    name: str,
    molecules_dict: Mapping[str, Mapping[str, Any]],
    *,
    width: int = 600,
    height: int = 450,
):
    """Display one molecule selected by name."""
    if name not in molecules_dict:
        available = ", ".join(sorted(molecules_dict))
        raise KeyError(
            f"Unknown molecule: {name!r}. Available molecules: {available}"
        )

    view = visualize_molecule(
        name,
        molecules_dict[name],
        width=width,
        height=height,
    )
    view.show()
    return view


def visualize_all(
    molecules_dict: Mapping[str, Mapping[str, Any]],
    *,
    names: Sequence[str] | None = None,
    width: int = 600,
    height: int = 450,
) -> dict[str, Any]:
    """Display all molecules, or a selected sequence of molecule names."""
    selected_names = list(molecules_dict) if names is None else list(names)
    unknown = [name for name in selected_names if name not in molecules_dict]
    if unknown:
        raise KeyError(f"Unknown molecule name(s): {', '.join(unknown)}")

    displayed_views = {}
    for name in selected_names:
        view = visualize_molecule(
            name,
            molecules_dict[name],
            width=width,
            height=height,
        )
        view.show()
        displayed_views[name] = view
    return displayed_views


__all__ = [
    "BOHR_TO_ANGSTROM",
    "COVALENT_RADII",
    "create_molecule",
    "visualize_all",
    "visualize_molecule",
    "visualize_one",
]
