"""Molecular-orbital calculation, export, plotting, and Jupyter viewing.

The Molden file and energy table contain every canonical molecular orbital.
Cube files are generated only for a requested frontier window by default,
because an all-orbital cube export can consume many gigabytes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyscf import dft, gto, scf
from pyscf.tools import cubegen, molden

from .utils import sanitize_name
from .xyz import geometry_in_angstrom, xyz_text


HARTREE_TO_EV = 27.211386245988


@dataclass
class OrbitalCalculation:
    """A converged PySCF calculation and the metadata used to create it."""

    name: str
    molecule_data: Mapping[str, Any]
    basis: str
    method: str
    mean_field: Any

    @property
    def molecule(self):
        return self.mean_field.mol

    @property
    def converged(self) -> bool:
        return bool(self.mean_field.converged)

    @property
    def total_energy(self) -> float:
        return float(self.mean_field.e_tot)

    @property
    def mo_coeff(self) -> np.ndarray:
        coefficients = np.asarray(self.mean_field.mo_coeff)
        if coefficients.ndim != 2:
            raise ValueError(
                "Only restricted/ROHF orbital coefficients are supported; "
                f"received shape {coefficients.shape}."
            )
        return coefficients

    @property
    def mo_energy(self) -> np.ndarray:
        return np.asarray(self.mean_field.mo_energy, dtype=float)

    @property
    def mo_occ(self) -> np.ndarray:
        return np.asarray(self.mean_field.mo_occ, dtype=float)


def run_orbital_calculation(
    name: str,
    data: Mapping[str, Any],
    *,
    basis: str = "cc-pVDZ",
    method: str = "hf",
    xc: str = "b3lyp",
    conv_tol: float = 1e-9,
    max_cycle: int = 100,
    verbose: int = 0,
    require_converged: bool = True,
) -> OrbitalCalculation:
    """Run a restricted HF/DFT calculation suitable for MO inspection.

    Closed-shell systems use RHF/RKS and open-shell systems use ROHF/ROKS,
    giving one spatial-orbital coefficient matrix in both cases.
    """
    method = method.lower()
    if method not in {"hf", "dft"}:
        raise ValueError("method must be 'hf' or 'dft'")

    molecule = gto.M(
        atom=geometry_in_angstrom(data),
        unit="Angstrom",
        basis=basis,
        charge=int(data.get("charge", 0)),
        spin=int(data.get("spin", int(data.get("multiplicity", 1)) - 1)),
        verbose=verbose,
    )

    if method == "hf":
        mean_field = scf.RHF(molecule) if molecule.spin == 0 else scf.ROHF(molecule)
        method_label = "RHF" if molecule.spin == 0 else "ROHF"
    else:
        mean_field = dft.RKS(molecule) if molecule.spin == 0 else dft.ROKS(molecule)
        mean_field.xc = xc
        method_label = f"{'RKS' if molecule.spin == 0 else 'ROKS'}-{xc}"

    mean_field.conv_tol = float(conv_tol)
    mean_field.max_cycle = int(max_cycle)
    mean_field.kernel()
    result = OrbitalCalculation(
        name=name,
        molecule_data=dict(data),
        basis=basis,
        method=method_label,
        mean_field=mean_field,
    )
    if require_converged and not result.converged:
        raise RuntimeError(
            f"{method_label}/{basis} did not converge for {name} in "
            f"{max_cycle} SCF cycles."
        )
    return result


def frontier_orbital_indices(
    calculation: OrbitalCalculation,
    *,
    occupied_below: int = 3,
    virtual_above: int = 3,
) -> list[int]:
    """Return a zero-based HOMO/LUMO window, clipped to valid MO indices."""
    if occupied_below < 0 or virtual_above < 0:
        raise ValueError("frontier window sizes cannot be negative")
    occupied = np.flatnonzero(calculation.mo_occ > 1e-8)
    virtual = np.flatnonzero(calculation.mo_occ <= 1e-8)
    chosen = list(occupied[-(occupied_below + 1):])
    chosen.extend(list(virtual[: virtual_above + 1]))
    return [int(index) for index in dict.fromkeys(chosen)]


def _frontier_labels(occupations: np.ndarray) -> dict[int, str]:
    occupied = np.flatnonzero(occupations > 1e-8)
    virtual = np.flatnonzero(occupations <= 1e-8)
    labels: dict[int, str] = {}
    if occupied.size:
        homo = int(occupied[-1])
        for index in occupied:
            offset = homo - int(index)
            labels[int(index)] = "HOMO" if offset == 0 else f"HOMO-{offset}"
        if occupations[homo] < 2.0 - 1e-8:
            labels[homo] = "SOMO (HOMO)"
    if virtual.size:
        lumo = int(virtual[0])
        for index in virtual:
            offset = int(index) - lumo
            labels[int(index)] = "LUMO" if offset == 0 else f"LUMO+{offset}"
    return labels


def orbital_table(calculation: OrbitalCalculation) -> pd.DataFrame:
    """Return energies and occupations for all canonical spatial orbitals."""
    energies = calculation.mo_energy
    occupations = calculation.mo_occ
    labels = _frontier_labels(occupations)
    return pd.DataFrame(
        {
            "mo_index": np.arange(len(energies), dtype=int),
            "mo_number": np.arange(1, len(energies) + 1, dtype=int),
            "energy_hartree": energies,
            "energy_ev": energies * HARTREE_TO_EV,
            "occupation": occupations,
            "frontier_label": [labels.get(i, "") for i in range(len(energies))],
        }
    )


def _validate_indices(
    calculation: OrbitalCalculation,
    indices: Sequence[int],
) -> list[int]:
    unique = list(dict.fromkeys(int(index) for index in indices))
    invalid = [i for i in unique if i < 0 or i >= len(calculation.mo_energy)]
    if invalid:
        raise IndexError(
            f"MO indices {invalid} are outside 0..{len(calculation.mo_energy) - 1}."
        )
    return unique


def export_molden(
    calculation: OrbitalCalculation,
    output_path: str | Path,
) -> Path:
    """Export all orbitals in a file readable by Molden, Avogadro, and Jmol."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    molden.from_scf(calculation.mean_field, str(path))
    return path


def export_orbital_cube(
    calculation: OrbitalCalculation,
    mo_index: int,
    output_dir: str | Path,
    *,
    grid: int = 80,
) -> Path:
    """Export one zero-based canonical MO as a Gaussian cube file."""
    mo_index = _validate_indices(calculation, [mo_index])[0]
    if grid < 10:
        raise ValueError("grid must be at least 10 points per axis")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mo_{mo_index:03d}.cube"
    cubegen.orbital(
        calculation.molecule,
        str(path),
        calculation.mo_coeff[:, mo_index],
        nx=grid,
        ny=grid,
        nz=grid,
    )
    return path


def plot_orbital_energy_diagram(
    calculation: OrbitalCalculation,
    *,
    indices: Sequence[int] | None = None,
    active_indices: Sequence[int] | None = None,
    output_path: str | Path | None = None,
):
    """Plot an MO energy-level diagram and optionally highlight an active set."""
    if indices is None:
        indices = list(range(len(calculation.mo_energy)))
    indices = _validate_indices(calculation, indices)
    active = set(_validate_indices(calculation, active_indices or []))
    labels = _frontier_labels(calculation.mo_occ)

    fig_height = max(5.0, min(12.0, 0.28 * len(indices) + 3.0))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    for index in indices:
        energy = calculation.mo_energy[index] * HARTREE_TO_EV
        occupation = calculation.mo_occ[index]
        color = "#7b2cbf" if index in active else (
            "#1565c0" if occupation >= 2 - 1e-8 else
            "#ef6c00" if occupation > 1e-8 else "#616161"
        )
        linewidth = 4.0 if index in active else 2.4
        ax.hlines(energy, 0.15, 0.72, color=color, linewidth=linewidth)
        annotation = f"MO {index}  occ={occupation:g}"
        if labels.get(index):
            annotation += f"  {labels[index]}"
        ax.text(0.76, energy, annotation, va="center", fontsize=8)

    ax.set_xlim(0, 1.75)
    ax.set_xticks([])
    ax.set_ylabel("Orbital energy (eV)")
    ax.set_title(
        f"{calculation.name}: canonical molecular orbitals\n"
        f"{calculation.method}/{calculation.basis}"
    )
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    return fig, ax


def export_orbital_bundle(
    calculation: OrbitalCalculation,
    *,
    output_dir: str | Path = "molecular_orbitals",
    indices: Sequence[int] | None = None,
    occupied_below: int = 3,
    virtual_above: int = 3,
    grid: int = 80,
    export_cubes: bool = True,
) -> dict[str, Any]:
    """Export all-MO metadata plus selected frontier-orbital cube files."""
    if indices is None:
        indices = frontier_orbital_indices(
            calculation,
            occupied_below=occupied_below,
            virtual_above=virtual_above,
        )
    indices = _validate_indices(calculation, indices)
    calculation_tag = sanitize_name(
        f"{calculation.method}_{calculation.basis}"
    )
    directory = (
        Path(output_dir) / sanitize_name(calculation.name) / calculation_tag
    )
    directory.mkdir(parents=True, exist_ok=True)

    molden_path = export_molden(calculation, directory / "all_orbitals.molden")
    table_path = directory / "orbital_energies.csv"
    orbital_table(calculation).to_csv(table_path, index=False)
    diagram_path = directory / "orbital_energy_diagram.png"
    fig, _ = plot_orbital_energy_diagram(
        calculation,
        indices=indices,
        output_path=diagram_path,
    )
    plt.close(fig)

    cube_paths = []
    if export_cubes:
        cube_paths = [
            export_orbital_cube(calculation, index, directory, grid=grid)
            for index in indices
        ]
    return {
        "directory": directory,
        "molden": molden_path,
        "table": table_path,
        "diagram": diagram_path,
        "cubes": cube_paths,
        "selected_indices": indices,
    }


def view_orbital_cube(
    calculation: OrbitalCalculation,
    cube_path: str | Path,
    *,
    isovalue: float = 0.03,
    width: int = 750,
    height: int = 550,
):
    """Create a py3Dmol positive/negative MO isosurface view for Jupyter."""
    if isovalue <= 0:
        raise ValueError("isovalue must be positive")
    try:
        import py3Dmol
    except ImportError as error:
        raise ImportError(
            "Jupyter MO viewing requires py3Dmol; install with "
            "`pip install -e '.[visualization]'`."
        ) from error

    cube_data = Path(cube_path).read_text(encoding="utf-8")
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_text(calculation.name, calculation.molecule_data), "xyz")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.22}})
    view.addVolumetricData(
        cube_data,
        "cube",
        {"isoval": isovalue, "color": "blue", "opacity": 0.70},
    )
    view.addVolumetricData(
        cube_data,
        "cube",
        {"isoval": -isovalue, "color": "red", "opacity": 0.70},
    )
    view.zoomTo()
    return view


__all__ = [
    "HARTREE_TO_EV",
    "OrbitalCalculation",
    "export_molden",
    "export_orbital_bundle",
    "export_orbital_cube",
    "frontier_orbital_indices",
    "orbital_table",
    "plot_orbital_energy_diagram",
    "run_orbital_calculation",
    "view_orbital_cube",
]
