"""Read-independent XYZ export for the molecular geometry database.

XYZ coordinates are always written in angstrom. Source geometries recorded in
bohr are converted during export, while the original in-memory data is left
unchanged.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .utils import sanitize_name


BOHR_TO_ANGSTROM = 0.529177210903
VALID_COORDINATE_UNITS = {"angstrom", "bohr"}


def geometry_in_angstrom(
    data: Mapping[str, Any],
) -> list[tuple[str, tuple[float, float, float]]]:
    """Return validated geometry coordinates converted to angstrom."""
    if "geometry" not in data:
        raise KeyError("Molecule data does not contain 'geometry'.")
    if "coordinate_unit" not in data:
        raise KeyError("Molecule data does not contain 'coordinate_unit'.")

    source_unit = str(data["coordinate_unit"]).lower()
    if source_unit not in VALID_COORDINATE_UNITS:
        raise ValueError(
            f"Unsupported coordinate unit {source_unit!r}; "
            "expected 'angstrom' or 'bohr'."
        )

    scale = BOHR_TO_ANGSTROM if source_unit == "bohr" else 1.0
    converted = []
    for atom_index, entry in enumerate(data["geometry"]):
        if len(entry) != 2:
            raise ValueError(f"Invalid geometry entry at atom {atom_index}: {entry!r}")
        symbol, coordinates = entry
        if len(coordinates) != 3:
            raise ValueError(
                f"Atom {atom_index} ({symbol}) must have exactly three coordinates."
            )
        x, y, z = (float(value) * scale for value in coordinates)
        converted.append((str(symbol), (x, y, z)))

    if not converted:
        raise ValueError("The molecular geometry is empty.")
    return converted


def xyz_text(
    name: str,
    data: Mapping[str, Any],
    *,
    precision: int = 8,
) -> str:
    """Serialize one molecule as standards-compatible XYZ text."""
    if precision < 1:
        raise ValueError("precision must be at least 1")

    geometry = geometry_in_angstrom(data)
    formula = data.get("formula", name)
    method = data.get(
        "geometry_optimized",
        data.get("geometry_method", "Unknown method"),
    )
    charge = data.get("charge", 0)
    spin = data.get("spin", 0)
    multiplicity = data.get("multiplicity", spin + 1)
    source_unit = str(data["coordinate_unit"]).lower()

    comment = (
        f"{name} | Formula={formula} | Method={method} | Charge={charge} | "
        f"Spin={spin} | Multiplicity={multiplicity} | Units=angstrom | "
        f"SourceUnits={source_unit}"
    )
    lines = [str(len(geometry)), comment]
    coordinate_width = precision + 7
    for symbol, (x, y, z) in geometry:
        lines.append(
            f"{symbol:<2} "
            f"{x:{coordinate_width}.{precision}f} "
            f"{y:{coordinate_width}.{precision}f} "
            f"{z:{coordinate_width}.{precision}f}"
        )
    return "\n".join(lines) + "\n"


def write_xyz(
    name: str,
    data: Mapping[str, Any],
    *,
    output_dir: str | Path = "xyz_files",
    precision: int = 8,
) -> Path:
    """Write one molecule to ``output_dir/<name>.xyz`` and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{sanitize_name(name)}.xyz"
    output_path.write_text(
        xyz_text(name, data, precision=precision),
        encoding="utf-8",
    )
    return output_path


def write_all_xyz(
    molecules_dict: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: str | Path = "xyz_files",
    names: Sequence[str] | None = None,
    precision: int = 8,
) -> dict[str, Path]:
    """Write every molecule, or a selected sequence, to individual XYZ files."""
    selected_names = list(molecules_dict) if names is None else list(names)
    unknown = [name for name in selected_names if name not in molecules_dict]
    if unknown:
        raise KeyError(f"Unknown molecule name(s): {', '.join(unknown)}")

    written = {}
    for name in selected_names:
        written[name] = write_xyz(
            name,
            molecules_dict[name],
            output_dir=output_dir,
            precision=precision,
        )
    return written


__all__ = [
    "BOHR_TO_ANGSTROM",
    "geometry_in_angstrom",
    "write_all_xyz",
    "write_xyz",
    "xyz_text",
]
