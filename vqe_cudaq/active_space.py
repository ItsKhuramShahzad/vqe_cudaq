"""Generation and analysis of complete-active-space configurations.

The functions in this module are independent of CUDA-Q and PySCF. They do not
mutate the molecular database, making them safe to use from notebooks, tests,
and the command-line interface.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import pandas as pd


ActiveSpace = dict[str, int]
REQUIRED_ACTIVE_SPACE_KEYS = ("ncore", "nele_cas", "norb_cas")


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    """Return an integer value or raise a descriptive validation error."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def validate_active_space(
    total_orbitals: int,
    ncore: int,
    nele_cas: int,
    norb_cas: int,
    *,
    require_external_virtual: bool = False,
) -> None:
    """Validate the physical capacity and orbital partition of one CAS."""
    total_orbitals = _require_int("total_orbitals", total_orbitals, minimum=1)
    ncore = _require_int("ncore", ncore, minimum=0)
    nele_cas = _require_int("nele_cas", nele_cas, minimum=1)
    norb_cas = _require_int("norb_cas", norb_cas, minimum=1)

    occupied_orbitals = ncore + norb_cas
    if occupied_orbitals > total_orbitals:
        raise ValueError(
            "ncore + norb_cas exceeds total_orbitals: "
            f"{ncore} + {norb_cas} > {total_orbitals}."
        )
    if nele_cas > 2 * norb_cas:
        raise ValueError(
            f"{nele_cas} active electrons do not fit in {norb_cas} "
            "spatial orbitals."
        )
    if require_external_virtual and occupied_orbitals >= total_orbitals:
        raise ValueError("At least one external virtual orbital is required.")


def generate_valid_active_spaces(
    total_orbitals: int = 40,
    total_electrons: int = 15,
    max_configs: int = 30,
    *,
    max_active_electrons: int = 8,
    min_ncore: int = 1,
    max_orbitals_per_active_electron: int = 2,
) -> list[ActiveSpace]:
    """Generate active spaces satisfying the project's selection constraints.

    Each generated configuration has at least one unoccupied active orbital
    and at least one external virtual orbital. Frozen-core orbitals are assumed
    to be doubly occupied.
    """
    total_orbitals = _require_int("total_orbitals", total_orbitals, minimum=3)
    total_electrons = _require_int("total_electrons", total_electrons, minimum=1)
    max_configs = _require_int("max_configs", max_configs, minimum=1)
    max_active_electrons = _require_int(
        "max_active_electrons", max_active_electrons, minimum=1
    )
    min_ncore = _require_int("min_ncore", min_ncore, minimum=0)
    max_orbitals_per_active_electron = _require_int(
        "max_orbitals_per_active_electron",
        max_orbitals_per_active_electron,
        minimum=1,
    )

    if total_electrons > 2 * total_orbitals:
        raise ValueError(
            f"{total_electrons} electrons exceed the capacity of "
            f"{total_orbitals} spatial orbitals."
        )
    if min_ncore >= total_orbitals - 1:
        raise ValueError(
            "min_ncore leaves no room for both active and external virtual orbitals."
        )

    valid_spaces: list[ActiveSpace] = []
    for ncore in range(min_ncore, total_orbitals - 1):
        nele_cas = total_electrons - 2 * ncore
        if nele_cas <= 0:
            break
        if nele_cas > max_active_electrons:
            continue

        # Reserve at least one orbital outside the active space.
        max_norb_cas = total_orbitals - ncore - 1
        for norb_cas in range(1, max_norb_cas + 1):
            if not (
                nele_cas <= 2 * norb_cas
                and norb_cas
                <= max_orbitals_per_active_electron * nele_cas
                and nele_cas <= 2 * (norb_cas - 1)
            ):
                continue

            validate_active_space(
                total_orbitals,
                ncore,
                nele_cas,
                norb_cas,
                require_external_virtual=True,
            )
            valid_spaces.append(
                {
                    "ncore": ncore,
                    "nele_cas": nele_cas,
                    "norb_cas": norb_cas,
                }
            )
            if len(valid_spaces) >= max_configs:
                return valid_spaces

    return valid_spaces


def generate_molecule_active_spaces(
    molecule_data: Mapping[str, Any],
    max_configs: int = 30,
    **kwargs: Any,
) -> list[ActiveSpace]:
    """Generate configurations using a molecule record's orbital metadata."""
    missing = [
        key
        for key in ("Total Spatial Orbitals", "Total Electrons")
        if key not in molecule_data
    ]
    if missing:
        raise KeyError(f"Molecule data is missing: {', '.join(missing)}")
    return generate_valid_active_spaces(
        total_orbitals=int(molecule_data["Total Spatial Orbitals"]),
        total_electrons=int(molecule_data["Total Electrons"]),
        max_configs=max_configs,
        **kwargs,
    )


def _orbital_range(start: int, stop: int) -> str:
    """Format a half-open orbital interval compactly."""
    if stop <= start:
        return "-"
    if stop == start + 1:
        return str(start)
    return f"{start} to {stop - 1}"


def analyze_active_space(
    total_orbitals: int,
    ncore: int,
    nele_cas: int,
    norb_cas: int,
    *,
    expected_total_electrons: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Describe the frozen, active, and external-virtual orbital partitions."""
    validate_active_space(total_orbitals, ncore, nele_cas, norb_cas)
    if expected_total_electrons is not None:
        expected_total_electrons = _require_int(
            "expected_total_electrons",
            expected_total_electrons,
            minimum=1,
        )

    active_stop = ncore + norb_cas
    frozen_electrons = 2 * ncore
    accounted_electrons = frozen_electrons + nele_cas
    summary = pd.DataFrame(
        {
            "Type": [
                "Frozen core",
                "Active space",
                "External virtual",
            ],
            "Orbital indices": [
                _orbital_range(0, ncore),
                _orbital_range(ncore, active_stop),
                _orbital_range(active_stop, total_orbitals),
            ],
            "Number of orbitals": [
                ncore,
                norb_cas,
                total_orbitals - active_stop,
            ],
            "Number of electrons": [
                frozen_electrons,
                nele_cas,
                0,
            ],
        }
    )
    summary.attrs.update(
        {
            "accounted_electrons": accounted_electrons,
            "expected_total_electrons": expected_total_electrons,
            "electron_count_matches": (
                None
                if expected_total_electrons is None
                else accounted_electrons == expected_total_electrons
            ),
        }
    )
    return summary, accounted_electrons


def summarize_active_spaces(
    total_orbitals: int,
    spaces: Sequence[Mapping[str, Any]],
    *,
    expected_total_electrons: int | None = None,
) -> pd.DataFrame:
    """Return one compact analysis row per active-space configuration."""
    rows = []
    for index, space in enumerate(spaces):
        missing = [key for key in REQUIRED_ACTIVE_SPACE_KEYS if key not in space]
        if missing:
            raise KeyError(
                f"Active-space configuration {index} is missing: {', '.join(missing)}"
            )

        ncore = int(space["ncore"])
        nele_cas = int(space["nele_cas"])
        norb_cas = int(space["norb_cas"])
        summary, accounted_electrons = analyze_active_space(
            total_orbitals,
            ncore,
            nele_cas,
            norb_cas,
            expected_total_electrons=expected_total_electrons,
        )
        external_virtual = int(summary.iloc[2]["Number of orbitals"])
        rows.append(
            {
                "index": index,
                "ncore": ncore,
                "nele_cas": nele_cas,
                "norb_cas": norb_cas,
                "external_virtual": external_virtual,
                "accounted_electrons": accounted_electrons,
                "expected_electrons": expected_total_electrons,
                "electron_count_matches": summary.attrs["electron_count_matches"],
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ActiveSpace",
    "analyze_active_space",
    "generate_molecule_active_spaces",
    "generate_valid_active_spaces",
    "summarize_active_spaces",
    "validate_active_space",
]
