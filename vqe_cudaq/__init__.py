"""
vqe_cudaq
=========

Active-space Variational Quantum Eigensolver (VQE) benchmarking for molecular
ground-state energies using CUDA-Q, OpenFermion and PySCF.

The molecule database and run configuration import without CUDA-Q, so
downstream analysis tooling can use them on any machine::

    from vqe_cudaq.molecules import molecules
    from vqe_cudaq import config

The execution engine (driver/vqe/ansatz/backend) requires CUDA-Q and is
imported lazily -- ``vqe_cudaq.run_one_molecule`` / ``run_all_molecules`` pull
it in on first access::

    from vqe_cudaq import run_one_molecule
"""

from . import config
from .molecules import molecules
from .xyz import convert_geometry_to_angstrom, write_all_xyz, write_xyz, xyz_text

__version__ = "1.0.0"

__all__ = [
    "molecules",
    "config",
    "run_one_molecule",
    "run_all_molecules",
    "write_all_xyz",
    "write_xyz",
    "xyz_text",
    "convert_geometry_to_angstrom",
    "analyze_active_space",
    "generate_molecule_active_spaces",
    "generate_valid_active_spaces",
    "summarize_active_spaces",
    "validate_active_space",
    "export_orbital_bundle",
    "frontier_orbital_indices",
    "orbital_table",
    "plot_orbital_energy_diagram",
    "run_orbital_calculation",
    "view_orbital_cube",
]


def __getattr__(name):
    # Lazy import so `import vqe_cudaq` / `from vqe_cudaq.molecules import ...`
    # do not require CUDA-Q to be installed.
    if name in ("run_one_molecule", "run_all_molecules"):
        from . import driver
        return getattr(driver, name)
    if name in (
        "analyze_active_space",
        "generate_molecule_active_spaces",
        "generate_valid_active_spaces",
        "summarize_active_spaces",
        "validate_active_space",
    ):
        from . import active_space
        return getattr(active_space, name)
    if name in (
        "export_orbital_bundle",
        "frontier_orbital_indices",
        "orbital_table",
        "plot_orbital_energy_diagram",
        "run_orbital_calculation",
        "view_orbital_cube",
    ):
        from . import orbitals
        return getattr(orbitals, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
