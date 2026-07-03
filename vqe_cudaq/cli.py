"""Command-line entry point.

Overrides the run :mod:`config` from CLI flags, then dispatches to the driver.

Examples
--------
    python -m vqe_cudaq.cli --molecule Methylene --target nvidia --basis cc-pVDZ
    python -m vqe_cudaq.cli --all --target qpp-cpu --optimizer COBYLA
    python -m vqe_cudaq.cli --export-xyz --xyz-dir xyz_files
    python -m vqe_cudaq.cli --export-xyz --molecule Adenine
"""

import os
import time
import argparse

from . import config
from .molecules import molecules
from .utils import sanitize_name, save_pkl
from .xyz import write_all_xyz, write_xyz


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Active-space VQE benchmarking on CUDA-Q (CPU/GPU).")
    p.add_argument("--molecule", default=None,
                   help="Molecule name (see vqe_cudaq.molecules). Omit with --all.")
    p.add_argument("--all", action="store_true",
                   help="Run every molecule in the database.")
    p.add_argument("--basis", default=config.BASIS)
    p.add_argument("--target", default=config.TARGET,
                   help='CUDA-Q target, e.g. "nvidia" or "qpp-cpu".')
    p.add_argument("--precision", default="default",
                   choices=["default", "fp32", "fp64"],
                   help="CUDA-Q precision for the nvidia target.")
    p.add_argument("--optimizer", default=config.OPTIMIZER)
    p.add_argument("--out_dir", default="pkl_results")
    p.add_argument("--space_idx", type=int, default=None,
                   help="Run only the i-th active space of a single molecule.")
    p.add_argument("--export-xyz", "--export_xyz", action="store_true",
                   help="Export molecular geometries as XYZ files instead of running VQE.")
    p.add_argument("--xyz-dir", "--xyz_dir", default="xyz_files",
                   help="XYZ output directory (default: xyz_files).")
    return p


def _apply_config(args):
    """Push CLI options into the shared config namespace."""
    config.BASIS = args.basis
    config.TARGET = args.target
    config.OPTIMIZER = args.optimizer
    config.TARGET_PRECISION = None if args.precision == "default" else args.precision


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.export_xyz:
        if args.molecule is not None:
            if args.molecule not in molecules:
                raise SystemExit(f"Molecule {args.molecule!r} not found.")
            paths = {
                args.molecule: write_xyz(
                    args.molecule,
                    molecules[args.molecule],
                    output_dir=args.xyz_dir,
                )
            }
        else:
            paths = write_all_xyz(molecules, output_dir=args.xyz_dir)

        for name, path in paths.items():
            print(f"[XYZ] {name} -> {path}", flush=True)
        print(f"[DONE] Exported {len(paths)} XYZ file(s).", flush=True)
        return

    _apply_config(args)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Keep CUDA-Q and the VQE engine optional for geometry-only CLI use.
    from .driver import run_one_molecule, run_all_molecules

    if args.all:
        run_all_molecules(molecules, out_dir=out_dir)
        return

    if args.molecule is None:
        raise SystemExit("Provide --molecule NAME or --all.")

    molecule_name = args.molecule
    if molecule_name not in molecules:
        raise ValueError(f"Molecule '{molecule_name}' not found!")

    spec = dict(molecules[molecule_name])

    if args.space_idx is not None:
        spaces = spec.get("valid_active_spaces", [])
        if args.space_idx < 0 or args.space_idx >= len(spaces):
            raise ValueError(f"--space_idx {args.space_idx} out of range")
        spec["valid_active_spaces"] = [spaces[args.space_idx]]

    print(
        f"[RUN] {molecule_name} | BASIS={config.BASIS} | TARGET={config.TARGET} "
        f"| PRECISION={config.TARGET_PRECISION or 'default'} | OPT={config.OPTIMIZER}",
        flush=True,
    )

    mol_res = run_one_molecule(molecule_name, spec)

    tag       = time.strftime("%d_%b_%Y").upper()
    mol_clean = sanitize_name(molecule_name)
    file_name = (
        f"{tag}_{mol_clean}_{sanitize_name(config.BASIS)}_"
        f"{sanitize_name(config.TARGET)}_{sanitize_name(config.OPTIMIZER)}_VQE_results.pkl"
    )
    out_path = os.path.join(out_dir, file_name)
    save_pkl({molecule_name: mol_res}, out_path)
    print(f"[DONE] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
