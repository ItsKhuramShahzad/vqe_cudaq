"""Command-line entry point.

Overrides the run :mod:`config` from CLI flags, then dispatches to the driver.

Examples
--------
    python -m vqe_cudaq.cli --molecule Methylene --target nvidia --basis cc-pVDZ
    python -m vqe_cudaq.cli --all --target qpp-cpu --optimizer COBYLA
    python -m vqe_cudaq.cli --export-xyz --xyz-dir xyz_files
    python -m vqe_cudaq.cli --export-xyz --molecule Adenine
    python -m vqe_cudaq.cli --generate-active-spaces --total-orbitals 40 --total-electrons 15
    python -m vqe_cudaq.cli --analyze-active-spaces --molecule Methylene
"""

import os
import time
import argparse

from . import config
from .active_space import (
    generate_molecule_active_spaces,
    generate_valid_active_spaces,
    summarize_active_spaces,
)
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
    p.add_argument("--generate-active-spaces", "--generate_active_spaces",
                   action="store_true",
                   help="Generate valid active-space configurations and exit.")
    p.add_argument("--analyze-active-spaces", "--analyze_active_spaces",
                   action="store_true",
                   help="Analyze stored active spaces for --molecule or --all and exit.")
    p.add_argument("--total-orbitals", "--total_orbitals", type=int, default=None,
                   help="Total spatial orbitals for custom active-space generation.")
    p.add_argument("--total-electrons", "--total_electrons", type=int, default=None,
                   help="Total molecular electrons for custom active-space generation.")
    p.add_argument("--max-configs", "--max_configs", type=int, default=30,
                   help="Maximum generated active spaces (default: 30).")
    p.add_argument("--max-active-electrons", "--max_active_electrons",
                   type=int, default=8,
                   help="Maximum electrons in a generated active space (default: 8).")
    return p


def _print_active_spaces(title, spaces):
    """Print active-space dictionaries as a compact indexed table."""
    import pandas as pd

    print(f"\n{title}")
    if not spaces:
        print("No configurations satisfy the requested constraints.")
        return
    table = pd.DataFrame(spaces)
    table.index.name = "index"
    print(table.to_string())


def _run_active_space_generation(args):
    """Handle custom, single-molecule, or all-molecule generation."""
    generation_kwargs = {
        "max_configs": args.max_configs,
        "max_active_electrons": args.max_active_electrons,
    }

    if args.all:
        if args.total_orbitals is not None or args.total_electrons is not None:
            raise SystemExit(
                "Do not combine --all with --total-orbitals/--total-electrons."
            )
        generated_count = 0
        for name, data in molecules.items():
            try:
                spaces = generate_molecule_active_spaces(data, **generation_kwargs)
            except KeyError as error:
                print(f"[SKIP] {name}: {error}")
                continue
            _print_active_spaces(f"Generated active spaces: {name}", spaces)
            generated_count += 1
        if generated_count == 0:
            raise SystemExit("No molecule had sufficient orbital metadata.")
        return

    if args.molecule is not None:
        if args.molecule not in molecules:
            raise SystemExit(f"Molecule {args.molecule!r} not found.")
        data = molecules[args.molecule]
        if args.total_orbitals is None and args.total_electrons is None:
            try:
                spaces = generate_molecule_active_spaces(data, **generation_kwargs)
            except KeyError as error:
                raise SystemExit(
                    f"{args.molecule}: {error}. Provide --total-orbitals and "
                    "--total-electrons explicitly."
                ) from error
        else:
            if args.total_orbitals is None or args.total_electrons is None:
                raise SystemExit(
                    "Provide both --total-orbitals and --total-electrons."
                )
            spaces = generate_valid_active_spaces(
                args.total_orbitals,
                args.total_electrons,
                **generation_kwargs,
            )
        _print_active_spaces(
            f"Generated active spaces: {args.molecule}", spaces
        )
        return

    if args.total_orbitals is None or args.total_electrons is None:
        raise SystemExit(
            "Provide --molecule NAME, --all, or both --total-orbitals and "
            "--total-electrons."
        )
    spaces = generate_valid_active_spaces(
        args.total_orbitals,
        args.total_electrons,
        **generation_kwargs,
    )
    _print_active_spaces("Generated active spaces: custom system", spaces)


def _run_active_space_analysis(args):
    """Analyze stored active spaces without mutating the molecule database."""
    if not args.all and args.molecule is None:
        raise SystemExit("Provide --molecule NAME or --all for analysis.")
    if args.all and args.molecule is not None:
        raise SystemExit("Choose either --molecule NAME or --all, not both.")

    names = list(molecules) if args.all else [args.molecule]
    analyzed_count = 0
    for name in names:
        if name not in molecules:
            raise SystemExit(f"Molecule {name!r} not found.")
        data = molecules[name]
        total_orbitals = (
            args.total_orbitals
            if args.total_orbitals is not None
            else data.get("Total Spatial Orbitals")
        )
        if total_orbitals is None:
            print(
                f"[SKIP] {name}: missing 'Total Spatial Orbitals'; provide "
                "--total-orbitals when analyzing this molecule alone."
            )
            continue

        spaces = list(data.get("valid_active_spaces", []))
        if args.space_idx is not None:
            if len(names) != 1:
                raise SystemExit("--space_idx can only be used with one molecule.")
            if args.space_idx < 0 or args.space_idx >= len(spaces):
                raise SystemExit(f"--space_idx {args.space_idx} out of range.")
            spaces = [spaces[args.space_idx]]

        table = summarize_active_spaces(
            int(total_orbitals),
            spaces,
            expected_total_electrons=int(data["Total Electrons"]),
        )
        print(f"\nActive-space analysis: {name}")
        print(table.to_string(index=False))
        mismatches = int((table["electron_count_matches"] == False).sum())
        if mismatches:
            print(
                f"[WARNING] {mismatches} configuration(s) do not account for "
                "the molecule's declared total electron count."
            )
        analyzed_count += 1

    if analyzed_count == 0:
        raise SystemExit("No molecule could be analyzed with the available metadata.")


def _apply_config(args):
    """Push CLI options into the shared config namespace."""
    config.BASIS = args.basis
    config.TARGET = args.target
    config.OPTIMIZER = args.optimizer
    config.TARGET_PRECISION = None if args.precision == "default" else args.precision


def main(argv=None):
    args = build_parser().parse_args(argv)

    utility_modes = (
        args.export_xyz,
        args.generate_active_spaces,
        args.analyze_active_spaces,
    )
    if sum(utility_modes) > 1:
        raise SystemExit(
            "Choose only one utility mode: --export-xyz, "
            "--generate-active-spaces, or --analyze-active-spaces."
        )

    if args.generate_active_spaces:
        _run_active_space_generation(args)
        return

    if args.analyze_active_spaces:
        _run_active_space_analysis(args)
        return

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
