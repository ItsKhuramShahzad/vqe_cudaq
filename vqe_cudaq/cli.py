"""Command-line entry point.

Overrides the run :mod:`config` from CLI flags, then dispatches to the driver.

Examples
--------
    python -m vqe_cudaq.cli --molecule Ethylene --target nvidia --basis cc-pVDZ
    python -m vqe_cudaq.cli --all --target qpp-cpu --optimizer COBYLA
    python -m vqe_cudaq.cli --export-xyz --xyz-dir xyz_files
    python -m vqe_cudaq.cli --export-xyz --molecule Adenine
    python -m vqe_cudaq.cli --generate-active-spaces --total-orbitals 40 --total-electrons 15
    python -m vqe_cudaq.cli --analyze-active-spaces --molecule Ethylene
    python -m vqe_cudaq.cli --export-mos --molecule Ethylene --basis cc-pVDZ

Choosing the active orbitals (default: energy, the published spaces):
    python -m vqe_cudaq.cli --molecule Ethylene --active-space-rule manual \
        --active-orbitals HOMO HOMO-2 LUMO LUMO+3
    python -m vqe_cudaq.cli --molecule Ethylene --active-space-rule bond-type \
        --bond-type-dir analysis/bond_type_casci_hpc --sizes 6,8 4,4 --rank 1
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
    p.add_argument("--precision", default="fp64",
                   choices=["default", "fp32", "fp64"],
                   help="CUDA-Q precision for the nvidia target (default fp64; "
                        "any run that is not fp64 stops).")
    p.add_argument("--optimizer", default=config.OPTIMIZER)
    p.add_argument("--out_dir", default="pkl_results")
    p.add_argument("--space_idx", type=int, default=None,
                   help="Run only the i-th active space of a single molecule.")
    p.add_argument("--integrals", default=None,
                   help="Read the Hamiltonian, E_HF, E_CASCI and CCSD amplitudes from "
                        "this integrals folder instead of running PySCF; missing files "
                        "are computed and saved there.")
    p.add_argument("--active-space-rule", "--orbital-rule", dest="active_space_rule", default="energy",
                   choices=["energy", "bond-type", "manual"],
                   help="How the active orbitals are chosen: energy = the molecule's published "
                        "spaces (nearest the gap); bond-type = the best spaces found by "
                        "scripts/bond_type_casci.py (--bond-type-dir, --sizes, --rank); "
                        "manual = the orbitals in --active-orbitals.")
    p.add_argument("--active-orbitals", nargs="+", default=None,
                   help="manual rule: orbitals such as HOMO HOMO-2 LUMO LUMO+3 (or MO numbers "
                        "from 0); the other occupied orbitals are frozen.")
    p.add_argument("--bond-type-dir", default=os.path.join("analysis", "bond_type_casci_hpc"),
                   help="bond-type rule: folder with <molecule>_bond_type_casci.csv "
                        "(and _left_out.csv). Sizes that are not there are searched and added.")
    p.add_argument("--bond-type-compute", default="missing", choices=["missing", "never"],
                   help="bond-type rule: missing = search a size that is not in --bond-type-dir yet "
                        "(SCF, orbital types, CASCI of that size; default); never = only read the files.")
    p.add_argument("--sizes", nargs="+", default=None,
                   help="bond-type rule: active-space sizes as electrons,orbitals, e.g. 6,8 4,4 "
                        "(the --rank best space of each), or 'all': every size up to "
                        "--bond-type-max-electrons/--bond-type-max-orbitals, selected with "
                        "--ratio-max and --select.")
    p.add_argument("--bond-type-max-electrons", type=int, default=8,
                   help="bond-type rule with --sizes all: largest number of active electrons (8).")
    p.add_argument("--bond-type-max-orbitals", type=int, default=10,
                   help="bond-type rule with --sizes all: largest number of active orbitals (10).")
    p.add_argument("--ratio-max", type=float, default=0.25,
                   help="bond-type rule with --sizes all: a space passes when "
                        "(E_corr - E_best)/|E_best| < this (0.25); E_best = best E_corr of all sizes.")
    p.add_argument("--select", default="best-per-size", choices=["best-per-size", "top", "all"],
                   help="bond-type rule with --sizes all: which passing spaces get a VQE run: "
                        "best-per-size = the best of each size (default), top = the --top best "
                        "overall, all = every passing space.")
    p.add_argument("--top", type=int, default=5,
                   help="with --select top: number of spaces (5).")
    p.add_argument("--rank", type=int, default=1,
                   help="bond-type rule: 1 = the best E_corr of that size, 2 = second best, ...")
    p.add_argument("--max-memory", "--max_memory", type=int, default=config.MAX_MEMORY,
                   help="PySCF memory limit in MB when a missing integral file is made.")
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
    p.add_argument("--export-mos", "--export_mos", action="store_true",
                   help="Calculate and export molecular orbitals instead of running VQE.")
    p.add_argument("--mo-method", choices=["hf", "dft"], default="hf",
                   help="Mean-field method for orbital generation (default: hf).")
    p.add_argument("--xc", default="b3lyp",
                   help="DFT functional used with --mo-method dft (default: b3lyp).")
    p.add_argument("--mo-output-dir", default="molecular_orbitals",
                   help="MO output directory (default: molecular_orbitals).")
    p.add_argument("--mo-window", type=int, default=3,
                   help="Orbitals below HOMO and above LUMO to export as cubes.")
    p.add_argument("--mo-indices", nargs="*", type=int, default=None,
                   help="Explicit zero-based MO indices for diagrams/cubes.")
    p.add_argument("--mo-grid", type=int, default=80,
                   help="Cube grid points per axis (default: 80).")
    p.add_argument("--no-mo-cubes", action="store_true",
                   help="Export Molden/table/diagram only; skip cube files.")
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


def chosen_active_spaces(args, name, spec):
    """The active spaces of the manual and bond-type rules, as driver space dicts."""
    from pyscf import gto
    from .chosen_spaces import bond_type_space, chosen_space
    from .xyz import geometry_in_angstrom

    if args.space_idx is not None:
        raise SystemExit("--space_idx picks a published (energy-rule) space; it cannot be "
                         f"used with --active-space-rule {args.active_space_rule}.")
    mol = gto.M(atom=geometry_in_angstrom(spec), unit="Angstrom", basis=args.basis,
                charge=int(spec["charge"]), spin=int(spec["multiplicity"]) - 1, verbose=0)
    nocc, nmo = mol.nelectron // 2, mol.nao_nr()
    if args.active_space_rule == "manual":
        if not args.active_orbitals:
            raise SystemExit("--active-space-rule manual needs --active-orbitals, e.g. HOMO HOMO-2 LUMO LUMO+3")
        return [chosen_space(args.active_orbitals, nocc, nmo, rule="manual")]
    if not args.sizes:
        raise SystemExit("--active-space-rule bond-type needs --sizes, e.g. --sizes 6,8 4,4 or --sizes all")
    if [s.lower() for s in args.sizes] == ["all"]:
        return passing_active_spaces(args, name, spec, mol)
    sizes = []
    for size in args.sizes:
        try:
            nele, norb = (int(v) for v in size.replace("(", "").replace(")", "")
                          .replace("e", "").replace("o", "").split(","))
        except ValueError:
            raise SystemExit(f"--sizes {size!r}: write electrons,orbitals, e.g. 6,8")
        if nele % 2 or nele < 2 or nele // 2 >= norb:
            raise SystemExit(f"--sizes {size!r}: needs an even number of electrons and more "
                             f"orbitals than electron pairs, e.g. 6,8")
        sizes.append((nele, norb))
    if args.bond_type_compute == "missing":
        from .bond_type_search import ensure_size, stored_sizes
        missing = [s for s in sizes if s not in stored_sizes(args.bond_type_dir, name)]
        if missing:
            from .hamiltonian import run_scf
            mf = run_scf(spec, args.basis, max_memory=args.max_memory)
            for nele, norb in missing:
                ensure_size(args.bond_type_dir, name, mf, nele, norb, max_memory=args.max_memory)
    return [bond_type_space(args.bond_type_dir, name, nele, norb, nocc, nmo, args.rank)
            for nele, norb in sizes]


def passing_active_spaces(args, name, spec, mol):
    """--sizes all: search every size up to the limits (missing ones only), then select the
    spaces that pass the ratio criterion (--ratio-max) with --select / --top."""
    from pyscf.data.elements import chemcore
    from .chosen_spaces import ChosenSpaceError, passing_spaces

    max_e, max_o = args.bond_type_max_electrons, args.bond_type_max_orbitals
    if max_e < 2 or max_e % 2 or max_o <= max_e // 2:
        raise SystemExit(f"--bond-type-max-electrons {max_e} / --bond-type-max-orbitals {max_o}: "
                         f"need an even number of electrons and more orbitals than electron pairs")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    nocc, nmo = mol.nelectron // 2, mol.nao_nr()
    if args.bond_type_compute == "missing":
        from .bond_type_search import all_sizes, ensure_sizes, stored_sizes
        sizes = all_sizes(max_e, max_o, nocc - chemcore(mol), nmo - nocc)
        if set(sizes) - stored_sizes(args.bond_type_dir, name):
            from .hamiltonian import run_scf
            mf = run_scf(spec, args.basis, max_memory=args.max_memory)
            ensure_sizes(args.bond_type_dir, name, mf, sizes, max_memory=args.max_memory, require=False)
    try:
        spaces = passing_spaces(args.bond_type_dir, name, nocc, nmo, max_e, max_o,
                                ratio_max=args.ratio_max, select=args.select, top=args.top)
    except ChosenSpaceError as err:
        raise SystemExit(str(err))
    if not spaces:
        raise SystemExit(f"{name}: no space passes ratio < {args.ratio_max:g}")
    s0 = spaces[0]
    print(f"[SELECT] {name}: E_best {s0['E_best_mHa']:.3f} mHa over all sizes up to ({max_e}e,{max_o}o); "
          f"{len(spaces)} space(s) with ratio < {args.ratio_max:g}, select = {args.select}"
          f"{f' (top {args.top})' if args.select == 'top' else ''}", flush=True)
    return spaces


def space_line(s):
    """One line about a chosen space for the run log."""
    line = f"({s['nele_cas']}e,{s['norb_cas']}o) {s['label']}"
    if s.get("left_out"):
        line += f" (left out {s['left_out']})"
    if "ratio" in s:
        line += f"  E_corr {s['E_corr_mHa']:.3f} mHa, ratio {s['ratio']:.3f}"
    return line


def _apply_config(args):
    """Push CLI options into the shared config namespace."""
    config.BASIS = args.basis
    config.TARGET = args.target
    config.OPTIMIZER = args.optimizer
    config.TARGET_PRECISION = None if args.precision == "default" else args.precision
    config.MAX_MEMORY = args.max_memory


def _run_orbital_export(args):
    """Calculate one molecule's canonical MOs and export research artifacts."""
    if args.all or args.molecule is None:
        raise SystemExit("--export-mos requires exactly one --molecule NAME.")
    if args.molecule not in molecules:
        raise SystemExit(f"Molecule {args.molecule!r} not found.")
    if args.mo_window < 0:
        raise SystemExit("--mo-window cannot be negative.")

    from .orbitals import export_orbital_bundle, orbital_table, run_orbital_calculation

    print(
        f"[SCF] {args.molecule} | METHOD={args.mo_method.upper()} | "
        f"BASIS={args.basis}",
        flush=True,
    )
    calculation = run_orbital_calculation(
        args.molecule,
        molecules[args.molecule],
        basis=args.basis,
        method=args.mo_method,
        xc=args.xc,
    )
    paths = export_orbital_bundle(
        calculation,
        output_dir=args.mo_output_dir,
        indices=args.mo_indices,
        occupied_below=args.mo_window,
        virtual_above=args.mo_window,
        grid=args.mo_grid,
        export_cubes=not args.no_mo_cubes,
    )

    table = orbital_table(calculation)
    occupied = table.loc[table["occupation"] > 1e-8]
    virtual = table.loc[table["occupation"] <= 1e-8]
    homo = int(occupied.iloc[-1]["mo_index"]) if not occupied.empty else None
    lumo = int(virtual.iloc[0]["mo_index"]) if not virtual.empty else None
    print(
        f"[DONE] converged={calculation.converged} | "
        f"E={calculation.total_energy:.12f} Ha | HOMO={homo} | LUMO={lumo}",
        flush=True,
    )
    print(f"[MO] all-orbital Molden -> {paths['molden']}", flush=True)
    print(f"[MO] energy table       -> {paths['table']}", flush=True)
    print(f"[MO] energy diagram     -> {paths['diagram']}", flush=True)
    if paths["cubes"]:
        print(
            f"[MO] {len(paths['cubes'])} cube file(s), indices "
            f"{paths['selected_indices']} -> {paths['directory']}",
            flush=True,
        )


def main(argv=None):
    args = build_parser().parse_args(argv)

    utility_modes = (
        args.export_xyz,
        args.generate_active_spaces,
        args.analyze_active_spaces,
        args.export_mos,
    )
    if sum(utility_modes) > 1:
        raise SystemExit(
            "Choose only one utility mode: --export-xyz, "
            "--generate-active-spaces, --analyze-active-spaces, or --export-mos."
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

    if args.export_mos:
        _run_orbital_export(args)
        return

    _apply_config(args)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Keep CUDA-Q and the VQE engine optional for geometry-only CLI use.
    from .driver import run_one_molecule, run_all_molecules

    if args.all:
        if args.active_space_rule != "energy":
            raise SystemExit(f"--active-space-rule {args.active_space_rule} runs one --molecule at a time.")
        run_all_molecules(molecules, out_dir=out_dir, integrals_dir=args.integrals)
        return

    if args.molecule is None:
        raise SystemExit("Provide --molecule NAME or --all.")

    molecule_name = args.molecule
    if molecule_name not in molecules:
        raise ValueError(f"Molecule '{molecule_name}' not found!")

    spec = dict(molecules[molecule_name])

    if args.active_space_rule != "energy":
        spec["valid_active_spaces"] = chosen_active_spaces(args, molecule_name, spec)
        for s in spec["valid_active_spaces"]:
            print(f"[SPACE] {args.active_space_rule}: {space_line(s)}", flush=True)
    elif args.space_idx is not None:
        spaces = spec.get("valid_active_spaces", [])
        if args.space_idx < 0 or args.space_idx >= len(spaces):
            raise ValueError(f"--space_idx {args.space_idx} out of range")
        spec["valid_active_spaces"] = [spaces[args.space_idx]]

    print(
        f"[RUN] {molecule_name} | BASIS={config.BASIS} | TARGET={config.TARGET} "
        f"| PRECISION={config.TARGET_PRECISION or 'default'} | OPT={config.OPTIMIZER}"
        f"{' | INTEGRALS=' + args.integrals if args.integrals else ''}"
        f" | ACTIVE SPACE RULE={args.active_space_rule}",
        flush=True,
    )

    mol_res = run_one_molecule(molecule_name, spec, integrals_dir=args.integrals)

    tag       = time.strftime("%d_%b_%Y").upper()
    mol_clean = sanitize_name(molecule_name)
    file_name = (
        f"{tag}_{mol_clean}_{sanitize_name(config.BASIS)}_"
        f"{sanitize_name(config.TARGET)}_{sanitize_name(config.OPTIMIZER)}"
        f"{'' if args.active_space_rule == 'energy' else '_' + sanitize_name(args.active_space_rule)}_VQE_results.pkl"
    )
    out_path = os.path.join(out_dir, file_name)
    if args.active_space_rule != "energy":
        mol_res["active_space_rule"] = args.active_space_rule
    save_pkl({molecule_name: mol_res}, out_path)
    print(f"[DONE] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
