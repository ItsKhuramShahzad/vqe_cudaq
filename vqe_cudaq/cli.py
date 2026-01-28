import argparse
import os
import pickle
from datetime import datetime

from .molecules import molecules
from .runner import run_molecule_all_spaces


def _save(results_obj, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, f"{stamp}_{tag}_vqe.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(results_obj, f)
    print(f"[saved] {outpath}")
    return outpath


def _interactive_pick(prompt, options, default=None):
    """
    Show numbered options; return chosen value.
    """
    print(f"\n{prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    if default is not None:
        print(f"Press Enter for default: {default}")

    while True:
        ans = input("Select option number: ").strip()
        if ans == "" and default is not None:
            return default
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice. Try again.")


def _interactive_int(prompt, default):
    while True:
        ans = input(f"{prompt} [default={default}]: ").strip()
        if ans == "":
            return default
        if ans.isdigit():
            return int(ans)
        print("Please enter an integer.")


def _interactive_str(prompt, default):
    ans = input(f"{prompt} [default={default}]: ").strip()
    return default if ans == "" else ans


def main():
    mol_names = sorted(list(molecules.keys()))

    p = argparse.ArgumentParser(
        description="VQE (CUDA-Q + OpenFermion/PySCF) runner"
    )
    mode = p.add_mutually_exclusive_group()

    mode.add_argument("--interactive", action="store_true",
                      help="Ask prompts to choose molecule/target/etc.")
    mode.add_argument("--molecule", type=str,
                      help="Run one molecule by name (e.g., Methylene)")
    mode.add_argument("--all", action="store_true",
                      help="Run all molecules with all defined active spaces")

    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--target", default="nvidia",
                   help="cudaq target: nvidia (GPU) or qpp (CPU simulator)")
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--method", default="COBYLA")
    p.add_argument("--outdir", default="results")

    args = p.parse_args()

    # -------- Interactive mode --------
    if args.interactive:
        run_mode = _interactive_pick(
            "What do you want to run?",
            options=["One molecule", "All molecules"],
            default="One molecule",
        )

        basis = _interactive_str("Basis set", args.basis)
        target = _interactive_str("CUDA-Q target (nvidia/qpp)", args.target)
        maxiter = _interactive_int("Max optimizer iterations", args.maxiter)
        method = _interactive_str("Optimizer method (COBYLA, Nelder-Mead, etc.)", args.method)
        outdir = _interactive_str("Output folder", args.outdir)

        if run_mode == "One molecule":
            molecule = _interactive_pick(
                "Select molecule:",
                options=mol_names,
                default="Methylene" if "Methylene" in mol_names else mol_names[0],
            )
            mol = molecules[molecule]
            runs = run_molecule_all_spaces(
                molecule, mol, basis=basis, target=target, maxiter=maxiter, method=method
            )
            _save(runs, outdir, tag=f"{molecule}_{basis}_{target}")

        else:  # All molecules
            all_results = {}
            for molecule in mol_names:
                mol = molecules[molecule]
                print(f"\n=== Running {molecule} ===")
                all_results[molecule] = run_molecule_all_spaces(
                    molecule, mol, basis=basis, target=target, maxiter=maxiter, method=method
                )
            _save(all_results, outdir, tag=f"ALL_{basis}_{target}")

        return

    # -------- Non-interactive (CLI) mode --------
    if args.all:
        all_results = {}
        for molecule in mol_names:
            mol = molecules[molecule]
            print(f"\n=== Running {molecule} ===")
            all_results[molecule] = run_molecule_all_spaces(
                molecule, mol, basis=args.basis, target=args.target,
                maxiter=args.maxiter, method=args.method
            )
        _save(all_results, args.outdir, tag=f"ALL_{args.basis}_{args.target}")
        return

    if args.molecule:
        if args.molecule not in molecules:
            raise ValueError(
                f"Molecule '{args.molecule}' not found. Available: {', '.join(mol_names)}"
            )
        mol = molecules[args.molecule]
        runs = run_molecule_all_spaces(
            args.molecule, mol, basis=args.basis, target=args.target,
            maxiter=args.maxiter, method=args.method
        )
        _save(runs, args.outdir, tag=f"{args.molecule}_{args.basis}_{args.target}")
        return

    # Default behavior if no mode chosen:
    print("No mode selected. Use one of:")
    print("  --interactive")
    print("  --molecule <name>")
    print("  --all")
    print("\nExample:")
    print("  python -m vqe_cudaq.cli --interactive")


if __name__ == "__main__":
    main()
