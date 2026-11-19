"""CASCI energy of each orbital-selection rule, at the same active-space sizes (no VQE).

    python scripts/compare_active_spaces.py --all --out analysis/active_space_comparison.csv

One CSV row per molecule, active space and rule. The energy rule is also compared with the stored
integral file, which checks that the orbitals here are the ones the benchmark used.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from pyscf import mcscf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import load_integrals, run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import DegenerateSplitError, classify_orbitals, select_active_orbitals

RULES = {
    "energy": dict(rule="energy"),
    "pi": dict(rule="character", include=["pi"]),
    "pi+lone": dict(rule="character", include=["pi", "lone"]),
    "no-xh": dict(rule="combined", exclude=["xh"]),
}


def casci_energy(mf, active, nele_cas):
    cas = mcscf.CASCI(mf, len(active), nele_cas)
    cas.verbose = 0
    return float(cas.kernel(cas.sort_mo([k + 1 for k in active]))[0])     # sort_mo counts from 1


def run_molecule(name, args, writer):
    t0 = time.time()
    mf = run_scf(molecules[name], args.basis, max_memory=args.max_memory)
    table = classify_orbitals(mf)
    names = table.set_index("index")["name"]
    labels = table.set_index("index")["label"]
    for space in molecules[name]["valid_active_spaces"]:
        ncore, nele, norb = int(space["ncore"]), int(space["nele_cas"]), int(space["norb_cas"])
        energies = {}
        reference = None
        for rule, options in RULES.items():
            try:
                s = select_active_orbitals(table, nele, norb, **options)
            except DegenerateSplitError as err:
                writer.writerow(dict(molecule=name, nele_cas=nele, norb_cas=norb, qubits=2 * norb, rule=rule,
                                     E_casci="", dE_vs_energy_mHa="", active="", names="", labels="",
                                     fallbacks="", warnings=f"not possible: {err}"))
                continue
            key = tuple(s.active)
            energies.setdefault(key, casci_energy(mf, s.active, nele))
            energy = energies[key]
            if rule == "energy":
                reference = energy
                stored = load_integrals(args.integrals, args.basis, name, ncore, nele, norb)["e_casci"]
                if abs(stored - energy) > 1e-7:
                    raise RuntimeError(f"{name} ({nele},{norb}): {energy} here, {stored} in the file")
            writer.writerow(dict(
                molecule=name, nele_cas=nele, norb_cas=norb, qubits=2 * norb, rule=rule,
                E_casci=f"{energy:.9f}", dE_vs_energy_mHa=f"{1e3 * (energy - reference):.4f}",
                active=" ".join(str(k) for k in s.active),
                names=" ".join(names[k] for k in s.active),
                labels=" ".join(labels[k] for k in s.active),
                fallbacks="; ".join(s.fallbacks), warnings="; ".join(s.warnings)))
    print(f"[DONE] {name}: {time.time() - t0:.0f} s", flush=True)


def plot(csv_path, png_path, rules=("pi", "no-xh"), sizes=((6, 6), (6, 7))):
    """Energy change against the energy rule, one group of bars per molecule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    data = pd.read_csv(csv_path)
    names = [m for m in molecules if m in set(data["molecule"])]
    order = ["Ethylene", "Benzene", "Naphthalene", "Tetracene", "Pentacene", "NH2-", "Methanamide",
             "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine"]
    names = sorted(names, key=order.index)
    colours = {"pi": ("#1d4ed8", "#93c5fd"), "no-xh": ("#c2410c", "#fdba74")}
    titles = {"pi": "π and π* first", "no-xh": "energy order, skipping C–H / N–H"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True, sharey=True)
    for ax, rule in zip(axes, rules):
        for i, (nele, norb) in enumerate(sizes):
            part = data[(data["rule"] == rule) & (data["nele_cas"] == nele) & (data["norb_cas"] == norb)]
            values = part.set_index("molecule")["dE_vs_energy_mHa"].reindex(names)
            ax.bar(np.arange(len(names)) + (i - 0.5) * 0.38, values, 0.38, color=colours[rule][i],
                   label=f"({nele}e,{norb}o), {2 * norb} qubits")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(-1.6, color="grey", ls="--", lw=1)
        ax.set_ylabel("ΔE_CASCI vs energy rule (mHa)", fontsize=11)
        ax.set_title(titles[rule], fontsize=13, loc="left")
        ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11)
        unchanged = data[data["rule"] == rule].groupby("molecule")["dE_vs_energy_mHa"].apply(lambda v: v.abs().max() < 1e-3)
        for i, name in enumerate(names):
            if unchanged[name]:
                ax.text(i, 0.8, "no change", ha="center", va="bottom", fontsize=10, color="#475569")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--integrals", default="integrals")
    p.add_argument("--max-memory", type=int, default=12000)
    p.add_argument("--out", default=os.path.join("analysis", "active_space_comparison.csv"))
    p.add_argument("--plot-only", default=None, help="write this PNG from the existing --out CSV and stop")
    args = p.parse_args()
    if args.plot_only:
        plot(args.out, args.plot_only)
        return
    names = list(molecules) if args.all else (args.molecule or [])
    if not names:
        raise SystemExit("Provide --molecule NAME ... or --all.")
    fields = ["molecule", "nele_cas", "norb_cas", "qubits", "rule", "E_casci", "dE_vs_energy_mHa",
              "active", "names", "labels", "fallbacks", "warnings"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name in names:
            run_molecule(name, args, writer)
            fh.flush()


if __name__ == "__main__":
    main()
