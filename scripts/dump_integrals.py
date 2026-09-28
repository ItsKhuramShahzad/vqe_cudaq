"""
Dump active-space integrals (and CCSD amplitudes) for all molecules, active
spaces and basis sets. One .npz per active space:

    <outdir>/<basis>/<molecule>/No#_<i>_No(c)_<ncore>_Ne(a)_<nele_cas>_No(a)_<norb_cas>.npz

Each file holds the active-space integrals (h1, eri, e_core), E_HF, E_CASCI and,
by default, the full-molecule CCSD amplitudes cut to the active space, so a VQE
run with ``--integrals`` needs nothing but the file. The integrals come from
PySCF CASCI (get_h1eff / get_h2eff), which only needs norb_cas^4 memory; see
:mod:`vqe_cudaq.hamiltonian`. The large molecules (Tetracene, Pentacene in
cc-pVDZ) need a compute node with a few hundred GB of memory.

Errors in one molecule do NOT stop the other molecules.

    python scripts/dump_integrals.py
    python scripts/dump_integrals.py --basis cc-pVDZ --molecule Ethylene
    python scripts/dump_integrals.py --basis cc-pVDZ --molecule Pentacene --max-memory 200000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.molecules import molecules
from vqe_cudaq.hamiltonian import (
    SpaceDoesNotFit,
    compute_active_space,
    integral_file_name,
    run_ccsd,
    run_scf,
    save_active_space,
)


def dump_molecule(mol_name, basis, outdir, max_memory=16000, with_ccsd=True):
    """Write the integral files of every active space of one molecule."""
    mol = molecules[mol_name]
    t0 = time.time()
    mf = run_scf(mol, basis, max_memory)
    print(f"  SCF E={mf.e_tot:.10f} nmo={mf.mo_coeff.shape[1]} {time.time() - t0:.0f}s",
          flush=True)

    ccsd = None
    if with_ccsd:
        t_cc = time.time()
        ccsd = run_ccsd(mf, max_memory)
        print(f"  CCSD E={ccsd[0]:.10f} {time.time() - t_cc:.0f}s", flush=True)

    mol_dir = os.path.join(outdir, basis, mol_name.replace(" ", "_"))
    os.makedirs(mol_dir, exist_ok=True)

    for i, space in enumerate(mol.get("valid_active_spaces", []), start=1):
        No_c, Ne_a, No_a = int(space["ncore"]), int(space["nele_cas"]), int(space["norb_cas"])
        name = integral_file_name(i, No_c, Ne_a, No_a)
        try:
            data = compute_active_space(mf, No_c, Ne_a, No_a, ccsd=ccsd)
        except SpaceDoesNotFit as exc:
            print(f"  [SKIP] {name}: {exc}", flush=True)
            continue
        save_active_space(os.path.join(mol_dir, name), data,
                          molecule=mol_name, basis=basis,
                          charge=int(mol["charge"]), multiplicity=int(mol["multiplicity"]))
        print(f"  [OK] {name}: {2 * No_a} qubits, E_CASCI={data['e_casci']:.10f}", flush=True)


def dump_integrals_all(
    bases=("cc-pVDZ", "sto-3g", "6-31g"),
    outdir="integrals",
    molecule_names=None,
    max_memory=16000,
    with_ccsd=True,
):
    """Dump the integrals for all (or the given) closed-shell molecules and bases."""
    names = molecule_names or [k for k, v in molecules.items() if int(v["multiplicity"]) == 1]
    for basis in bases:
        print(f"\n==============================")
        print(f" Basis set: {basis}")
        print(f"==============================")
        for mol_name in names:
            print(f"\n→ Molecule: {mol_name}", flush=True)
            try:
                dump_molecule(mol_name, basis, outdir, max_memory, with_ccsd)
            except Exception as exc:
                print(f"  [FAIL] {mol_name} {basis}: {exc!r}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--basis", nargs="+", default=["cc-pVDZ", "sto-3g", "6-31g"])
    p.add_argument("--molecule", nargs="*", default=None,
                   help="default: all closed-shell molecules")
    p.add_argument("--outdir", default="integrals")
    p.add_argument("--max-memory", "--max_memory", type=int, default=16000,
                   help="PySCF memory limit in MB")
    p.add_argument("--no-ccsd", action="store_true",
                   help="skip CCSD and store integrals only (no VQE starting point)")
    a = p.parse_args()
    dump_integrals_all(bases=a.basis, outdir=a.outdir, molecule_names=a.molecule,
                       max_memory=a.max_memory, with_ccsd=not a.no_ccsd)


if __name__ == "__main__":
    main()
