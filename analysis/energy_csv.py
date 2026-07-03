# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Build a per-(molecule, active-space) energy comparison CSV from the
25_FEB_2026 CPU + GPU VQE pkl results.

Columns
-------
  Molecule
  Active Space               (Ne,No) = (nele_cas, norb_cas)
  %AS_wrt_Orbitals           norb_cas / Total Spatial Orbitals * 100
  %AS_wrt_Electrons          nele_cas / Total Electrons       * 100
  E_VQE_GPU - E_HF   [Ha]
  E_VQE_CPU - E_HF   [Ha]
  E_VQE_GPU - E_VQE_CPU [Ha]
  Correlation Energy         (E_VQE_GPU - E_HF) / E_VQE_GPU
"""

import os, sys, csv, glob, pickle

BASE = os.path.dirname(os.path.abspath(__file__))     # .../vqe_cudaq/analysis
ROOT = os.path.dirname(BASE)                          # repo root

# Point these at the CPU/GPU result folders for the run you want to tabulate.
CPU_DIR = os.path.join(ROOT, "results", "pkl_results", "cpu_pkl_results")
GPU_DIR = os.path.join(ROOT, "results", "pkl_results", "gpu_pkl_results")
OUT_CSV = os.path.join(BASE, "vqe_energy_table.csv")

# Only keep active spaces that are CURRENTLY listed in the molecule database
# (vqe_cudaq.molecules). Some pkls may have been generated from an older
# version with extra spaces; this filter self-corrects so the CSV always
# matches the active valid_active_spaces.
sys.path.insert(0, ROOT)                              # so `vqe_cudaq` is importable
from vqe_cudaq.molecules import molecules as MOL_DATA
ALLOWED_SPACES = {
    mol: {(s["nele_cas"], s["norb_cas"]) for s in spec.get("valid_active_spaces", [])}
    for mol, spec in MOL_DATA.items()
}

# canonical molecule order (matches the latex generator)
MOL_ORDER = [
    "Methylene", "Ethylene",
    "Benzene", "Naphthalene", "Benzaanthracene", "Pentacene",
    "Methanamide",""
    "Adenine", "Thymine", "Uracil", "Cytosine", "Guanine",
]


def parse_pkl(fp):
    """Return (mol_name, refs, runs) where runs is keyed by (ne, no)."""
    with open(fp, "rb") as fh:
        raw = pickle.load(fh)
    mol_key = next(iter(raw))
    mol     = raw[mol_key]
    inp     = mol.get("input_spec", {})
    ref     = mol.get("references", {})

    refs = {
        "E_hf"        : ref.get("E_hf_full"),
        "E_ccsd"      : ref.get("E_ccsd_full"),
        "n_electrons" : inp.get("Total Electrons"),
        "n_orbitals"  : inp.get("Total Spatial Orbitals"),
    }

    runs = {}
    for r in mol.get("active_space_runs", []):
        if r.get("skipped", False):
            continue
        sp  = r.get("space", {})
        vqe = r.get("vqe", {})
        ne  = sp.get("nele_cas")
        no  = sp.get("norb_cas")
        E   = vqe.get("E_total")          # active-space VQE total energy (2*No qubits)
        if ne is None or no is None or E is None:
            continue
        runs[(ne, no)] = {"ne": ne, "no": no, "ncore": sp.get("ncore"),
                          "E_vqe": E, "rt": vqe.get("runtime"),
                          "E_casci": r.get("casci", {}).get("E_casci_total")}
    return mol_key, refs, runs


def load_dir(dirpath):
    out = {}
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.pkl"))):
        mol_key, refs, runs = parse_pkl(fp)
        out[mol_key] = {"refs": refs, "runs": runs}
    return out


def main():
    cpu = load_dir(CPU_DIR)
    gpu = load_dir(GPU_DIR)

    matched = set(cpu) & set(gpu)
    order   = [m for m in MOL_ORDER if m in matched] + \
              sorted(m for m in matched if m not in MOL_ORDER)

    only_cpu = sorted(set(cpu) - set(gpu))
    only_gpu = sorted(set(gpu) - set(cpu))
    if only_cpu:
        print(f"  CPU-only (skipped): {only_cpu}")
    if only_gpu:
        print(f"  GPU-only (skipped): {only_gpu}")

    header = [
        "Molecule",
        "Active Space (Ne,No)",
        "%AS wrt Orbitals (No/Total Orbitals)",
        "%AS wrt Electrons (Ne/Total Electrons)",
        "E_VQE_GPU - E_HF [Ha]",
        "E_VQE_CPU - E_HF [Ha]",
        "E_VQE_GPU - E_VQE_CPU [Ha]",
        "Correlation Energy (E_VQE_GPU-E_HF)/E_VQE_GPU",
        "GPU Speedup (t_CPU/t_GPU)",
        "CCSD Recovery (E_VQE_GPU-E_HF)/(E_CCSD-E_HF)",
        "CASCI Recovery (E_VQE_GPU-E_HF)/(E_CASCI-E_HF)",
    ]

    rows = []
    for mol in order:
        c_refs, c_runs = cpu[mol]["refs"], cpu[mol]["runs"]
        g_refs, g_runs = gpu[mol]["refs"], gpu[mol]["runs"]

        E_hf       = g_refs["E_hf"] if g_refs["E_hf"] is not None else c_refs["E_hf"]
        E_ccsd     = g_refs.get("E_ccsd") if g_refs.get("E_ccsd") is not None else c_refs.get("E_ccsd")
        n_orb      = g_refs["n_orbitals"] or c_refs["n_orbitals"]
        n_ele      = g_refs["n_electrons"] or c_refs["n_electrons"]
        ccsd_corr  = (E_ccsd - E_hf) if (E_ccsd is not None and E_hf is not None) else None

        # keep active-space ordering as stored in the GPU pkl
        keys = list(g_runs.keys())
        for k in c_runs:                       # append any CPU-only spaces
            if k not in g_runs:
                keys.append(k)

        # restrict to active spaces still listed in molecules_data.py
        allowed = ALLOWED_SPACES.get(mol)
        if allowed is not None:
            dropped = [k for k in keys if k not in allowed]
            if dropped:
                print(f"  {mol}: dropped extra active spaces {dropped}")
            keys = [k for k in keys if k in allowed]

        for key in keys:
            ne, no = key
            gr = g_runs.get(key)
            cr = c_runs.get(key)
            E_gpu = gr["E_vqe"] if gr else None
            E_cpu = cr["E_vqe"] if cr else None
            rt_gpu = gr["rt"] if gr else None
            rt_cpu = cr["rt"] if cr else None
            # CASCI = exact diagonalization within the SAME active space (classical
            # reference, identical for CPU/GPU). Active-space-consistent denominator.
            E_casci = (gr["E_casci"] if gr and gr.get("E_casci") is not None
                       else (cr["E_casci"] if cr else None))

            pct_orb = 100.0 * no / n_orb if n_orb else None
            pct_ele = 100.0 * ne / n_ele if n_ele else None
            d_gpu_hf = (E_gpu - E_hf) if (E_gpu is not None and E_hf is not None) else None
            d_cpu_hf = (E_cpu - E_hf) if (E_cpu is not None and E_hf is not None) else None
            d_gpu_cpu = (E_gpu - E_cpu) if (E_gpu is not None and E_cpu is not None) else None
            corr = (d_gpu_hf / E_gpu) if (d_gpu_hf is not None and E_gpu) else None
            speedup = (rt_cpu / rt_gpu) if (rt_cpu and rt_gpu and rt_gpu > 0) else None
            ccsd_rec = (d_gpu_hf / ccsd_corr) if (d_gpu_hf is not None and ccsd_corr) else None
            casci_corr = (E_casci - E_hf) if (E_casci is not None and E_hf is not None) else None
            casci_rec = (d_gpu_hf / casci_corr) if (d_gpu_hf is not None and casci_corr) else None

            rows.append([
                mol,
                f"({ne},{no})",
                f"{pct_orb:.2f}"   if pct_orb   is not None else "",
                f"{pct_ele:.2f}"   if pct_ele   is not None else "",
                f"{d_gpu_hf:.8f}"  if d_gpu_hf  is not None else "",
                f"{d_cpu_hf:.8f}"  if d_cpu_hf  is not None else "",
                f"{d_gpu_cpu:.3e}" if d_gpu_cpu is not None else "",
                f"{corr:.8e}"      if corr      is not None else "",
                f"{speedup:.3f}"   if speedup   is not None else "",
                f"{ccsd_rec:.6f}"  if ccsd_rec  is not None else "",
                f"{casci_rec:.6f}" if casci_rec is not None else "",
            ])

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    print(f"\n  Wrote {len(rows)} rows for {len(order)} molecules")
    print(f"  -> {OUT_CSV}")


if __name__ == "__main__":
    main()
