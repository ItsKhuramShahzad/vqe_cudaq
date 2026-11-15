"""Independent checks of the bond-type CASCI results (CSV files and Excel summaries).

For each molecule: row counts against the CSV files, no space twice, left-out spaces do not hold
the orbital they leave out, no degenerate set split, E_corr, % of CCSD, Excel formulas of ratio,
passes and E_best present (the number passing is computed here), type counts add up to the orbitals, and a fresh CASCI
for a few spaces (the best ones with and without a left-out orbital and some random ones).

    python scripts/check_bond_type_casci.py --outdir analysis/bond_type_casci_hpc --molecule Ethylene Benzene
"""

import argparse
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from pyscf import mcscf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules



def index_of(name, nocc):
    if name.startswith("HOMO"):
        return nocc - 1 - (int(name[5:]) if name != "HOMO" else 0)
    return nocc + (int(name[5:]) if name != "LUMO" else 0)


def orbitals(text, nocc):
    return sorted(index_of(t, nocc) for t in re.sub(r"[\[\]|]|split", " ", text).split())


def counted_total(text):
    return sum(int(n) for n in re.findall(r"\((\d+)\)\s*(?:\+|$)", text)) if isinstance(text, str) else 0


def check(name, outdir, basis, n_casci, max_memory):
    base = os.path.join(outdir, f"{name}_bond_type_casci")
    sheets = pd.read_excel(base + "_summary.xlsx", sheet_name=None)
    main_csv = pd.read_csv(base + ".csv", encoding="utf-8-sig")
    left_csv = pd.read_csv(base + "_left_out.csv", encoding="utf-8-sig")
    mf = run_scf(molecules[name], basis, max_memory)
    nocc = mf.mol.nelectron // 2
    e = np.asarray(mf.mo_energy)
    degenerate = [set(np.flatnonzero(abs(e - v) < 1e-5)) for v in e]

    a = sheets["all spaces"].fillna({"left out": ""})
    col = lambda start: next(c for c in a if c.startswith(start))
    E, P = col("E_corr"), col("%")
    problems = []

    if (a["left out"] == "").sum() != main_csv["active"].nunique():
        problems.append("main row count")
    if (a["left out"] != "").sum() != len(left_csv):
        problems.append("left-out row count")
    spaces = [tuple(orbitals(t, nocc)) for t in a["active orbitals"]]
    if len(set(spaces)) != len(spaces):
        problems.append(f"{len(spaces) - len(set(spaces))} spaces twice")
    for i, (s, r) in enumerate(zip(spaces, a.to_dict("records"))):
        row = i + 2
        occ, vir = [k for k in s if k < nocc], [k for k in s if k >= nocc]
        if 2 * len(occ) != r["electrons"] or len(s) != r["orbitals"]:
            problems.append(f"size, row {row}")
        if counted_total(r["occupied types (number of orbitals)"]) != len(occ):
            problems.append(f"occupied count, row {row}")
        if counted_total(r["virtual types (number of orbitals)"]) != len(vir):
            problems.append(f"virtual count, row {row}")
        if abs(r[E] - 1e3 * (r["E_CASCI (Ha)"] - r["E_HF (Ha)"])) > 2e-3:
            problems.append(f"E_corr, row {row}")
        if abs(r[P] - 100 * r[E] / r["CCSD E_corr (mHa)"]) > 0.011:
            problems.append(f"% of CCSD, row {row}")
        if any(not degenerate[k] <= set(s) for k in s) and "split" not in r["active orbitals"]:
            problems.append(f"degenerate set split, row {row}")
        if r["left out"]:
            out = [index_of(t, nocc) for t in r["left out"].split("/")]
            if any(k in s for k in out) or "split" in r["active orbitals"]:
                problems.append(f"left-out orbital in the space, row {row}")
    # ratio, passes and E_best are Excel formulas reading the summary sheet: check the formulas
    from openpyxl import load_workbook
    wb = load_workbook(base + "_summary.xlsx", read_only=True)
    summary = {r[0].value: r[1].value for r in wb["summary"].iter_rows(max_col=2) if r[0].value}
    head = [c.value for c in next(wb["all spaces"].iter_rows(max_row=1))]
    row2 = [c.value for c in next(wb["all spaces"].iter_rows(min_row=2, max_row=2))]
    cells = dict(zip(head, row2))
    threshold = next((v for k, v in summary.items() if str(k).startswith("Threshold")), None)
    e_best_cell = next((v for k, v in summary.items() if str(k).startswith("E_best")), "")
    ratio_cell = next((v for k, v in cells.items() if str(k).startswith("ratio")), "")
    pass_cell = next((v for k, v in cells.items() if str(k).startswith("passes")), "")
    if not isinstance(threshold, (int, float)):
        problems.append("threshold cell is not a number")
    if not str(e_best_cell).startswith("=MIN('all spaces'!"):
        problems.append("E_best formula")
    if "/ABS(summary!$B$" not in str(ratio_cell) or not str(pass_cell).startswith("=IF("):
        problems.append("ratio or passes formula")
    best = a[E].min()                                   # best of the molecule, all sizes
    n_pass = int(((a[E] - best) / abs(best) < (threshold or 0)).sum())
    passing = sheets.get(f"passes ratio < {threshold:g}") if isinstance(threshold, (int, float)) else None
    if passing is None or len(passing) != n_pass or not passing[E].is_monotonic_increasing \
            or (passing[E] > (1 - threshold) * best + 1e-9).any():
        problems.append("passes sheet")
    if "best per size" in sheets or "without HOMO-k or LUMO+k" in sheets:
        problems.append("old sheet still there")

    picks = list(a[a["left out"] != ""].nsmallest(2, E).index) + list(a[a["left out"] == ""].nsmallest(2, E).index)
    picks += random.Random(1).sample(range(len(a)), max(0, n_casci - len(picks)))
    worst = 0.0
    for i in picks:
        s = spaces[i]
        cas = mcscf.CASCI(mf, len(s), int(a.loc[i, "electrons"]))
        cas.verbose = 0
        worst = max(worst, abs(cas.kernel(cas.sort_mo([k + 1 for k in s]))[0] - a.loc[i, "E_CASCI (Ha)"]))
    if worst > 1e-6:
        problems.append(f"fresh CASCI differs by {worst:.1e} Ha")

    left = a[a["left out"] != ""]["left out"].value_counts().to_dict()
    print(f"{name}: {len(a)} spaces = {(a['left out'] == '').sum()} main + {len(left_csv)} left out {left}; "
          f"pass {n_pass} (ratio < {threshold}); fresh CASCI for {len(picks)} spaces, largest difference "
          f"{worst:.1e} Ha; problems: {problems[:6] or 'none'}", flush=True)
    return not problems


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="+", required=True)
    p.add_argument("--outdir", default=os.path.join("analysis", "bond_type_casci"))
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--casci", type=int, default=6, help="spaces recomputed with a fresh CASCI")
    p.add_argument("--max-memory", type=int, default=6000)
    args = p.parse_args()
    ok = [check(m, args.outdir, args.basis, args.casci, args.max_memory) for m in args.molecule]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
