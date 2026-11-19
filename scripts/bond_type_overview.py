"""One Excel file with one row per molecule, taken from the "summary" sheet of each
<molecule>_bond_type_casci_summary.xlsx (made by bond_type_casci.py).

    python scripts/bond_type_overview.py --outdir analysis/bond_type_casci_hpc
"""

import argparse
import os

import pandas as pd

MOLECULES = ["Ethylene", "Methanamide", "NH2-", "Benzene", "Naphthalene", "Tetracene",
             "Pentacene", "Adenine", "Thymine", "Guanine", "Cytosine", "Uracil"]

COLUMNS = {                                   # summary item -> overview column
    "Electrons": "electrons",
    "Basis functions": "basis functions",
    "Occupied orbitals (core frozen + valence)": "occupied (core + valence)",
    "E_HF (Ha)": "E_HF (Ha)",
    "CCSD E_corr, whole molecule, frozen core (mHa)": "CCSD E_corr (mHa)",
    "Different active spaces, type combinations": "spaces, type combinations",
    "Different active spaces, HOMO-k or LUMO+k left out": "spaces, left out",
    "Different active spaces, total": "spaces, total",
    "E_best = best E_corr of the molecule, all sizes (mHa)": "E_best (mHa)",
    "   size": "best size",
    "   % of CCSD E_corr": "best % of CCSD",
    "   occupied types": "best occupied types",
    "   virtual types": "best virtual types",
    "   active orbitals": "best active orbitals",
    "   left out": "best: left out",
    "   vs energy order of the same size (mHa)": "best vs energy order (mHa)",
    "   same as": "passes when",
    "Spaces that pass": "spaces that pass",
    "Check: largest |S2| of a CASCI state (0 = singlet)": "largest |S2|",
    "Check: spaces that split a degenerate set": "split degenerate sets",
    "Check: E_corr = (E_CASCI − E_HF) × 1000 in every row": "E_corr formula holds",
    "Run: host": "HPC node",
    "Run: wall time (min)": "wall time (min)",
    "Run: SCF + orbital types + CCSD (s)": "SCF + types + CCSD (s)",
    "Run: type-combination CASCI (s)": "type-combination CASCI (s)",
    "Run: left-out CASCI (s)": "left-out CASCI (s)",
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--outdir", default=os.path.join("analysis", "bond_type_casci_hpc"))
    args = p.parse_args()
    rows = []
    for m in MOLECULES:
        path = os.path.join(args.outdir, f"{m}_bond_type_casci_summary.xlsx")
        if not os.path.exists(path):
            print(f"{m}: no summary file, skipped")
            continue
        s = pd.read_excel(path, sheet_name="summary").dropna(subset=["item"])
        values = dict(zip(s["item"], s["value"]))
        # E_best and the number passing are formulas in the file: compute them from the E_corr column
        spaces = pd.read_excel(path, sheet_name="all spaces")
        e_corr = spaces[next(c for c in spaces if c.startswith("E_corr"))]
        threshold = float(next(v for k, v in values.items() if str(k).startswith("Threshold")))
        best = e_corr.min()
        values["E_best = best E_corr of the molecule, all sizes (mHa)"] = round(best, 3)
        values["Spaces that pass"] = int(((e_corr - best) / abs(best) < threshold).sum())
        values["   same as"] = f"ratio < {threshold:g}: E_corr ≤ {round((1 - threshold) * best, 3)} mHa"
        rows.append({"molecule": m, **{new: values.get(old, "") for old, new in COLUMNS.items()}})
    table = pd.DataFrame(rows)
    out = os.path.join(args.outdir, "all_molecules_overview.xlsx")
    with pd.ExcelWriter(out) as xl:
        table.to_excel(xl, sheet_name="overview", index=False)
        ws = xl.sheets["overview"]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(len(str(c.value or "")) for c in col) + 2)
    print(f"{len(table)} molecules -> {out}")


if __name__ == "__main__":
    main()
