"""CASCI in active spaces made of chosen bond types, compared by the correlation energy they recover.

Each canonical orbital gets a type from its bond shares (orbital_selection.classify_orbitals):
occupied orbitals by their largest bond share, virtual orbitals by their largest bond share leaving
out the part outside the valence space, or "no bond (outside valence)" when no bond share reaches BOND_SHARE_MIN (0.1). An active space takes occupied orbitals of some types and virtual orbitals
of some types, nearest the HOMO-LUMO gap first, degenerate orbitals always together (a set that
does not fit is never split: the space stops before it). The type sets are each type alone, pairs
of types, families (all π, all X–X σ, all X–H σ, all lone pairs, and the antibonds) and pairs of
families. "all" takes every orbital in energy order (the usual choice), for comparison at the
same size. The published VQE runs (results/) are added where a space has the same orbitals.

The correlation energy of the whole molecule (frozen core CCSD) gives the percentage each space
recovers.

With --leave-out / --leave-out-virtual the same type spaces are also built without HOMO-k or
LUMO+k (with its degenerate set); the new spaces go at the end of the "all spaces" sheet, marked in
"left out". Selection criterion (Excel formulas, threshold editable in the summary sheet):
ratio = (E_corr - E_best) / |E_best|, where E_best is
the best E_corr of the molecule (all sizes, all spaces); a space passes when the ratio is below
RATIO_MAX (0.25).

    python scripts/bond_type_casci.py --molecule Benzene --max-electrons 8 --max-orbitals 10
    python scripts/bond_type_casci.py --molecule Ethylene --leave-out 1 2 3 --leave-out-virtual 1 2 3
    python scripts/bond_type_casci.py --molecule Ethylene Methanamide NH2- Benzene Naphthalene \
        Adenine Thymine Guanine Cytosine Uracil
"""

import argparse
import datetime
import glob
import itertools
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.bond_type_search import (ALL, FAMILIES, casci, cumulative, degenerate_sets,  # noqa: F401
                                        frozen_core_ccsd, groups, left_out_rows, orbital_type,
                                        type_sets, type_spaces, typed_orbitals, with_energy_order_gain)

HARTREE_TO_EV = 27.211386245988


def vqe_runs(molecule, root="results"):
    """The published VQE runs of one molecule (results/cpu, else results/gpu), keyed by orbital set."""
    for backend in ("cpu", "gpu"):
        for path in sorted(glob.glob(os.path.join(root, backend, f"*_{molecule}_*.pkl"))):
            with open(path, "rb") as fh:
                (name, res), = pickle.load(fh).items()
            out = {}
            for r in res["active_space_runs"]:
                if not r.get("skipped"):
                    key = tuple(sorted(int(k) for k in r["active_indices"]["active_indices"]))
                    out[key] = dict(nele=r["space"]["nele_cas"], norb=r["space"]["norb_cas"],
                                    E_casci=r["casci"]["E_casci_total"], E_vqe=r["vqe"]["E_total"],
                                    backend=backend, file=os.path.basename(path))
            return out
    return {}


def orbital_name(k, nocc):
    return ("HOMO" + (f"-{nocc - 1 - k}" if k < nocc - 1 else "")) if k < nocc else \
           ("LUMO" + (f"+{k - nocc}" if k > nocc else ""))


# Excel headers that say how the column is computed
E_CORR = "E_corr (mHa) = (E_CASCI − E_HF) × 1000"
PERCENT = "% of CCSD = E_corr ÷ CCSD E_corr × 100"
VQE_GAP = "VQE − CASCI (mHa) = (E_VQE − E_CASCI) × 1000"
E_CORR_VQE = "E_corr VQE (mHa) = (E_VQE − E_HF) × 1000"
def counted(types):
    """ "C–C π (1) + C–H σ (2)": each type with the number of orbitals of that type."""
    n = {}
    for x in types:
        n[x] = n.get(x, 0) + 1
    return " + ".join(f"{x} ({n[x]})" for x in sorted(n))


VS_ORDER = "vs energy order (mHa) = E_corr − E_corr of the energy-ordered space of the same size"
RATIO_MAX = 0.25
RATIO = "ratio = (E_corr − E_best) ÷ |E_best|"
BEST_ALL = "E_best (mHa) = best E_corr of the molecule, all sizes"
PASS = "passes (ratio < threshold in summary)"
THRESHOLD = "Threshold: a space passes when ratio < (change this value)"
E_BEST = "E_best = best E_corr of the molecule, all sizes (mHa)"


def skipped_occupied(shown, nocc):
    """Occupied orbitals between HOMO and the deepest active one that are not in the space."""
    names = shown.replace("[", " ").replace("]", " ").replace("|", " ").replace("split", " ").split()
    occ = {nocc - 1 - (int(n[5:]) if n != "HOMO" else 0) for n in names if n.startswith("HOMO")}
    if not occ:
        return ""
    return " ".join(orbital_name(k, nocc) for k in range(nocc - 1, min(occ) - 1, -1) if k not in occ)


def skipped_virtual(shown, nocc):
    """Virtual orbitals between LUMO and the highest active one that are not in the space."""
    names = shown.replace("[", " ").replace("]", " ").replace("|", " ").replace("split", " ").split()
    vir = {nocc + (int(n[5:]) if n != "LUMO" else 0) for n in names if n.startswith("LUMO")}
    if not vir:
        return ""
    return " ".join(orbital_name(k, nocc) for k in range(nocc, max(vir) + 1) if k not in vir)


def add_ratio(frame, best, ratio_max=RATIO_MAX):
    """The selection criterion: how far each space is below the best space of the molecule
    (the most negative E_corr of all spaces, of any size). In the Excel file these three columns
    become formulas that read E_best and the threshold from the summary sheet."""
    ratio = (frame[E_CORR] - best) / abs(best)
    frame[BEST_ALL] = round(best, 3)
    frame[RATIO] = ratio.round(4)
    frame[PASS] = np.where(ratio < ratio_max, "yes", "no")
    return frame


def column_letter(ws, heading):
    from openpyxl.utils import get_column_letter
    for cell in ws[1]:
        if cell.value == heading:
            return get_column_letter(cell.column)
    raise KeyError(heading)


def add_formulas(xl, n_rows):
    """Ratio, passes and E_best as Excel formulas, so the threshold can be changed in the file:
    the summary holds the threshold (a number) and E_best (=MIN of the E_corr column)."""
    summary, spaces = xl.sheets["summary"], xl.sheets["all spaces"]
    where = {c.value: c.row for c in summary["A"] if c.value}
    eb, th = where[E_BEST], where[THRESHOLD]
    E, B, R, P = (column_letter(spaces, h) for h in (E_CORR, BEST_ALL, RATIO, PASS))
    last = n_rows + 1
    summary[f"B{eb}"] = f"=MIN('all spaces'!{E}2:{E}{last})"
    summary[f"B{where['   same as']}"] = (f'="E_corr ≤ "&TEXT((1-B{th})*B{eb},"0.000")&" mHa (at least "'
                                          f'&ROUND(100*(1-B{th}),2)&"% of E_best)"')
    summary[f"B{where['Spaces that pass']}"] = f"=COUNTIF('all spaces'!{P}2:{P}{last},\"yes\")"
    for r in range(2, last + 1):
        spaces[f"{B}{r}"] = f"=summary!$B${eb}"
        spaces[f"{R}{r}"] = f"=({E}{r}-summary!$B${eb})/ABS(summary!$B${eb})"
        spaces[f"{P}{r}"] = f'=IF({R}{r}<summary!$B${th},"yes","no")'
        spaces[f"{R}{r}"].number_format = "0.0000"
    spaces.auto_filter.ref = spaces.dimensions          # filter arrows: passes = yes, sort by E_corr


def left_out_sheet(path, bracketed, e_hf):
    """Rows for the spaces with one HOMO-k or LUMO+k left out (added at the end of "all spaces")."""
    data = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for r in data.itertuples():
        names, types = r.names.split(), r.types.split(" | ")
        shown, groups = bracketed([int(k) for k in r.active.split()])
        rows.append({"left out": r.left_out, "size": f"({r.nele}e,{r.norb}o)", "electrons": r.nele,
                     "orbitals": r.norb,
                     "occupied types (number of orbitals)": counted([t for n, t in zip(names, types) if n.startswith("HOMO")]),
                     "virtual types (number of orbitals)": counted([t for n, t in zip(names, types) if n.startswith("LUMO")]),
                     "E_CASCI (Ha)": round(r.E_casci, 8), "E_HF (Ha)": round(e_hf, 8),
                     E_CORR: round(r.E_corr_mHa, 3), "CCSD E_corr (mHa)": round(r.ccsd_corr_mHa, 3),
                     PERCENT: round(r.percent_of_ccsd, 2),
                     VS_ORDER: None if pd.isna(r.gain_vs_energy_order_mHa) else round(r.gain_vs_energy_order_mHa, 3),
                     "active orbitals": shown, "degenerate sets": groups})
    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(dict.fromkeys(data["left_out"]))}
    return out.sort_values(["left out", "electrons", "orbitals", E_CORR],
                           key=lambda c: c.map(order) if c.name == "left out" else c)


def now():
    return datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")


def record_run(csv_path, **fields):
    """Add fields to <molecule>_bond_type_casci_run.json next to the CSV (laptop or cluster)."""
    path = csv_path.replace(".csv", "_run.json")
    info = json.load(open(path)) if os.path.exists(path) else {}
    info.update(fields)
    with open(path, "w") as fh:
        json.dump(info, fh, indent=1)


def settings(args):
    """The search options of this run, for the summary sheet."""
    return {"setting: max electrons": args.max_electrons, "setting: max orbitals": args.max_orbitals,
            "setting: max virtual": args.max_virtual or "same as max orbitals",
            "setting: max types mixed on one side": args.max_union,
            "setting: bond types used": ", ".join(args.types) if args.types else "all",
            "setting: outside-valence virtuals": "not used" if args.no_outside_valence else "used",
            "setting: left out HOMO-k, k =": " ".join(map(str, args.leave_out)) or "-",
            "setting: left out LUMO+k, k =": " ".join(map(str, args.leave_out_virtual)) or "-"}


def machine():
    """Computer, threads and the git commit of this code."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=here).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--", "scripts/bond_type_casci.py", "vqe_cudaq"],
                                 capture_output=True, text=True, cwd=os.path.dirname(here)).stdout.strip()
        if commit and changed:
            commit += " + uncommitted changes"
    except OSError:
        commit = ""
    from pyscf import lib
    return {"host": socket.gethostname(), "threads": lib.num_threads(), "commit": commit or "unknown"}


def run_info(outdir, molecule):
    """Run details: the record written by this script, else the cluster log (older runs)."""
    path = os.path.join(outdir, f"{molecule}_bond_type_casci_run.json")
    if os.path.exists(path):
        info = json.load(open(path))
        when = lambda t: datetime.datetime.strptime(t, "%a %b %d %H:%M:%S %Y")
        if "start" in info and "end" in info:
            info["wall time (min)"] = round((when(info["end"]) - when(info["start"])).total_seconds() / 60, 1)
        info["record file"] = os.path.basename(path)
        return info
    return run_log(outdir, molecule)


def run_log(outdir, molecule):
    """Times and host of the run from the cluster log (outdir/logs/*.log), if there is one."""
    for path in sorted(glob.glob(os.path.join(outdir, "logs", "*.log"))):
        text = open(path, errors="replace").read()
        if f"molecule: {molecule} " not in text:
            continue
        out = {"log file": os.path.basename(path)}
        m = re.search(r"host: (\S+)\s+commit: (\S+)\s+start: (.+)", text)
        if m:
            out["host"], out["commit"], out["start"] = m.group(1), m.group(2), m.group(3).strip()
        m = re.search(r"^end: (.+)$", text, re.M)
        if m:
            out["end"] = m.group(1).strip()
        when = lambda t: datetime.datetime.strptime(" ".join(t.split()[:4] + t.split()[5:6]), "%a %b %d %H:%M:%S %Y")
        if "start" in out and "end" in out:
            out["wall time (min)"] = round((when(out["end"]) - when(out["start"])).total_seconds() / 60, 1)
        m = re.search(r"CCSD correlation \(frozen core (\d+)\) \S+ mHa, (\d+) s", text)
        if m:
            out["frozen core orbitals"], out["SCF + orbital types + CCSD (s)"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"active spaces \((\d+) different\), largest \|S2\| (\S+), (\d+) s", text)
        if m and "SCF + orbital types + CCSD (s)" in out:
            out["type-combination CASCI (s)"] = int(m.group(3)) - out["SCF + orbital types + CCSD (s)"]
        m = re.search(rf"^{re.escape(molecule)}: (\d+) new spaces, (\d+) s", text, re.M)
        if m:
            out["left-out CASCI (s)"] = int(m.group(2))
        return out
    return {}


def summary_sheet(molecule, mf, data, left, spaces, e_total, outdir, ratio_max=RATIO_MAX):
    """One sheet with the totals, the checks and the time taken."""
    from pyscf.data.elements import chemcore
    mol, nocc = mf.mol, mf.mol.nelectron // 2
    ncore = chemcore(mol)
    best = spaces.loc[spaces[E_CORR].idxmin()]
    passing = spaces[spaces[PASS] == "yes"]
    order = best.get(VS_ORDER)
    s2 = [data["S2"].abs().max()] + ([left["S2"].abs().max()] if left is not None and len(left) else [])
    left_counts = left["left_out"].value_counts() if left is not None and len(left) else pd.Series(dtype=int)
    log = run_info(outdir, molecule)
    rows = [
        ("Molecule", molecule),
        ("Basis", mol.basis),
        ("Electrons", mol.nelectron),
        ("Basis functions", mol.nao),
        ("Occupied orbitals (core frozen + valence)", f"{nocc} ({ncore} + {nocc - ncore})"),
        ("Virtual orbitals", mol.nao - nocc),
        ("", ""),
        ("E_HF (Ha)", round(mf.e_tot, 8)),
        ("CCSD E_corr, whole molecule, frozen core (mHa)", round(e_total, 3)),
        ("E_CCSD total = E_HF + CCSD E_corr (Ha)", round(mf.e_tot + e_total / 1e3, 8)),
        ("", ""),
        ("Largest active space", f"({data['nele'].max()}e,{data['norb'].max()}o)"),
        ("Different active spaces, type combinations", data["active"].nunique()),
        ("Different active spaces, HOMO-k or LUMO+k left out", 0 if left is None else len(left)),
        ("   of which without", ", ".join(f"{k}: {v}" for k, v in left_counts.items()) or "-"),
        ("Different active spaces, total", len(spaces)),
        ("Sizes (electrons, orbitals)", spaces["size"].nunique()),
        ("", ""),
        (E_BEST, round(best[E_CORR], 3)),
        ("   size", best["size"]),
        ("   % of CCSD E_corr", round(best[PERCENT], 2)),
        ("   occupied types", best["occupied types (number of orbitals)"]),
        ("   virtual types", best["virtual types (number of orbitals)"]),
        ("   active orbitals", best["active orbitals"]),
        ("   left out", best.get("left out", "") or "-"),
        ("   vs energy order of the same size (mHa)", "-" if pd.isna(order) or order is None else round(order, 3)),
        ("", ""),
        ("Selection criterion", "ratio = (E_corr − E_best) ÷ |E_best|"),
        (THRESHOLD, ratio_max),
        ("   same as", f"E_corr ≤ {1 - ratio_max:g} × E_best = {round(best[E_CORR] * (1 - ratio_max), 3)} mHa"
                      f" (at least {100 * (1 - ratio_max):g}% of E_best)"),
        ("Spaces that pass", len(passing)),
        ("   list of them", f"sheet 'passes ratio < {ratio_max:g}', best first (made with this threshold); "
                           f"for another threshold: 'all spaces', filter 'passes' = yes"),
        ("", ""),
        ("Check: largest |S2| of a CASCI state (0 = singlet)", float(f"{max(s2):.1e}")),
        ("Check: spaces that split a degenerate set", int(spaces["active orbitals"].str.contains("split").sum())),
        ("Check: E_corr = (E_CASCI − E_HF) × 1000 in every row",
         "yes" if np.allclose(spaces[E_CORR], 1e3 * (spaces["E_CASCI (Ha)"] - spaces["E_HF (Ha)"]), atol=2e-3) else "NO"),
        ("Full independent check", "python scripts/check_bond_type_casci.py --outdir <folder> --molecule " + molecule),
        ("", ""),
    ]
    if log:
        rows += [("Run: " + k, v) for k, v in log.items()]
    else:
        rows += [("Run", "no run record (_run.json) or cluster log found")]
    return pd.DataFrame(rows, columns=["item", "value"])


def summarize(csv_path, xlsx_path, mf, molecule, ratio_max=RATIO_MAX):
    """Excel file for reading: one row per different active space, labelled by the types it holds,
    degenerate sets in [brackets], the selection ratio against the best space of the molecule, the
    spaces with a HOMO-k or LUMO+k left out, and the published VQE runs."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    mo_energy, nocc, e_hf = np.asarray(mf.mo_energy), mf.mol.nelectron // 2, mf.e_tot
    e_total = data["ccsd_corr_mHa"].iloc[0]                                           # CCSD, mHa
    sets = degenerate_sets(mo_energy)
    vqe = vqe_runs(molecule)

    def bracketed(index):
        label = {k: orbital_name(k, nocc) for k in index}
        shown, groups, done = [], [], set()
        for k in sorted(index):
            if k in done:
                continue
            whole = sets.get(k, [k])
            members = [m for m in whole if m in label]
            done.update(members)
            if len(whole) > 1 and len(members) < len(whole):
                shown.append("[" + " ".join(label[m] for m in members) + " | split]")
                groups.append("/".join(orbital_name(m, nocc) for m in whole) + " (split)")
            elif len(members) > 1:
                shown.append("[" + " ".join(label[m] for m in members) + "]")
                groups.append("/".join(label[m] for m in members))
            else:
                shown.append(label[k])
        return " ".join(shown), ", ".join(groups)
    rows = []
    for r in data.drop_duplicates("active").itertuples():
        names, types = r.names.split(), r.types.split(" | ")
        index = [int(k) for k in r.active.split()]
        occ = [t for n, t in zip(names, types) if n.startswith("HOMO")]
        vir = [t for n, t in zip(names, types) if n.startswith("LUMO")]
        shown, groups = bracketed(index)
        rows.append({"size": f"({r.nele}e,{r.norb}o)", "electrons": r.nele, "orbitals": r.norb,
                     "occupied types (number of orbitals)": counted(occ),
                     "virtual types (number of orbitals)": counted(vir),
                     "E_CASCI (Ha)": round(r.E_casci, 8), "E_HF (Ha)": round(e_hf, 8),
                     E_CORR: round(r.E_corr_mHa, 3), "CCSD E_corr (mHa)": round(e_total, 3),
                     PERCENT: round(r.percent_of_ccsd, 2),
                     VS_ORDER: (None if pd.isna(r.gain_vs_energy_order_mHa) else round(r.gain_vs_energy_order_mHa, 3)),
                     "E_VQE (Ha)": round(vqe[tuple(sorted(index))]["E_vqe"], 8) if tuple(sorted(index)) in vqe else None,
                     VQE_GAP: (round(1e3 * (vqe[tuple(sorted(index))]["E_vqe"] - r.E_casci), 4)
                               if tuple(sorted(index)) in vqe else None),
                     "active orbitals": shown, "degenerate sets": groups})
    spaces = pd.DataFrame(rows).sort_values(["electrons", "orbitals", E_CORR])
    left = csv_path.replace(".csv", "_left_out.csv")
    left_rows = left_out_sheet(left, bracketed, e_hf) if os.path.exists(left) else None
    if left_rows is not None:          # the spaces without a HOMO-k or LUMO+k go at the end, marked
        spaces.insert(0, "left out", "")
        spaces = pd.concat([spaces, left_rows], ignore_index=True)
    spaces.insert(1 if "left out" in spaces else 0, "skipped occupied (not in space)",
                  [skipped_occupied(t, nocc) for t in spaces["active orbitals"]])
    at = list(spaces.columns).index("skipped occupied (not in space)") + 1
    spaces.insert(at, "skipped virtual (not in space)",
                  [skipped_virtual(t, nocc) for t in spaces["active orbitals"]])
    best_all = spaces[E_CORR].min()                 # main and left-out spaces together
    spaces = add_ratio(spaces, best_all, ratio_max)
    if left_rows is not None:
        left_rows.insert(1, "skipped occupied (not in space)",
                         [skipped_occupied(t, nocc) for t in left_rows["active orbitals"]])
        left_rows.insert(2, "skipped virtual (not in space)",
                         [skipped_virtual(t, nocc) for t in left_rows["active orbitals"]])
        left_rows = add_ratio(left_rows, best_all, ratio_max)
    types = {}
    for r in data.drop_duplicates("active").itertuples():
        names_, types_ = r.names.split(), r.types.split(" | ")
        index = tuple(sorted(int(k) for k in r.active.split()))
        types[index] = (counted([t for n, t in zip(names_, types_) if n.startswith("HOMO")]),
                        counted([t for n, t in zip(names_, types_) if n.startswith("LUMO")]))
    vqe_rows = []
    for index, v in sorted(vqe.items(), key=lambda kv: (-kv[1]["nele"], kv[1]["norb"])):
        shown, groups = bracketed(list(index))
        vqe_rows.append({"size": f"({v['nele']}e,{v['norb']}o)", "active orbitals (paper, by energy)": shown,
                         "degenerate sets": groups, "E_VQE (Ha)": round(v["E_vqe"], 8),
                         "E_CASCI (Ha)": round(v["E_casci"], 8), "E_HF (Ha)": round(e_hf, 8),
                         VQE_GAP: round(1e3 * (v["E_vqe"] - v["E_casci"]), 4),
                         E_CORR_VQE: round(1e3 * (v["E_vqe"] - e_hf), 3), "CCSD E_corr (mHa)": round(e_total, 3),
                         "% of CCSD (VQE) = E_corr VQE ÷ CCSD E_corr × 100": round(100 * 1e3 * (v["E_vqe"] - e_hf) / e_total, 2),
                         "same orbitals as bond-type space": " | ".join(types[index]) if index in types else "",
                         "source": f"results/{v['backend']}/{v['file']}"})
    with pd.ExcelWriter(xlsx_path) as xl:
        def arranged(frame):                    # orbitals first, the skipped/left-out marks last
            first = [c for c in ("active orbitals", "degenerate sets") if c in frame]
            last = [c for c in ("left out", "skipped occupied (not in space)", "skipped virtual (not in space)")
                    if c in frame]
            return frame[first + [c for c in frame if c not in first + last] + last]
        spaces = arranged(spaces)
        passing = spaces[spaces[PASS] == "yes"].sort_values(E_CORR, kind="stable")    # best first
        if left_rows is not None:
            left_rows = arranged(left_rows)
        left_csv = pd.read_csv(left, encoding="utf-8-sig") if left_rows is not None else None
        summary = summary_sheet(molecule, mf, data, left_csv, spaces, e_total, os.path.dirname(csv_path), ratio_max)
        sheets = [("summary", summary), ("all spaces", spaces), (f"passes ratio < {ratio_max:g}", passing)]
        if vqe_rows:
            sheets.append(("VQE (paper spaces)", pd.DataFrame(vqe_rows)))
        for sheet, frame in sheets:
            frame.to_excel(xl, sheet_name=sheet, index=False)
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                text = [str(c.value or "") for c in col if not str(c.value or "").startswith("=")]
                ws.column_dimensions[col[0].column_letter].width = min(70, max(map(len, text or [""])) + 2)
        add_formulas(xl, len(spaces))
    return len(spaces)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="+", default=["Benzene"])
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--max-electrons", type=int, default=8)
    p.add_argument("--max-orbitals", type=int, default=10)
    p.add_argument("--max-virtual", type=int, default=None, help="largest number of virtual orbitals in a space")
    p.add_argument("--max-union", type=int, default=2, help="largest number of single bond types mixed on one side")
    p.add_argument("--types", nargs="+", default=None,
                   help='use only these bond types (occupied and virtual), e.g. "C–C π" "C–C π*"')
    p.add_argument("--no-outside-valence", action="store_true",
                   help='do not use "no bond (outside valence)" virtual orbitals')
    p.add_argument("--ratio-max", type=float, default=RATIO_MAX,
                   help="selection threshold written into the Excel file (it can be changed there)")
    p.add_argument("--outdir", default=os.path.join("analysis", "bond_type_casci"))
    p.add_argument("--leave-out", nargs="+", type=int, default=[],
                   help="also run the type spaces without HOMO-k (with its degenerate set), e.g. 1 2 3")
    p.add_argument("--leave-out-virtual", nargs="+", type=int, default=[],
                   help="also run the type spaces without LUMO+k (with its degenerate set), e.g. 1 2 3")
    p.add_argument("--max-memory", type=int, default=16000, help="PySCF memory in MB (SCF, CCSD, CASCI)")
    p.add_argument("--summary-only", action="store_true", help="only rebuild the Excel summary from the CSV")
    args = p.parse_args()
    for name in args.molecule:
        if args.summary_only:
            path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
            xlsx = path.replace(".csv", "_summary.xlsx")
            mf = run_scf(molecules[name], args.basis, args.max_memory)
            print(f"{summarize(path, xlsx, mf, name, args.ratio_max)} different active spaces -> {xlsx}")
        else:
            path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
            leave = args.leave_out or args.leave_out_virtual
            if not leave or not os.path.exists(path):      # the main run, unless it is already done
                run_molecule(name, args)
            if leave:
                run_left_out(name, args)


def run_molecule(name, args):
    t0 = time.time()
    start = now()
    mf = run_scf(molecules[name], args.basis, args.max_memory)
    table = typed_orbitals(mf)
    ncore, nocc = table.attrs["ncore"], table.attrs["nocc"]
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]
    if args.types:
        unknown = set(args.types) - set(table["type"])
        if unknown:
            print(f"{name}: no orbital of type {sorted(unknown)}; types here: {sorted(set(table['type']) - {'core'})}",
                  flush=True)

    # whole-molecule correlation energy, frozen core like the active spaces
    e_corr_total = frozen_core_ccsd(mf, ncore, args.max_memory)
    t_ccsd = time.time() - t0
    print(f"{name}: E_HF {mf.e_tot:.8f} Ha, CCSD correlation (frozen core {ncore}) {e_corr_total * 1e3:.3f} mHa, "
          f"{t_ccsd:.0f} s", flush=True)

    cache = {}
    rows = type_spaces(mf, table, occ, vir, args, cache, e_corr_total, name)
    data = with_energy_order_gain(rows)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
    data.round(6).to_csv(path, index=False, encoding="utf-8-sig")   # the marker lets Excel read π, σ, –
    print(f"{len(data)} active spaces ({len(cache)} different), largest |S2| {data['S2'].abs().max():.1e}, "
          f"{time.time() - t0:.0f} s -> {path}", flush=True)
    if os.path.exists(path.replace(".csv", "_run.json")):
        os.remove(path.replace(".csv", "_run.json"))            # a new main run starts a new record
    record_run(path, **machine(), **settings(args), start=start, end=now(), **{
        "frozen core orbitals": int(ncore), "SCF + orbital types + CCSD (s)": round(t_ccsd),
        "type-combination CASCI (s)": round(time.time() - t0 - t_ccsd)})
    write_summary(path, mf, name, args.ratio_max)


def write_summary(path, mf, name, ratio_max=RATIO_MAX):
    xlsx = path.replace(".csv", "_summary.xlsx")
    try:
        print(f"{summarize(path, xlsx, mf, name, ratio_max)} different active spaces -> {xlsx}", flush=True)
    except Exception as err:            # e.g. no openpyxl on the cluster: the CSV files are complete
        print(f"Excel summary not written ({err!r}); rebuild it later with --summary-only", flush=True)


def run_left_out(name, args):
    """The same type spaces with one orbital left out, with its whole degenerate set: HOMO-k for k
    in ``args.leave_out`` and LUMO+k for k in ``args.leave_out_virtual``. Only spaces not already in
    the main CSV are kept."""
    t0 = time.time()
    start = now()
    path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
    main_data = pd.read_csv(path, encoding="utf-8-sig")
    e_corr_total = main_data["ccsd_corr_mHa"].iloc[0] / 1e3
    mf = run_scf(molecules[name], args.basis, args.max_memory)
    table = typed_orbitals(mf)
    rows = left_out_rows(mf, table, args, {}, e_corr_total, name, main_data,
                         progress=lambda label, n: print(f"  without {label}: {n} new spaces so far, "
                                                         f"{time.time() - t0:.0f} s", flush=True))
    out = path.replace(".csv", "_left_out.csv")
    pd.DataFrame(rows).round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{name}: {len(rows)} new spaces, {time.time() - t0:.0f} s -> {out}", flush=True)
    first = {} if os.path.exists(path.replace(".csv", "_run.json")) else {**machine(), "start": start}
    record_run(path, **first, end=now(), **{"left-out CASCI (s)": round(time.time() - t0)})
    write_summary(path, mf, name, args.ratio_max)


if __name__ == "__main__":
    main()
