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
"left out". Selection criterion, in every sheet: ratio = (E_corr - E_best) / |E_best|, where E_best is
the best E_corr of the same size; a space passes when the ratio is below RATIO_MAX (0.25).

    python scripts/bond_type_casci.py --molecule Benzene --max-electrons 8 --max-orbitals 10
    python scripts/bond_type_casci.py --molecule Ethylene --leave-out 1 2 3 --leave-out-virtual 1 2 3
    python scripts/bond_type_casci.py --molecule Ethylene Methanamide NH2- Benzene Naphthalene \
        Adenine Thymine Guanine Cytosine Uracil
"""

import argparse
import glob
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from pyscf import cc, mcscf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import BOND_SHARE_MIN, NON_VALENCE, classify_orbitals

HARTREE_TO_EV = 27.211386245988
ALL = "all"


def orbital_type(shares, occupied):
    """Largest bond share; a virtual orbital with no bond share above BOND_SHARE_MIN is "no bond"."""
    bonds = {n: w for n, w in shares.items() if n != NON_VALENCE}
    if not bonds or (not occupied and max(bonds.values()) < BOND_SHARE_MIN):
        return NON_VALENCE
    return max(bonds, key=bonds.get)


def groups(table, rows, nearest_first):
    """Degenerate sets as lists of orbital numbers, nearest the gap first."""
    out = {}
    for _, r in rows.iterrows():
        out.setdefault(int(r["degenerate_group"]), []).append(int(r["index"]))
    keys = sorted(out, reverse=nearest_first == "down")
    return [out[k] for k in keys]


FAMILIES = {
    "all π": lambda t: t.endswith(" π") or "lone pair (π)" in t,
    "all X–X σ": lambda t: t.endswith(" σ") and "–H" not in t,
    "all X–H σ": lambda t: t.endswith(" σ") and "–H" in t,
    "all lone pairs": lambda t: "lone pair" in t,
    "all π*": lambda t: t.endswith(" π*"),
    "all X–X σ*": lambda t: t.endswith(" σ*") and "–H" not in t,
    "all X–H σ*": lambda t: t.endswith(" σ*") and "–H" in t,
}


def type_sets(types, max_union):
    """{label: set of types}: each type, unions up to ``max_union`` types, families of two or more
    types and pairs of families, and all types together. Sets with the same types appear once."""
    out = {}
    for n in range(1, min(max_union, len(types)) + 1):
        for c in itertools.combinations(sorted(types), n):
            out[" + ".join(c)] = frozenset(c)
    families = {name: frozenset(t for t in types if test(t)) for name, test in FAMILIES.items()}
    families = {k: v for k, v in families.items() if len(v) >= 2}
    out.update(families)
    for (a, sa), (b, sb) in itertools.combinations(families.items(), 2):
        out[f"{a} + {b}"] = sa | sb
    out[" + ".join(sorted(types))] = frozenset(types)
    unique = {}
    for k, v in out.items():
        unique.setdefault(v, k)                     # the first (shortest) label of each set
    return {k: v for v, k in unique.items()}


def cumulative(group_list, limit):
    """(number of orbitals, orbitals) after taking 1, 2, ... groups, up to ``limit`` orbitals."""
    out, taken = [], []
    for g in group_list:
        if len(taken) + len(g) > limit:
            break
        taken = taken + g
        out.append(list(taken))
    return out


def casci(mf, active, nelec, cache):
    key = tuple(sorted(active))
    if key not in cache:
        cas = mcscf.CASCI(mf, len(active), nelec)
        cas.verbose = 0
        e = cas.kernel(cas.sort_mo([k + 1 for k in key]))[0]
        s2 = cas.fcisolver.spin_square(cas.ci, len(active), nelec)[0]
        cache[key] = (float(e), float(s2))
    return cache[key]


DEGENERATE_HA = 1e-5        # orbital energies closer than this are one degenerate set


def degenerate_sets(mo_energy):
    """Orbital number -> its degenerate set (sorted orbital numbers), for sets of two or more."""
    order = np.argsort(mo_energy)
    sets, current = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if mo_energy[b] - mo_energy[a] < DEGENERATE_HA:
            current.append(int(b))
        else:
            sets.append(current)
            current = [int(b)]
    sets.append(current)
    return {k: sorted(s) for s in sets if len(s) > 1 for k in s}


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
RATIO = "ratio = (E_corr − best E_corr of the same size) ÷ |best E_corr|"
PASS = f"passes (ratio < {RATIO_MAX})"
VS_BEST = "vs best of same size (mHa) = E_corr − best E_corr of the same size in the \"all spaces\" sheet"


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


def add_ratio(frame, best_by_size):
    """The selection criterion: how far each space is below the best space of its size."""
    best = frame["size"].map(best_by_size)
    ratio = (frame[E_CORR] - best) / best.abs()
    frame[RATIO] = ratio.round(4)
    frame[PASS] = np.where(ratio < RATIO_MAX, "yes", "no")
    return frame


def left_out_sheet(path, bracketed, e_hf):
    """Spaces with one HOMO-k or LUMO+k left out: one row per space, best first in each size."""
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
                     VS_BEST: None if pd.isna(r.vs_best_with_it_mHa) else round(r.vs_best_with_it_mHa, 3),
                     "active orbitals": shown, "degenerate sets": groups})
    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(dict.fromkeys(data["left_out"]))}
    return out.sort_values(["left out", "electrons", "orbitals", E_CORR],
                           key=lambda c: c.map(order) if c.name == "left out" else c)


def summarize(csv_path, xlsx_path, mf, molecule):
    """Excel file for reading: one row per different active space, labelled by the types it holds,
    degenerate sets in [brackets], the best space of each size, and the published VQE runs."""
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
        spaces = pd.concat([spaces, left_rows.drop(columns=[VS_BEST])], ignore_index=True)
    spaces.insert(1 if "left out" in spaces else 0, "skipped occupied (not in space)",
                  [skipped_occupied(t, nocc) for t in spaces["active orbitals"]])
    at = list(spaces.columns).index("skipped occupied (not in space)") + 1
    spaces.insert(at, "skipped virtual (not in space)",
                  [skipped_virtual(t, nocc) for t in spaces["active orbitals"]])
    best_by_size = spaces.groupby("size")[E_CORR].min()
    spaces = add_ratio(spaces, best_by_size)
    if left_rows is not None:
        left_rows.insert(1, "skipped occupied (not in space)",
                         [skipped_occupied(t, nocc) for t in left_rows["active orbitals"]])
        left_rows.insert(2, "skipped virtual (not in space)",
                         [skipped_virtual(t, nocc) for t in left_rows["active orbitals"]])
        left_rows = add_ratio(left_rows, best_by_size)
    best = spaces.loc[spaces.groupby(["electrons", "orbitals"])[E_CORR].idxmin()]
    best = best.sort_values(["electrons", "orbitals"])
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
        sheets = [("all spaces", spaces), ("best per size", best)]
        if vqe_rows:
            sheets.append(("VQE (paper spaces)", pd.DataFrame(vqe_rows)))
        if left_rows is not None:
            sheets.append(("without HOMO-k or LUMO+k", left_rows))
        for sheet, frame in sheets:
            frame.to_excel(xl, sheet_name=sheet, index=False)
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                text = [str(c.value or "") for c in col if not str(c.value or "").startswith("=")]
                ws.column_dimensions[col[0].column_letter].width = min(70, max(map(len, text or [""])) + 2)
    return len(spaces)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="+", default=["Benzene"])
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--max-electrons", type=int, default=8)
    p.add_argument("--max-orbitals", type=int, default=10)
    p.add_argument("--max-union", type=int, default=2, help="largest number of single bond types mixed on one side")
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
            print(f"{summarize(path, xlsx, mf, name)} different active spaces -> {xlsx}")
        else:
            path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
            leave = args.leave_out or args.leave_out_virtual
            if not leave or not os.path.exists(path):      # the main run, unless it is already done
                run_molecule(name, args)
            if leave:
                run_left_out(name, args)


def type_spaces(mf, table, occ, vir, args, cache, e_corr_total, name):
    """One row per (occupied type set, virtual type set, size): the CASCI energy of the space."""
    occ_sets = {**type_sets(sorted(set(occ["type"])), args.max_union), ALL: None}
    vir_sets = {**type_sets(sorted(set(vir["type"])), args.max_union), ALL: None}
    max_occ = args.max_electrons // 2

    rows = []
    for (o_label, os_), (v_label, vs) in itertools.product(occ_sets.items(), vir_sets.items()):
        o_rows = occ if os_ is None else occ[occ["type"].isin(os_)]
        v_rows = vir if vs is None else vir[vir["type"].isin(vs)]
        o_choices = cumulative(groups(table, o_rows, "down"), max_occ)
        v_choices = cumulative(groups(table, v_rows, "up"), args.max_orbitals)
        for o_list, v_list in itertools.product(o_choices, v_choices):
            if len(o_list) + len(v_list) > args.max_orbitals:
                continue
            active = o_list + v_list
            nelec = 2 * len(o_list)
            e, s2 = casci(mf, active, nelec, cache)
            rows.append(dict(
                molecule=name, occupied_types=o_label, virtual_types=v_label,
                nele=nelec, norb=len(active), E_casci=e, E_corr_mHa=1e3 * (e - mf.e_tot),
                percent_of_ccsd=100 * (e - mf.e_tot) / e_corr_total, ccsd_corr_mHa=1e3 * e_corr_total, S2=s2,
                active=" ".join(str(k) for k in sorted(active)),
                names=" ".join(table.loc[sorted(active), "name"]),
                types=" | ".join(table.loc[sorted(active), "type"])))
    return rows


def run_molecule(name, args):
    t0 = time.time()
    mf = run_scf(molecules[name], args.basis, args.max_memory)
    table = classify_orbitals(mf)
    ncore, nocc = table.attrs["ncore"], table.attrs["nocc"]
    table["type"] = [orbital_type(d, o) if not c else "core"
                     for d, o, c in zip(table["bond_shares"], table["occupied"], table["core"])]
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]

    # whole-molecule correlation energy, frozen core like the active spaces
    ccsd = cc.CCSD(mf, frozen=ncore)
    ccsd.max_memory = args.max_memory
    ccsd.verbose = 0
    e_corr_total = float(ccsd.kernel()[0])
    print(f"{name}: E_HF {mf.e_tot:.8f} Ha, CCSD correlation (frozen core {ncore}) {e_corr_total * 1e3:.3f} mHa, "
          f"{time.time() - t0:.0f} s", flush=True)

    cache = {}
    rows = type_spaces(mf, table, occ, vir, args, cache, e_corr_total, name)
    data = pd.DataFrame(rows)
    ref = data[(data["occupied_types"] == ALL) & (data["virtual_types"] == ALL)]
    ref = ref.set_index(["nele", "norb"])["E_corr_mHa"]
    data["gain_vs_energy_order_mHa"] = [r.E_corr_mHa - ref[(r.nele, r.norb)] if (r.nele, r.norb) in ref.index
                                        else np.nan for r in data.itertuples()]
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
    data.round(6).to_csv(path, index=False, encoding="utf-8-sig")   # the marker lets Excel read π, σ, –
    print(f"{len(data)} active spaces ({len(cache)} different), largest |S2| {data['S2'].abs().max():.1e}, "
          f"{time.time() - t0:.0f} s -> {path}", flush=True)
    write_summary(path, mf, name)


def write_summary(path, mf, name):
    xlsx = path.replace(".csv", "_summary.xlsx")
    try:
        print(f"{summarize(path, xlsx, mf, name)} different active spaces -> {xlsx}", flush=True)
    except Exception as err:            # e.g. no openpyxl on the cluster: the CSV files are complete
        print(f"Excel summary not written ({err!r}); rebuild it later with --summary-only", flush=True)


def run_left_out(name, args):
    """The same type spaces with one orbital left out, with its whole degenerate set: HOMO-k for k
    in ``args.leave_out`` and LUMO+k for k in ``args.leave_out_virtual``. Only spaces not already in
    the main CSV are kept."""
    t0 = time.time()
    path = os.path.join(args.outdir, f"{name}_bond_type_casci.csv")
    main_data = pd.read_csv(path, encoding="utf-8-sig")
    e_corr_total = main_data["ccsd_corr_mHa"].iloc[0] / 1e3
    mf = run_scf(molecules[name], args.basis, args.max_memory)
    table = classify_orbitals(mf)
    nocc = table.attrs["nocc"]
    table["type"] = [orbital_type(d, o) if not c else "core"
                     for d, o, c in zip(table["bond_shares"], table["occupied"], table["core"])]
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]
    sets = degenerate_sets(np.asarray(mf.mo_energy))
    ref = main_data[(main_data["occupied_types"] == ALL) & (main_data["virtual_types"] == ALL)]
    ref = ref.set_index(["nele", "norb"])["E_corr_mHa"]
    best = main_data.groupby(["nele", "norb"])["E_corr_mHa"].min()
    seen = set(main_data["active"])
    rows, cache, done = [], {}, set()
    targets = [nocc - 1 - k for k in args.leave_out if nocc - 1 - k in set(occ["index"])] + \
              [nocc + k for k in args.leave_out_virtual if nocc + k in set(vir["index"])]
    for orbital in targets:
        removed = tuple(sets.get(orbital, [orbital]))
        if removed in done:                     # e.g. HOMO-3 in the same degenerate set as HOMO-2
            continue
        done.add(removed)
        label = "/".join(orbital_name(m, nocc) for m in sorted(removed, reverse=orbital < nocc))
        o_keep, v_keep = occ[~occ["index"].isin(removed)], vir[~vir["index"].isin(removed)]
        for r in type_spaces(mf, table, o_keep, v_keep, args, cache, e_corr_total, name):
            if r["active"] in seen:
                continue
            seen.add(r["active"])
            size = (r["nele"], r["norb"])
            r["left_out"] = label
            r["gain_vs_energy_order_mHa"] = r["E_corr_mHa"] - ref[size] if size in ref.index else np.nan
            r["vs_best_with_it_mHa"] = r["E_corr_mHa"] - best[size] if size in best.index else np.nan
            rows.append(r)
        print(f"  without {label}: {len(rows)} new spaces so far, {time.time() - t0:.0f} s", flush=True)
    out = path.replace(".csv", "_left_out.csv")
    pd.DataFrame(rows).round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{name}: {len(rows)} new spaces, {time.time() - t0:.0f} s -> {out}", flush=True)
    write_summary(path, mf, name)


if __name__ == "__main__":
    main()
