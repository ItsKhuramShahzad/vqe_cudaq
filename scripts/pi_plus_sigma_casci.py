"""CASCI in the π active space plus any choice of σ and σ* orbitals, skipping allowed.

The base space is every π orbital (π bonds and π lone pairs) with the same number of π* orbitals
nearest the gap. To it, any subset of the σ orbitals (occupied, not π, not core) and of the σ*
orbitals (virtual, type ending in σ*) nearest the gap is added; a subset may skip orbitals, e.g.
HOMO and HOMO-2 without HOMO-1. Degenerate orbitals are always taken together. Orbital types are
those of bond_type_casci.py (largest bond share; "no bond (outside valence)" virtuals are not σ*).

Each space is compared with the energy-ordered space of the same size (the usual choice) and
with the whole-molecule frozen-core CCSD correlation energy.

    python scripts/pi_plus_sigma_casci.py --molecule Ethylene
    python scripts/pi_plus_sigma_casci.py --molecule Ethylene --sigma-window 5 --sigma-star-window 6
    python scripts/pi_plus_sigma_casci.py --molecule Ethylene Benzene --sizes 6,7 6,8
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from pyscf import cc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bond_type_casci import (E_CORR, FAMILIES, PERCENT, casci, counted, groups, orbital_name,
                             orbital_type)
from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import classify_orbitals

VS_ORDER = "vs energy order (mHa) = E_corr − E_corr of the energy-ordered space of the same size"


def take(group_list, n):
    """Whole groups from the front until at least ``n`` orbitals are taken."""
    out = []
    for g in group_list:
        if len(out) >= n:
            break
        out += g
    return out


def skipped(chosen, pool):
    """Pool orbitals between the gap and the farthest chosen one that were left out."""
    flat = [k for g in pool for k in g]
    if not chosen:
        return []
    last = max(flat.index(k) for k in chosen)
    return [k for k in flat[:last] if k not in chosen]


def subsets(pool, limit):
    """Every subset of the groups in ``pool`` with at most ``limit`` orbitals, as flat lists."""
    out = []
    for n in range(len(pool) + 1):
        for c in itertools.combinations(pool, n):
            flat = [k for g in c for k in g]
            if len(flat) <= limit:
                out.append(flat)
    return out


def run_molecule(name, args):
    t0 = time.time()
    mf = run_scf(molecules[name], args.basis)
    table = classify_orbitals(mf)
    ncore, nocc = table.attrs["ncore"], table.attrs["nocc"]
    table["type"] = [orbital_type(d, o) if not c else "core"
                     for d, o, c in zip(table["bond_shares"], table["occupied"], table["core"])]
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]

    is_pi = FAMILIES["all π"]
    pi_occ = [k for g in groups(table, occ[occ["type"].map(is_pi)], "down") for k in g]
    pi_vir = take(groups(table, vir[vir["type"].map(FAMILIES["all π*"])], "up"), len(pi_occ))
    sigma = groups(table, occ[~occ["type"].map(is_pi)], "down")[:args.sigma_window]
    sigma_star = groups(table, vir[vir["type"].str.endswith(" σ*")], "up")[:args.sigma_star_window]
    base = pi_occ + pi_vir
    if 2 * len(pi_occ) > args.max_electrons or len(base) > args.max_orbitals:
        print(f"{name}: π space ({2 * len(pi_occ)}e,{len(base)}o) is already above the limits")
        return

    ccsd = cc.CCSD(mf, frozen=ncore)
    ccsd.verbose = 0
    e_ccsd = float(ccsd.kernel()[0])
    print(f"{name}: E_HF {mf.e_tot:.8f} Ha, CCSD correlation {e_ccsd * 1e3:.3f} mHa; π space "
          f"{' '.join(table.loc[sorted(base), 'name'])}; σ window "
          f"{' '.join(table.loc[[k for g in sigma for k in g], 'name'])}; σ* window "
          f"{' '.join(table.loc[[k for g in sigma_star for k in g], 'name'])}", flush=True)

    occ_all, vir_all = groups(table, occ, "down"), groups(table, vir, "up")
    cache, rows = {}, []
    occ_room = args.max_electrons // 2 - len(pi_occ)
    for s in subsets(sigma, occ_room):
        for ss in subsets(sigma_star, args.max_orbitals - len(base) - len(s)):
            active = sorted(base + s + ss)
            n_o, n_v = len(pi_occ) + len(s), len(pi_vir) + len(ss)
            if args.sizes and (2 * n_o, n_o + n_v) not in args.sizes:
                continue
            e, s2 = casci(mf, active, 2 * n_o, cache)
            # the energy-ordered space of the same size, if it does not split a degenerate set
            o_ref, v_ref = take(occ_all, n_o), take(vir_all, n_v)
            e_ref = casci(mf, o_ref + v_ref, 2 * n_o, cache)[0] \
                if (len(o_ref), len(v_ref)) == (n_o, n_v) else np.nan
            skip = skipped(s, sigma) + skipped(ss, sigma_star)
            rows.append({
                "size": f"({2 * n_o}e,{len(active)}o)", "electrons": 2 * n_o, "orbitals": len(active),
                "added σ": " ".join(orbital_name(k, nocc) for k in sorted(s, reverse=True)),
                "added σ*": " ".join(orbital_name(k, nocc) for k in sorted(ss)),
                "skipped (nearer the gap, left out)": " ".join(orbital_name(k, nocc) for k in sorted(skip)),
                "occupied types (number of orbitals)": counted(table.loc[[k for k in active if k < nocc], "type"]),
                "virtual types (number of orbitals)": counted(table.loc[[k for k in active if k >= nocc], "type"]),
                "E_CASCI (Ha)": round(e, 8), "E_HF (Ha)": round(mf.e_tot, 8),
                E_CORR: round(1e3 * (e - mf.e_tot), 3), "CCSD E_corr (mHa)": round(1e3 * e_ccsd, 3),
                PERCENT: round(100 * (e - mf.e_tot) / e_ccsd, 2),
                VS_ORDER: round(1e3 * (e - e_ref), 3) if np.isfinite(e_ref) else None,
                "S2": round(s2, 6),
                "active orbitals": " ".join(orbital_name(k, nocc) for k in active)})

    data = pd.DataFrame(rows).sort_values(["electrons", "orbitals", E_CORR])
    best = data.loc[data.groupby(["electrons", "orbitals"])[E_CORR].idxmin()]
    os.makedirs(args.outdir, exist_ok=True)
    tag = "".join(f"_{e}e{o}o" for e, o in sorted(args.sizes or []))
    csv = os.path.join(args.outdir, f"{name}_pi_plus_sigma_casci{tag}.csv")
    data.to_csv(csv, index=False, encoding="utf-8-sig")
    xlsx = csv.replace(".csv", ".xlsx")
    with pd.ExcelWriter(xlsx) as xl:
        for sheet, frame in (("all spaces", data), ("best per size", best),
                             ("with skipped orbitals", data[data["skipped (nearer the gap, left out)"] != ""])):
            frame.to_excel(xl, sheet_name=sheet, index=False)
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = \
                    min(70, max(len(str(c.value or "")) for c in col) + 2)
    print(f"{len(data)} active spaces, largest |S2| {data['S2'].abs().max():.1e}, "
          f"{time.time() - t0:.0f} s -> {xlsx}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", nargs="+", default=["Ethylene"])
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--max-electrons", type=int, default=8)
    p.add_argument("--max-orbitals", type=int, default=10)
    p.add_argument("--sigma-window", type=int, default=5, help="σ sets nearest the gap to choose from")
    p.add_argument("--sigma-star-window", type=int, default=6, help="σ* sets nearest the gap to choose from")
    p.add_argument("--sizes", nargs="+", type=lambda x: tuple(int(v) for v in x.split(",")),
                   help="only these sizes, as electrons,orbitals, e.g. --sizes 6,7 6,8")
    p.add_argument("--outdir", default=os.path.join("analysis", "pi_plus_sigma_casci"))
    args = p.parse_args()
    for name in args.molecule:
        run_molecule(name, args)


if __name__ == "__main__":
    main()
