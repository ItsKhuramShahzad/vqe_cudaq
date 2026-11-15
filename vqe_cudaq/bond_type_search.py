"""Search for active spaces by bond type (used by scripts/bond_type_casci.py and the VQE CLI).

Every canonical orbital gets a type from its bond shares (:func:`typed_orbitals`): occupied
orbitals by their largest bond share, virtual orbitals by their largest antibond share, or
"no bond (outside valence)" when no share reaches BOND_SHARE_MIN. A candidate space takes
occupied orbitals of some types and virtual orbitals of some types, nearest the HOMO-LUMO gap
first, degenerate orbitals always together; :func:`type_spaces` runs CASCI for every one.
:func:`left_out_rows` repeats the search with one HOMO-k or LUMO+k removed.

:func:`ensure_size` makes sure a molecule's result files hold the spaces of one size: if not,
only that size is searched (type sets and left-out runs) and appended to the CSV files, so the
VQE CLI can use the bond-type rule for any molecule and size without a separate run.
"""

import itertools
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pyscf import cc, mcscf

from .chosen_spaces import orbital_name
from .orbital_selection import BOND_SHARE_MIN, NON_VALENCE, classify_orbitals

ALL = "all"                     # the energy-ordered reference, not a bond type
DEGENERATE_HA = 1e-5            # orbital energies closer than this are one degenerate set
LEAVE_OUT = (1, 2, 3)           # HOMO-1, -2, -3 and LUMO+1, +2, +3, as in the published search

FAMILIES = {
    "all π": lambda t: t.endswith(" π") or "lone pair (π)" in t,
    "all X–X σ": lambda t: t.endswith(" σ") and "–H" not in t,
    "all X–H σ": lambda t: t.endswith(" σ") and "–H" in t,
    "all lone pairs": lambda t: "lone pair" in t,
    "all π*": lambda t: t.endswith(" π*"),
    "all X–X σ*": lambda t: t.endswith(" σ*") and "–H" not in t,
    "all X–H σ*": lambda t: t.endswith(" σ*") and "–H" in t,
}


def options(**kw):
    """Search options with the defaults of the published search; any can be given."""
    base = dict(max_electrons=8, max_orbitals=10, max_virtual=None, max_union=2, types=None,
                no_outside_valence=False, leave_out=list(LEAVE_OUT), leave_out_virtual=list(LEAVE_OUT),
                sizes=None)
    base.update(kw)
    return SimpleNamespace(**base)


def orbital_type(shares, occupied):
    """Largest bond share; a virtual orbital with no bond share above BOND_SHARE_MIN is "no bond"."""
    bonds = {n: w for n, w in shares.items() if n != NON_VALENCE}
    if not bonds or (not occupied and max(bonds.values()) < BOND_SHARE_MIN):
        return NON_VALENCE
    return max(bonds, key=bonds.get)


def typed_orbitals(mf):
    """classify_orbitals(mf) with a "type" column (``core`` for the frozen core)."""
    table = classify_orbitals(mf)
    table["type"] = [orbital_type(d, o) if not c else "core"
                     for d, o, c in zip(table["bond_shares"], table["occupied"], table["core"])]
    return table


def groups(table, rows, nearest_first):
    """Degenerate sets as lists of orbital numbers, nearest the gap first."""
    out = {}
    for _, r in rows.iterrows():
        out.setdefault(int(r["degenerate_group"]), []).append(int(r["index"]))
    keys = sorted(out, reverse=nearest_first == "down")
    return [out[k] for k in keys]


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
    """(E_CASCI, <S^2>) of one space; every space is computed once (``cache``)."""
    key = tuple(sorted(active))
    if key not in cache:
        cas = mcscf.CASCI(mf, len(active), nelec)
        cas.verbose = 0
        e = cas.kernel(cas.sort_mo([k + 1 for k in key]))[0]
        s2 = cas.fcisolver.spin_square(cas.ci, len(active), nelec)[0]
        cache[key] = (float(e), float(s2))
    return cache[key]


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


def frozen_core_ccsd(mf, ncore, max_memory=16000):
    """CCSD correlation energy of the whole molecule (Ha), core frozen like the active spaces."""
    ccsd = cc.CCSD(mf, frozen=ncore)
    ccsd.max_memory = max_memory
    ccsd.verbose = 0
    return float(ccsd.kernel()[0])


def type_spaces(mf, table, occ, vir, opts, cache, e_corr_total, name):
    """One row per (occupied type set, virtual type set, size): the CASCI energy of the space.
    ``opts.sizes`` (a set of (electrons, orbitals)) keeps only those sizes; None keeps all."""
    occ_t, vir_t = occ, vir                         # orbitals the type sets may use
    if opts.types:
        occ_t, vir_t = occ_t[occ_t["type"].isin(opts.types)], vir_t[vir_t["type"].isin(opts.types)]
    if opts.no_outside_valence:
        vir_t = vir_t[vir_t["type"] != NON_VALENCE]
    sizes = getattr(opts, "sizes", None)
    occ_sets = {**type_sets(sorted(set(occ_t["type"])), opts.max_union), ALL: None}
    vir_sets = {**type_sets(sorted(set(vir_t["type"])), opts.max_union), ALL: None}
    max_occ = opts.max_electrons // 2
    max_vir = min(opts.max_orbitals, opts.max_virtual or opts.max_orbitals)

    rows = []
    for (o_label, os_), (v_label, vs) in itertools.product(occ_sets.items(), vir_sets.items()):
        o_rows = occ if os_ is None else occ_t[occ_t["type"].isin(os_)]     # None = energy order
        v_rows = vir if vs is None else vir_t[vir_t["type"].isin(vs)]
        o_choices = cumulative(groups(table, o_rows, "down"), max_occ)
        v_choices = cumulative(groups(table, v_rows, "up"), max_vir)
        for o_list, v_list in itertools.product(o_choices, v_choices):
            if len(o_list) + len(v_list) > opts.max_orbitals:
                continue
            active = o_list + v_list
            nelec = 2 * len(o_list)
            if sizes is not None and (nelec, len(active)) not in sizes:
                continue
            e, s2 = casci(mf, active, nelec, cache)
            rows.append(dict(
                molecule=name, occupied_types=o_label, virtual_types=v_label,
                nele=nelec, norb=len(active), E_casci=e, E_corr_mHa=1e3 * (e - mf.e_tot),
                percent_of_ccsd=100 * (e - mf.e_tot) / e_corr_total, ccsd_corr_mHa=1e3 * e_corr_total, S2=s2,
                active=" ".join(str(k) for k in sorted(active)),
                names=" ".join(table.loc[sorted(active), "name"]),
                types=" | ".join(table.loc[sorted(active), "type"])))
    return rows


def with_energy_order_gain(rows):
    """DataFrame of the type spaces with E_corr minus the energy-ordered space of the same size."""
    data = pd.DataFrame(rows)
    ref = data[(data["occupied_types"] == ALL) & (data["virtual_types"] == ALL)]
    ref = ref.set_index(["nele", "norb"])["E_corr_mHa"]
    data["gain_vs_energy_order_mHa"] = [r.E_corr_mHa - ref[(r.nele, r.norb)] if (r.nele, r.norb) in ref.index
                                        else np.nan for r in data.itertuples()]
    return data


def left_out_rows(mf, table, opts, cache, e_corr_total, name, main_data, progress=None):
    """The type spaces again with one orbital left out, with its whole degenerate set: HOMO-k for
    k in ``opts.leave_out`` and LUMO+k for k in ``opts.leave_out_virtual``. Only spaces not in
    ``main_data`` (the type-space table) are returned."""
    nocc = table.attrs["nocc"]
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]
    sets = degenerate_sets(np.asarray(mf.mo_energy))
    ref = main_data[(main_data["occupied_types"] == ALL) & (main_data["virtual_types"] == ALL)]
    ref = ref.set_index(["nele", "norb"])["E_corr_mHa"]
    best = main_data.groupby(["nele", "norb"])["E_corr_mHa"].min()
    seen = set(main_data["active"])
    rows, done = [], set()
    targets = [nocc - 1 - k for k in opts.leave_out if nocc - 1 - k in set(occ["index"])] + \
              [nocc + k for k in opts.leave_out_virtual if nocc + k in set(vir["index"])]
    for orbital in targets:
        removed = tuple(sets.get(orbital, [orbital]))
        if removed in done:                     # e.g. HOMO-3 in the same degenerate set as HOMO-2
            continue
        done.add(removed)
        label = "/".join(orbital_name(m, nocc) for m in sorted(removed, reverse=orbital < nocc))
        o_keep, v_keep = occ[~occ["index"].isin(removed)], vir[~vir["index"].isin(removed)]
        for r in type_spaces(mf, table, o_keep, v_keep, opts, cache, e_corr_total, name):
            if r["active"] in seen:
                continue
            seen.add(r["active"])
            size = (r["nele"], r["norb"])
            r["left_out"] = label
            r["gain_vs_energy_order_mHa"] = r["E_corr_mHa"] - ref[size] if size in ref.index else np.nan
            r["vs_best_with_it_mHa"] = r["E_corr_mHa"] - best[size] if size in best.index else np.nan
            rows.append(r)
        if progress:
            progress(label, len(rows))
    return rows


def csv_paths(folder, molecule):
    main = os.path.join(folder, f"{molecule}_bond_type_casci.csv")
    return main, main.replace(".csv", "_left_out.csv")


def stored_sizes(folder, molecule):
    """Sizes (electrons, orbitals) already in a molecule's result files."""
    sizes = set()
    for path in csv_paths(folder, molecule):
        if os.path.exists(path):
            d = pd.read_csv(path, encoding="utf-8-sig", usecols=["nele", "norb"])
            sizes |= {(int(a), int(b)) for a, b in zip(d["nele"], d["norb"])}
    return sizes


def all_sizes(max_electrons, max_orbitals, n_valence_occ=None, n_virtual=None):
    """Every size (electrons, orbitals) up to the limits, with at least one occupied and one
    virtual orbital; with the orbital counts, only sizes the molecule can hold."""
    out = []
    for nele in range(2, max_electrons + 1, 2):
        for norb in range(nele // 2 + 1, max_orbitals + 1):
            if n_valence_occ is not None and nele // 2 > n_valence_occ:
                continue
            if n_virtual is not None and norb - nele // 2 > n_virtual:
                continue
            out.append((nele, norb))
    return out


def ensure_sizes(folder, molecule, mf, sizes, max_memory=16000, opts=None, log=print, require=True):
    """Make sure the result files of ``molecule`` hold the spaces of every size in ``sizes``.

    Sizes already there are not computed again. The missing ones are searched together in
    one pass (type sets and left-out runs, ``opts`` as in the published search) and appended
    to ``<molecule>_bond_type_casci.csv`` and ``..._left_out.csv`` in ``folder``. Returns the
    sizes that were searched. ``require``: a missing size that cannot be built is an error
    (otherwise it is skipped). ``mf`` is the converged RHF of the molecule."""
    have = stored_sizes(folder, molecule)
    missing = sorted({tuple(s) for s in sizes} - have)
    if not missing:
        return []
    opts = opts or options()
    opts = SimpleNamespace(**{**vars(opts), "sizes": set(missing),
                              "max_electrons": max([opts.max_electrons] + [e for e, _ in missing]),
                              "max_orbitals": max([opts.max_orbitals] + [o for _, o in missing])})
    table = typed_orbitals(mf)
    ncore = table.attrs["ncore"]
    main_path, left_path = csv_paths(folder, molecule)
    old_main = pd.read_csv(main_path, encoding="utf-8-sig") if os.path.exists(main_path) else None
    if old_main is not None and len(old_main):
        e_corr_total = float(old_main["ccsd_corr_mHa"].iloc[0]) / 1e3      # same molecule, same CCSD
    else:
        log(f"[BOND-TYPE] {molecule}: frozen-core CCSD of the whole molecule", flush=True)
        e_corr_total = frozen_core_ccsd(mf, ncore, max_memory)
    shown = ", ".join(f"({e}e,{o}o)" for e, o in missing)
    log(f"[BOND-TYPE] {molecule}: {len(missing)} size(s) not in {folder}, searching them now: {shown}", flush=True)
    cache = {}
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]
    rows = type_spaces(mf, table, occ, vir, opts, cache, e_corr_total, molecule)
    built = {(r["nele"], r["norb"]) for r in rows}
    impossible = [s for s in missing if s not in built]
    if impossible and require:
        raise ValueError(f"{molecule}: no active space of size "
                         f"{', '.join(f'({e}e,{o}o)' for e, o in impossible)} can be built "
                         f"({len(occ)} valence occupied, {len(vir)} virtual orbitals)")
    if not rows:
        return []
    new_main = with_energy_order_gain(rows)
    left = left_out_rows(mf, table, opts, cache, e_corr_total, molecule, new_main)
    os.makedirs(folder, exist_ok=True)
    main = new_main if old_main is None else pd.concat([old_main, new_main.round(6)], ignore_index=True)
    main.round(6).to_csv(main_path, index=False, encoding="utf-8-sig")
    if left:
        old_left = pd.read_csv(left_path, encoding="utf-8-sig") if os.path.exists(left_path) else None
        new_left = pd.DataFrame(left)
        out = new_left if old_left is None else pd.concat([old_left, new_left.round(6)], ignore_index=True)
        out.round(6).to_csv(left_path, index=False, encoding="utf-8-sig")
    log(f"[BOND-TYPE] {molecule}: {new_main['active'].nunique()} type spaces + {len(left)} left-out "
        f"spaces, {len(cache)} CASCI -> {main_path}", flush=True)
    return sorted(built)


def ensure_size(folder, molecule, mf, nele, norb, max_memory=16000, opts=None, log=print):
    """One size: True if it was searched now, False if it was already in the files."""
    return bool(ensure_sizes(folder, molecule, mf, [(nele, norb)], max_memory, opts, log))
