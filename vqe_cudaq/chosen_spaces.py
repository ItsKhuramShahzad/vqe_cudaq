"""Active spaces given as a list of orbitals, not only as the block nearest the gap.

The published runs use the energy rule: an active space ``(ncore, nele_cas, norb_cas)`` is the
block of ``norb_cas`` orbitals right after the ``ncore`` lowest ones. Here a space can instead be
any set of orbitals:

* ``manual``: orbital names such as ``HOMO-2 HOMO LUMO LUMO+3`` (or MO numbers from 0);
* ``bond-type``: a space found by ``scripts/bond_type_casci.py``, read from its CSV files, e.g.
  the best (6e,8o) space of a molecule.

Every occupied orbital that is not chosen stays doubly occupied (frozen), every virtual orbital
that is not chosen stays empty, so ``ncore = nocc - (chosen occupied)`` and
``nele_cas = 2 * (chosen occupied)``. Such a space is passed to the driver as the usual
dict plus ``"active_orbitals"`` (sorted MO numbers) and ``"rule"``.
"""

import os
import re

import numpy as np
import pandas as pd

RULES = ("energy", "bond-type", "manual")
DEGENERATE_HA = 1e-5            # orbital energies closer than this form one degenerate set


class ChosenSpaceError(ValueError):
    """The orbitals asked for do not make a valid active space."""


def orbital_index(name, nocc):
    """MO number (from 0) of ``HOMO``, ``HOMO-k``, ``LUMO``, ``LUMO+k`` or a plain number."""
    text = str(name).strip().replace("−", "-")
    if re.fullmatch(r"\d+", text):
        return int(text)
    m = re.fullmatch(r"(HOMO|LUMO)(?:([+-])(\d+))?", text.upper())
    if not m:
        raise ChosenSpaceError(f"orbital {name!r}: use HOMO, HOMO-k, LUMO, LUMO+k or an MO number")
    kind, sign, k = m.group(1), m.group(2), int(m.group(3) or 0)
    if kind == "HOMO" and sign == "+" and k or kind == "LUMO" and sign == "-" and k:
        raise ChosenSpaceError(f"orbital {name!r}: HOMO goes down (HOMO-k), LUMO goes up (LUMO+k)")
    return nocc - 1 - k if kind == "HOMO" else nocc + k


def orbital_name(k, nocc):
    """``HOMO-2``, ``HOMO``, ``LUMO``, ``LUMO+3`` for MO number ``k``."""
    if k < nocc:
        return "HOMO" if k == nocc - 1 else f"HOMO-{nocc - 1 - k}"
    return "LUMO" if k == nocc else f"LUMO+{k - nocc}"


def chosen_space(orbitals, nocc, nmo, rule="manual", mo_energy=None, source=None):
    """Active-space dict for a list of orbitals (names or MO numbers).

    Returns ``{"ncore", "nele_cas", "norb_cas", "active_orbitals", "rule", "label", "warnings"}``.
    With ``mo_energy`` a space that takes only part of a degenerate set gets a warning (any mix
    of a degenerate set is an equally valid orbital, so such a space depends on that mix)."""
    index = [orbital_index(o, nocc) for o in orbitals]
    if len(set(index)) != len(index):
        raise ChosenSpaceError(f"an orbital is given twice: {list(orbitals)}")
    active = sorted(index)
    if active[0] < 0 or active[-1] >= nmo:
        raise ChosenSpaceError(f"orbitals must be between 0 and {nmo - 1}, got {active}")
    occ = [k for k in active if k < nocc]
    if not occ or len(occ) == len(active):
        raise ChosenSpaceError(f"an active space needs occupied and virtual orbitals, got "
                               f"{[orbital_name(k, nocc) for k in active]}")
    warnings = []
    if mo_energy is not None:
        e = np.asarray(mo_energy)
        for k in active:
            partners = [int(j) for j in np.flatnonzero(abs(e - e[k]) < DEGENERATE_HA) if j not in active]
            if partners:
                warnings.append(f"{orbital_name(k, nocc)} is degenerate with "
                                f"{', '.join(orbital_name(j, nocc) for j in partners)}, which is not in the space")
    space = {"ncore": nocc - len(occ), "nele_cas": 2 * len(occ), "norb_cas": len(active),
             "active_orbitals": active, "rule": rule,
             "label": " ".join(orbital_name(k, nocc) for k in active), "warnings": warnings}
    if source:
        space["source"] = source
    return space


def bond_type_spaces(folder, molecule):
    """All spaces of one molecule from the bond-type CASCI CSV files, best E_corr first:
    columns ``nele, norb, active`` (MO numbers), ``E_casci``, ``E_corr_mHa``, ``left_out``."""
    main = os.path.join(folder, f"{molecule}_bond_type_casci.csv")
    if not os.path.exists(main):
        raise FileNotFoundError(f"no bond-type results for {molecule}: {main} (the CLI searches "
                                f"them with --bond-type-compute missing, or run scripts/bond_type_casci.py)")
    frames = [pd.read_csv(main, encoding="utf-8-sig").assign(left_out="")]
    left = main.replace(".csv", "_left_out.csv")
    if os.path.exists(left):
        frames.append(pd.read_csv(left, encoding="utf-8-sig").fillna({"left_out": ""}))
    table = pd.concat(frames, ignore_index=True).drop_duplicates("active")
    table = table.sort_values("E_corr_mHa", kind="stable").reset_index(drop=True)
    return table[["nele", "norb", "active", "E_casci", "E_corr_mHa", "left_out"]]


def bond_type_space(folder, molecule, nele, norb, nocc, nmo, rank=1):
    """The ``rank``-th best (1 = best E_corr) bond-type space of size (nele, norb)."""
    table = bond_type_spaces(folder, molecule)
    same = table[(table["nele"] == nele) & (table["norb"] == norb)]
    if len(same) < rank or rank < 1:
        sizes = sorted({(int(a), int(b)) for a, b in zip(table["nele"], table["norb"])})
        raise ChosenSpaceError(f"{molecule}: no space number {rank} of size ({nele}e,{norb}o); "
                               f"{len(same)} of that size, sizes available: {sizes}")
    row = same.iloc[rank - 1]
    active = [int(k) for k in str(row["active"]).split()]
    space = chosen_space(active, nocc, nmo, rule="bond-type",
                         source=f"{os.path.join(folder, molecule)}_bond_type_casci*.csv, "
                                f"({nele}e,{norb}o) rank {rank}")
    space["E_casci_bond_type"] = float(row["E_casci"])          # checked against the run
    if row["left_out"]:
        space["left_out"] = str(row["left_out"])
    return space


def space_key(space):
    """What identifies a space (for seeds and file names): the orbitals when they are chosen."""
    key = (int(space["ncore"]), int(space["nele_cas"]), int(space["norb_cas"]))
    if space.get("active_orbitals") is not None:
        key += (tuple(int(k) for k in space["active_orbitals"]),)
    return key


SELECT = ("best-per-size", "top", "all")


def passing_spaces(folder, molecule, nocc, nmo, max_electrons, max_orbitals, ratio_max=0.25,
                   select="best-per-size", top=5):
    """Bond-type spaces that pass the selection criterion, as driver space dicts, best first.

    Only spaces up to (max_electrons, max_orbitals) count. E_best is the most negative E_corr
    of all of them (every size), ratio = (E_corr - E_best) / |E_best|, and a space passes when
    ratio < ratio_max, exactly as in the Excel files of scripts/bond_type_casci.py.

    select: ``best-per-size`` = the best passing space of each size; ``top`` = the ``top`` best
    passing spaces overall; ``all`` = every passing space."""
    if select not in SELECT:
        raise ChosenSpaceError(f"select must be one of {SELECT}, got {select!r}")
    table = bond_type_spaces(folder, molecule)
    table = table[(table["nele"] <= max_electrons) & (table["norb"] <= max_orbitals)]
    if table.empty:
        raise ChosenSpaceError(f"{molecule}: no bond-type spaces up to ({max_electrons}e,{max_orbitals}o) in {folder}")
    e_best = float(table["E_corr_mHa"].min())
    table = table.assign(ratio=(table["E_corr_mHa"] - e_best) / abs(e_best))
    passing = table[table["ratio"] < ratio_max]                  # already best first
    if select == "best-per-size":
        passing = passing.drop_duplicates(["nele", "norb"])
    elif select == "top":
        if top < 1:
            raise ChosenSpaceError(f"top must be at least 1, got {top}")
        passing = passing.head(top)
    spaces = []
    for rank, row in enumerate(passing.itertuples(), start=1):
        space = chosen_space([int(k) for k in str(row.active).split()], nocc, nmo, rule="bond-type",
                             source=f"{os.path.join(folder, molecule)}_bond_type_casci*.csv, "
                                    f"{select}, ratio < {ratio_max:g}")
        space.update(E_casci_bond_type=float(row.E_casci), E_corr_mHa=float(row.E_corr_mHa),
                     E_best_mHa=e_best, ratio=float(row.ratio), selection=select, selection_rank=rank)
        if row.left_out:
            space["left_out"] = str(row.left_out)
        spaces.append(space)
    return spaces
