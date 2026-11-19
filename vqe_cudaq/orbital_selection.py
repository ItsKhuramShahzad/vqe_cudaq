"""Orbital character and the choice of active orbitals among canonical HF orbitals.

classify_orbitals labels each orbital from the orbitals themselves. A flat molecule must lie in a
coordinate plane, as all the benchmark geometries do. select_active_orbitals picks the orbitals of
one active space by a rule.
Each orbital also gets its share of every bond type, named by the atoms (for example "C–H σ"):
occupied orbitals from intrinsic bond orbitals, virtual orbitals from localized intrinsic valence
virtual orbitals. These shares describe the orbitals; the selection rules use the label.
Orbital numbers are zero-based, as in ``--mo-indices``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyscf import lo
from pyscf.data.elements import chemcore
from pyscf.lo import vvo
from pyscf.lo.iao import reference_mol
from pyscf.tools import mo_mapping

PI_SCORE = 0.5            # above: mostly built from functions pointing out of the plane
ALIGNED = 0.99            # the plane's normal must lie this close to the x, y or z axis
OUT_OF_PLANE = {0: ["px", "dxy", "dxz"], 1: ["py", "dxy", "dyz"], 2: ["pz", "dxz", "dyz"]}
PLANAR_LIMIT = 0.5        # Bohr: heavy atoms further than this from the plane = not planar
IBO_ATOM_SHARE = 0.15     # an atom takes part in a bond orbital above this share of its charge
MAIN_CLASS_SHARE = 0.5    # a canonical orbital gets a bond label above this share, else "mixed"
VIRTUAL_H_POP = 0.25      # virtual sigma*: Lowdin population on the H atoms, above = X-H
DEGENERATE_TOL = 1e-5     # Hartree
BOND_SHARE_MIN = 0.1      # a bond type is listed for an orbital when its share is above this
LONE_PAIR_MIN = 0.6       # a lone pair has at least this share on its atom; less = a spread-out bond
NON_VALENCE = "no bond (outside valence)"   # part of a virtual orbital outside every bond and antibond

CLASSES = {
    "pi": {"pi", "pi*"},
    "lone": {"n"},
    "xh": {"s(X-H)", "s*(X-H)", "pi(X-H)"},
    "xx": {"s(X-X)", "s*"},
    "mixed": {"s(mixed)"},
}

def molecular_plane(mol):
    """(centre of mass, unit normal, largest distance of a heavy atom from the plane in Bohr).

    The normal is the axis of largest moment of inertia. Hydrogens can be out of the plane, as in
    a methyl group, so flatness is judged on the heavy atoms where there are at least three.
    """
    
    xyz = mol.atom_coords()
    mass = mol.atom_mass_list()
    centre = (mass[:, None] * xyz).sum(0) / mass.sum()
    normal = np.linalg.eigh(mol.inertia_moment())[1][:, 2]
    heavy = np.array([mol.atom_charge(a) > 1 for a in range(mol.natm)])
    atoms = xyz[heavy] if heavy.sum() >= 3 else xyz
    return centre, normal, float(np.abs((atoms - centre) @ normal).max())


def _pi_scores(mol, mo_coeff, normal):
    """Share of each orbital on the p and d functions that point out of the molecular plane."""
    axis = int(np.argmax(np.abs(normal)))
    if abs(normal[axis]) < ALIGNED:
        raise ValueError(f"the molecular plane has normal {np.round(normal, 3).tolist()}, not along "
                         "x, y or z; rotate the geometry so that the molecule lies in a coordinate plane")
    return mo_mapping.mo_comps(OUT_OF_PLANE[axis], mol, mo_coeff, orth_method="lowdin")


def _iaos(mf):
    nocc = mf.mol.nelectron // 2
    return lo.vec_lowdin(lo.iao.iao(mf.mol, mf.mo_coeff[:, :nocc]), mf.get_ovlp())


def _split_sigma_pi(mol, orbitals, normal):
    """Rotate a set of orbitals into a pi part and a sigma part.

    The share on the functions pointing out of the plane is diagonalised inside the set: in a planar
    molecule its eigenvalues are 1 (pi) and 0 (sigma). Localizing each part on its own keeps sigma
    and pi apart; otherwise two lone pairs on one atom (one sigma, one pi, as in NH2-) can mix in any
    proportion without changing the localization measure, and the result depends on rounding.
    """
    axis = int(np.argmax(np.abs(normal)))
    odd = mol.search_ao_label(OUT_OF_PLANE[axis])
    c = lo.orth.lowdin(mol.intor_symmetric("int1e_ovlp")).T @ mol.intor_symmetric("int1e_ovlp") @ orbitals
    share = c[odd].T @ c[odd]
    w, v = np.linalg.eigh(share)
    rotated = orbitals @ v
    return rotated[:, w > PI_SCORE], rotated[:, w <= PI_SCORE]


def _localize(mol, orbitals, iao, normal, planar):
    """IBO localization, separately in the pi and sigma parts of a planar molecule."""
    if not planar:
        return lo.ibo.ibo(mol, orbitals, iaos=iao, verbose=0)
    parts = [p for p in _split_sigma_pi(mol, orbitals, normal) if p.shape[1]]
    return np.hstack([lo.ibo.ibo(mol, p, iaos=iao, verbose=0) for p in parts])


def _atom_charges(mol, iao, s, orbitals):
    """Charge of each localized orbital on each atom, from the IAOs; rows add up to 1."""
    amplitudes = iao.T @ s @ orbitals
    slices = reference_mol(mol).aoslice_by_atom()
    return np.array([(amplitudes[first:last] ** 2).sum(0) for _, _, first, last in slices]).T


def _bond_name(mol, charges, pi, virtual):
    """Name of a localized orbital from its atoms, for example "C–H σ", "C–O π*", "N lone pair"."""
    atoms = np.flatnonzero(charges > IBO_ATOM_SHARE)
    if len(atoms) <= 1 and charges.max() < LONE_PAIR_MIN:
        # one atom above the cut-off but not dominant: a bond spread over several atoms (inner rings
        # of the acenes), named by its two largest atoms
        atoms = np.argsort(-charges)[:2]
    if len(atoms) > 2:
        atoms = atoms[np.argsort(-charges[atoms])[:2]]
    symbols = sorted((mol.atom_symbol(a) for a in atoms), key=lambda s: (s == "H", s))
    pi = pi and "H" not in symbols        # a bond to H is sigma, even where a methyl group leaves the plane
    if len(symbols) == 1:
        return f"{symbols[0]} {'non-bonding' if virtual else 'lone pair'}" + (" (π)" if pi else "")
    return f"{symbols[0]}–{symbols[1]} {'π' if pi else 'σ'}{'*' if virtual else ''}"


def _localized(mf, ncore, pi_scores):
    """Intrinsic bond orbitals of the occupied valence orbitals (Knizia, JCTC 9, 4834, 2013).

    Returns the class of each bond orbital ("xh", "pi", "lone", "xx"), its name by atoms, the
    matrix U, where U[i, k] is the overlap of bond orbital i with canonical valence orbital k, the
    bond orbitals themselves and their charge on each atom.
    """
    mol = mf.mol
    nocc = mol.nelectron // 2
    s = mf.get_ovlp()
    valence = mf.mo_coeff[:, ncore:nocc]
    iao = _iaos(mf)
    _, normal, out_of_plane = molecular_plane(mol)
    ibo = _localize(mol, valence, iao, normal, out_of_plane <= PLANAR_LIMIT)
    overlap = ibo.T @ s @ valence
    charges = _atom_charges(mol, iao, s, ibo)
    is_h = np.array([mol.atom_charge(a) == 1 for a in range(mol.natm)])
    classes, names = [], []
    for i in range(ibo.shape[1]):
        atoms = np.flatnonzero(charges[i] > IBO_ATOM_SHARE)
        is_pi = (overlap[i] ** 2 * pi_scores[ncore:nocc]).sum() > PI_SCORE
        if is_h[atoms].any():                       # first: a methyl C-H can look pi-like
            classes.append("xh")
        elif is_pi:
            classes.append("pi")
        else:
            classes.append("lone" if len(atoms) == 1 else "xx")
        names.append(_bond_name(mol, charges[i], is_pi and classes[-1] != "xh", virtual=False))
    return np.array(classes), np.array(names), overlap, ibo, charges


def bond_orbitals(mf, ncore, pi_scores):
    """Class of each intrinsic bond orbital and its overlap with the canonical valence orbitals."""
    classes, _, overlap, _, _ = _localized(mf, ncore, pi_scores)
    return classes, overlap


def antibond_orbitals(mf, normal, planar):
    """Localized intrinsic valence virtual orbitals (PySCF lo.vvo.livvo), named by their atoms.

    There are as many as IAOs minus occupied orbitals: the antibonding partners of the bonds.
    Returns their names and V, where V[j, k] is the overlap of orbital j with virtual orbital k.
    The squares of a column add up to the valence part of that virtual orbital; the rest is outside
    the valence space (the extra functions of the basis set beyond a minimal basis).
    """
    mol = mf.mol
    nocc = mol.nelectron // 2
    s = mf.get_ovlp()
    iao = _iaos(mf)
    # PySCF's valence virtual orbitals (lo.vvo.vvo), localized like lo.vvo.livvo but sigma and pi apart
    valence_virtual = vvo.vvo(mol, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], iaos=iao, s=s)
    livvo = _localize(mol, valence_virtual, iao, normal, planar)
    charges = _atom_charges(mol, iao, s, livvo)
    pi = _pi_scores(mol, livvo, normal) > PI_SCORE if planar else np.zeros(livvo.shape[1], bool)
    names = np.array([_bond_name(mol, charges[j], pi[j], virtual=True) for j in range(livvo.shape[1])])
    return names, livvo.T @ s @ mf.mo_coeff[:, nocc:], livvo, charges


def localized_orbitals(mf):
    """Everything needed to tabulate the bond and antibond orbitals of one molecule.

    Keys: iao_labels ("C0 2px", ...); for "bond" (IBOs, occupied valence) and "antibond" (LIVVOs):
    names, charges (orbital x atom), iao_weights (orbital x IAO, squared amplitudes, rows add up
    to 1) and overlap (orbital x canonical orbital, signed); mo_charges: charge of each occupied
    valence orbital on each atom.
    """
    mol = mf.mol
    nocc, ncore = mol.nelectron // 2, chemcore(mol)
    s = mf.get_ovlp()
    _, normal, out_of_plane = molecular_plane(mol)
    planar = out_of_plane <= PLANAR_LIMIT
    pi = _pi_scores(mol, mf.mo_coeff, normal) if planar else np.zeros(mf.mo_coeff.shape[1])
    iao = _iaos(mf)
    _, bnames, boverlap, ibo, bcharges = _localized(mf, ncore, pi)
    anames, aoverlap, livvo, acharges = antibond_orbitals(mf, normal, planar)
    ref = reference_mol(mol)
    labels = [f"{symbol}{atom} {shell}{comp}" for atom, symbol, shell, comp in ref.ao_labels(fmt=False)]
    return dict(iao_labels=labels, ncore=ncore, nocc=nocc,
                mo_charges=_atom_charges(mol, iao, s, mf.mo_coeff[:, ncore:nocc]),
                bond=dict(names=bnames, charges=bcharges, overlap=boverlap,
                          iao_weights=((iao.T @ s @ ibo) ** 2).T),
                antibond=dict(names=anames, charges=acharges, overlap=aoverlap,
                              iao_weights=((iao.T @ s @ livvo) ** 2).T))


def _shares_by_name(names, overlap):
    """Per canonical orbital: {bond name: sum of squared overlaps of the localized orbitals with that name}."""
    weights = overlap ** 2
    return [{n: float(weights[names == n, k].sum()) for n in dict.fromkeys(names)}
            for k in range(overlap.shape[1])]


def bond_summary(shares, minimum=BOND_SHARE_MIN):
    """The bond types above ``minimum``, largest first, as [(name, share), ...]."""
    return sorted(((n, w) for n, w in shares.items() if w > minimum), key=lambda x: -x[1])


def _label(row, planar):
    if row["occupied"]:
        shares = {"xh": row["share_xh"], "xx": row["share_xx"], "lone": row["share_lone"]}
        if planar and row["pi_score"] > PI_SCORE:
            return "pi" if row["share_pi"] >= row["share_xh"] else "pi(X-H)"
        best = max(shares, key=shares.get)
        if shares[best] < MAIN_CLASS_SHARE:
            return "s(mixed)"
        return {"xh": "s(X-H)", "xx": "s(X-X)", "lone": "n"}[best]
    if planar and row["pi_score"] > PI_SCORE:
        return "pi*"
    if planar:
        return "s*(X-H)" if row["h_pop"] > VIRTUAL_H_POP else "s*"
    return "vir"


def classify_orbitals(mf):
    """One row per orbital: number, HOMO/LUMO name, energy, shares, label, degenerate group.

    Occupied valence orbitals are labelled by their overlap with intrinsic bond orbitals. Pi or
    sigma comes from the share on functions pointing out of the molecular plane (PySCF mo_comps).
    Virtual orbitals have no bond orbitals, so sigma* uses the Lowdin population on H.
    """
    mol = mf.mol
    mo = np.asarray(mf.mo_coeff)
    if mol.spin != 0:
        raise ValueError("classify_orbitals is for closed-shell molecules")
    nocc, nmo, ncore = mol.nelectron // 2, mo.shape[1], chemcore(mol)
    _, normal, out_of_plane = molecular_plane(mol)
    planar = out_of_plane <= PLANAR_LIMIT
    pi = _pi_scores(mol, mo, normal) if planar else np.zeros(nmo)

    s = mf.get_ovlp()
    population = (lo.orth.lowdin(s).T @ s @ mo) ** 2
    h_aos = np.zeros(mol.nao, bool)
    for a, (_, _, first, last) in enumerate(mol.aoslice_by_atom()):
        h_aos[first:last] = mol.atom_charge(a) == 1

    classes, bond_names, overlap, _, _ = _localized(mf, ncore, pi)
    antibond_names, voverlap, _, _ = antibond_orbitals(mf, normal, planar)
    shares = [{} for _ in range(nmo)]
    for k, d in zip(range(ncore, nocc), _shares_by_name(bond_names, overlap)):
        shares[k] = d
    for k, d in zip(range(nocc, nmo), _shares_by_name(antibond_names, voverlap)):
        shares[k] = {**d, NON_VALENCE: max(0.0, 1.0 - sum(d.values()))}
    names = [("HOMO" if k < nocc else "LUMO")
             + (f"{k - nocc + 1 if k < nocc else k - nocc:+d}" if k not in (nocc - 1, nocc) else "")
             for k in range(nmo)]
    table = pd.DataFrame(dict(index=range(nmo), name=names, occupied=[k < nocc for k in range(nmo)],
                              energy=mf.mo_energy, core=[k < ncore for k in range(nmo)],
                              pi_score=pi, h_pop=population[h_aos].sum(0)))
    for cls in ("pi", "xh", "xx", "lone"):
        share = np.zeros(nmo)
        share[ncore:nocc] = (overlap[classes == cls] ** 2).sum(0)
        table[f"share_{cls}"] = share
    table["degenerate_group"] = np.cumsum(np.r_[0, np.diff(table["energy"]) > DEGENERATE_TOL])
    # Any mix of a degenerate set is equally valid, so the set gets one label from its average.
    scores = ["pi_score", "h_pop", "share_pi", "share_xh", "share_xx", "share_lone"]
    averaged = table[scores].groupby(table["degenerate_group"]).transform("mean")
    labelled = pd.concat([table[["occupied"]], averaged], axis=1)
    table["label"] = ["core" if c else _label(r, planar)
                      for c, (_, r) in zip(table["core"], labelled.iterrows())]
    # bond shares by name, averaged over a degenerate set like the label
    for g in np.unique(table["degenerate_group"]):
        members = np.flatnonzero(table["degenerate_group"].to_numpy() == g)
        keys = dict.fromkeys(n for k in members for n in shares[k])
        mean = {n: float(np.mean([shares[k].get(n, 0.0) for k in members])) for n in keys}
        for k in members:
            shares[k] = dict(mean)
    table["bond_shares"] = shares
    table["bonds"] = [", ".join(f"{n} {w:.2f}" for n, w in bond_summary(d)) for d in shares]
    table["main_bond"] = [max(d, key=d.get) if d else "" for d in shares]
    table["no_bond_outside_valence"] = [d.get(NON_VALENCE, 0.0) for d in shares]
    table.attrs.update(planar=planar, out_of_plane_bohr=out_of_plane, nocc=nocc, ncore=ncore,
                       bond_counts={c: int((classes == c).sum()) for c in ("pi", "xh", "xx", "lone")},
                       bond_names=sorted(set(bond_names)), antibond_names=sorted(set(antibond_names)))
    return table


def validate_active_orbitals(active, nocc, nmo, nele_cas):
    """Refuse a list that cannot be an active space of a closed-shell determinant."""
    active = [int(k) for k in active]
    if len(set(active)) != len(active):
        raise ValueError(f"duplicate orbitals in {active}")
    if any(k < 0 or k >= nmo for k in active):
        raise ValueError(f"orbital numbers must be between 0 and {nmo - 1}: {active}")
    if nele_cas % 2:
        raise ValueError(f"nele_cas must be even for a closed-shell determinant, got {nele_cas}")
    occupied = sum(k < nocc for k in active)
    if occupied != nele_cas // 2:
        raise ValueError(f"{nele_cas} active electrons need {nele_cas // 2} occupied orbitals, "
                         f"{active} has {occupied}")
    if occupied == len(active):
        raise ValueError(f"{active} has no virtual orbital")
    return sorted(active)


@dataclass
class Selection:
    active: list
    rule: str
    tag: str
    fallbacks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    nele: int = 0               # active electrons and orbitals of the space actually chosen
    norb: int = 0


def _classes(names):
    labels = set()
    for name in names or ():
        if name not in CLASSES:
            raise ValueError(f"unknown orbital class {name!r}; choose from {sorted(CLASSES)}")
        labels |= CLASSES[name]
    return labels


class DegenerateSplitError(ValueError):
    """The requested size would take only part of a set of degenerate orbitals."""


def select_active_orbitals(table, nele_cas, norb_cas, rule="energy", include=None, exclude=None,
                           active=None, max_virtual_energy=None, extend_degenerate=False):
    """Choose the orbitals of one active space.

    energy: nearest the gap. character: ``include`` classes in the order given, nearest the gap
    inside each class, filled up by energy if a class runs out. combined: energy order without the
    ``exclude`` classes. manual: the list ``active``. Core orbitals are never chosen automatically.

    Degenerate orbitals: any mix of a degenerate set is an equally valid orbital, so taking only
    part of a set gives a result that depends on that arbitrary mix. The character and combined
    rules therefore take whole sets only: a size that would split a set raises
    DegenerateSplitError, which names the sizes that keep it whole; with ``extend_degenerate`` the
    whole set is taken instead and the space grows (Selection.nele, Selection.norb, a warning).
    The energy rule and manual lists keep the requested orbitals, so that published active spaces
    are reproduced; a split set is reported in Selection.warnings.
    """
    nocc = int(table["occupied"].sum())
    n_occ, n_vir = nele_cas // 2, norb_cas - nele_cas // 2
    occ = [r for _, r in table[table["occupied"]].iloc[::-1].iterrows() if not r["core"]]
    vir = [r for _, r in table[~table["occupied"]].iterrows()]
    if max_virtual_energy is not None and rule != "energy":
        vir = [r for r in vir if r["energy"] <= max_virtual_energy]
    fallbacks, extended = [], []
    group = table["degenerate_group"].to_numpy()
    name = (table["name"] if "name" in table else table["index"]).astype(str).to_numpy()

    def fill(orbitals, n, side):
        if rule == "energy":
            return [int(r["index"]) for r in orbitals[:n]]
        if rule == "combined":
            drop = _classes(exclude)
            kept = [r for r in orbitals if r["label"] not in drop]
        else:
            kept = []
            for label_name in include or ():
                kept += [r for r in orbitals if r["label"] in _classes([label_name])]
        preferred = [int(r["index"]) for r in kept]
        on_side = [int(r["index"]) for r in orbitals]
        order = preferred + [k for k in on_side if k not in preferred]     # fill up by energy
        picked = []
        for k in order:
            if len(picked) >= n:
                break
            if k in picked:
                continue
            members = [m for m in on_side if group[m] == group[k]]        # the whole degenerate set
            if len(picked) + len(members) > n:
                if not extend_degenerate:
                    raise DegenerateSplitError(
                        f"{side}: {n} orbitals would take only part of the degenerate set "
                        f"{[name[m] for m in members]}; {len(picked)} or {len(picked) + len(members)} "
                        f"{side} orbitals keep it whole (or use extend_degenerate=True)")
                extended.append(f"{side}: took the whole degenerate set {[name[m] for m in members]}, "
                                f"{len(picked) + len(members)} orbitals instead of {n}")
            picked += members
        n_filled = sum(k not in preferred for k in picked)
        if n_filled:
            fallbacks.append(f"{side}: {n_filled} of {len(picked)} filled by energy")
        return picked

    if rule == "manual":
        chosen = list(active or [])
        tag = "manual_" + "-".join(str(k) for k in sorted(chosen))
    elif rule == "energy":
        chosen, tag = fill(occ, n_occ, "occupied") + fill(vir, n_vir, "virtual"), "energy"
    elif rule == "character":
        if not include:
            raise ValueError("rule 'character' needs include=[...]")
        _classes(include)
        chosen = fill(occ, n_occ, "occupied") + fill(vir, n_vir, "virtual")
        tag = "+".join(include)
    elif rule == "combined":
        if not exclude:
            raise ValueError("rule 'combined' needs exclude=[...]")
        chosen = fill(occ, n_occ, "occupied") + fill(vir, n_vir, "virtual")
        tag = "no-" + "+".join(exclude)
    else:
        raise ValueError(f"unknown rule {rule!r}")

    nele_cas = 2 * sum(k < nocc for k in chosen) if extended else nele_cas
    chosen = validate_active_orbitals(chosen, nocc, len(table), nele_cas)
    warnings = list(extended)
    groups = table["degenerate_group"].to_numpy()
    for g in sorted({int(groups[k]) for k in chosen}):
        members = np.flatnonzero(groups == g)
        taken = [int(k) for k in members if k in chosen]
        if len(taken) != len(members):
            warnings.append(f"degenerate orbitals split: takes {taken} of {[int(k) for k in members]}")
    return Selection(chosen, rule, tag, fallbacks, warnings, nele=nele_cas, norb=len(chosen))
