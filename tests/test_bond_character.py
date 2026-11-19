"""Strict checks of bond type and character for every orbital of every molecule.

Each label is checked against something computed independently of the labelling code:
the bonds of the geometry (from atom distances), the Lewis structures, a reflection computed with a
mirrored copy of the molecule, PySCF's own valence virtual orbitals (lo.vvo.vvo) and AVAS.
Tetracene and Pentacene are marked slow.
"""

from collections import Counter

import numpy as np
import pytest
from pyscf import gto
from pyscf.lo import vvo
from pyscf.lo.iao import reference_mol
from pyscf.mcscf import avas

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import (LONE_PAIR_MIN, NON_VALENCE, OUT_OF_PLANE, classify_orbitals,
                                         localized_orbitals, molecular_plane)

COVALENT_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}     # Angstrom
# occupied pi orbitals = pi bonds + lone pairs in the pi system, from the Lewis structures
LEWIS_PI = {"Ethylene": 1, "NH2-": 1, "Methanamide": 2, "Benzene": 3, "Naphthalene": 5, "Uracil": 5,
            "Thymine": 5, "Cytosine": 5, "Adenine": 6, "Guanine": 7, "Tetracene": 9, "Pentacene": 11}
# the only orbitals where the out-of-plane share and the reflection disagree, both in molecules
# that are not exactly planar; see test_pi_share_agrees_with_the_reflection
DISAGREE = {"Methanamide": {"LUMO+26"}, "Thymine": {"LUMO+6"}}

SLOW = {"Tetracene", "Pentacene"}
ALL = [pytest.param(m, marks=pytest.mark.slow) if m in SLOW else m for m in molecules]

_cache = {}


def data(name):
    """Everything about one molecule, computed once."""
    if name not in _cache:
        mf = run_scf(molecules[name], "cc-pVDZ")
        _cache[name] = dict(mf=mf, table=classify_orbitals(mf), loc=localized_orbitals(mf))
    return _cache[name]


def symbol_pair(a, b):
    return "–".join(sorted((a, b), key=lambda s: (s == "H", s)))


def bonds(mol):
    """Bonded atom pairs from the geometry: closer than 1.2 x the sum of covalent radii."""
    xyz = mol.atom_coords(unit="Angstrom")
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    return {frozenset((i, j)) for i in range(mol.natm) for j in range(i)
            if np.linalg.norm(xyz[i] - xyz[j]) < 1.2 * (COVALENT_RADIUS[sym[i]] + COVALENT_RADIUS[sym[j]])}


def bond_counts(mol):
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    return Counter(symbol_pair(*(sym[i] for i in pair)) for pair in bonds(mol))


def main_atoms(charges):
    atoms = np.flatnonzero(charges > 0.15)
    if len(atoms) <= 1 and charges.max() < LONE_PAIR_MIN:
        atoms = np.argsort(-charges)[:2]
    return atoms[np.argsort(-charges[atoms])][:2]


def counts(names, suffix):
    return Counter({n[:-len(suffix)]: c for n, c in Counter(names).items() if n.endswith(suffix)})


def reflection(mol, mo_coeff, normal):
    """<psi|R|psi> for every orbital, R the mirror through the molecular plane, built from scratch:
    overlap with a mirrored copy of the molecule, sign -1 for the functions odd in the normal."""
    axis = int(np.argmax(np.abs(normal)))
    mass = mol.atom_mass_list()
    xyz = mol.atom_coords()
    centre = (mass[:, None] * xyz).sum(0) / mass.sum()
    mirrored = xyz.copy()
    mirrored[:, axis] = 2 * centre[axis] - mirrored[:, axis]
    mol2 = mol.copy()
    mol2.set_geom_(mirrored, unit="Bohr", inplace=True)
    s12 = gto.intor_cross("int1e_ovlp", mol, mol2)
    odd = np.array([(shell[1:] + comp) in OUT_OF_PLANE[axis] for _, _, shell, comp in mol.ao_labels(fmt=False)])
    return np.einsum("mk,mn,n,nk->k", mo_coeff, s12, np.where(odd, -1.0, 1.0), mo_coeff)


# ── localized orbitals against the bonds of the geometry ────────────────

@pytest.mark.parametrize("name", ALL)
def test_one_sigma_bond_orbital_for_every_bond_of_the_geometry(name):
    d = data(name)
    assert counts(d["loc"]["bond"]["names"], " σ") == bond_counts(d["mf"].mol)


@pytest.mark.parametrize("name", ALL)
def test_one_sigma_antibond_orbital_for_every_bond_of_the_geometry(name):
    d = data(name)
    assert counts(d["loc"]["antibond"]["names"], " σ*") == bond_counts(d["mf"].mol)


@pytest.mark.parametrize("name", ALL)
def test_every_two_atom_orbital_sits_on_a_bonded_pair(name):
    d = data(name)
    mol = d["mf"].mol
    pairs = bonds(mol)
    for part in ("bond", "antibond"):
        for n, ch in zip(d["loc"][part]["names"], d["loc"][part]["charges"]):
            atoms = main_atoms(ch)
            if len(atoms) == 2:
                assert frozenset(atoms.tolist()) in pairs, f"{part} {n} on {atoms}"
                assert n.split()[0] == symbol_pair(*(mol.atom_symbol(a) for a in atoms)), n


@pytest.mark.parametrize("name", ALL)
def test_pi_orbitals_match_the_lewis_structure(name):
    names = data(name)["loc"]["bond"]["names"]
    assert sum("π" in n for n in names) == LEWIS_PI[name]


@pytest.mark.parametrize("name", ALL)
def test_one_pi_antibond_for_every_pi_bond(name):
    loc = data(name)["loc"]
    pi_bonds = sum(n.endswith(" π") for n in loc["bond"]["names"])
    pi_lone = sum("lone pair (π)" in n for n in loc["bond"]["names"])
    assert pi_bonds + pi_lone == LEWIS_PI[name]
    assert sum(n.endswith(" π*") for n in loc["antibond"]["names"]) == pi_bonds


@pytest.mark.parametrize("name", ALL)
def test_the_remaining_valence_pairs_are_sigma_lone_pairs(name):
    d = data(name)
    t = d["table"]
    valence_pairs = int((t["occupied"] & ~t["core"]).sum())
    sigma_lone = sum(n.endswith("lone pair") for n in d["loc"]["bond"]["names"])
    assert sigma_lone == valence_pairs - sum(bond_counts(d["mf"].mol).values()) - LEWIS_PI[name]


@pytest.mark.parametrize("name", ALL)
def test_lone_pairs_are_only_on_n_and_o_and_sit_on_one_atom(name):
    d = data(name)
    for n, ch in zip(d["loc"]["bond"]["names"], d["loc"]["bond"]["charges"]):
        if "lone pair" in n:
            assert n.split()[0] in ("N", "O"), n
            assert ch.max() >= LONE_PAIR_MIN, (n, ch.max())
    assert not any("non-bonding" in n for n in d["loc"]["antibond"]["names"])


@pytest.mark.parametrize("name", ALL)
def test_names_use_only_the_atoms_of_the_molecule(name):
    d = data(name)
    present = {d["mf"].mol.atom_symbol(a) for a in range(d["mf"].mol.natm)}
    for n in list(d["loc"]["bond"]["names"]) + list(d["loc"]["antibond"]["names"]):
        assert set(n.split()[0].split("–")) <= present, n


# ── shares: exact identities, for every orbital ─────────────────────────

@pytest.mark.parametrize("name", ALL)
def test_one_antibond_orbital_for_each_iao_beyond_the_occupied_orbitals(name):
    d = data(name)
    n_iao = reference_mol(d["mf"].mol).nao
    assert len(d["loc"]["antibond"]["names"]) == n_iao - d["table"].attrs["nocc"]


@pytest.mark.parametrize("name", ALL)
def test_bond_shares_of_every_orbital_add_up_to_one(name):
    t = data(name)["table"]
    for core, occupied, shares in zip(t["core"], t["occupied"], t["bond_shares"]):
        if core:
            assert shares == {}
        else:
            assert sum(shares.values()) == pytest.approx(1.0, abs=1e-8)
            assert all(-1e-12 <= w <= 1 + 1e-12 for w in shares.values())
            assert (NON_VALENCE in shares) == (not occupied)


@pytest.mark.parametrize("name", ALL)
def test_each_localized_orbital_is_shared_out_completely(name):
    """Each bond orbital lies in the occupied valence space, each antibond in the virtual space."""
    loc = data(name)["loc"]
    assert np.allclose((loc["bond"]["overlap"] ** 2).sum(1), 1.0, atol=1e-8)
    assert np.allclose((loc["antibond"]["overlap"] ** 2).sum(1), 1.0, atol=1e-8)
    for part in ("bond", "antibond"):
        assert np.allclose(loc[part]["charges"].sum(1), 1.0, atol=1e-8)
        assert np.allclose(loc[part]["iao_weights"].sum(1), 1.0, atol=1e-8)


@pytest.mark.parametrize("name", ALL)
def test_non_valence_part_matches_pyscf_valence_virtual_orbitals(name):
    d = data(name)
    mf, t = d["mf"], d["table"]
    nocc = t.attrs["nocc"]
    s = mf.get_ovlp()
    orbitals = vvo.vvo(mf.mol, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], s=s)
    expected = 1 - ((orbitals.T @ s @ mf.mo_coeff[:, nocc:]) ** 2).sum(0)
    virtual = t[~t["occupied"]]
    groups = virtual["degenerate_group"].to_numpy()
    expected = np.array([expected[groups == g].mean() for g in groups])     # the table averages a degenerate set
    assert np.allclose(virtual["no_bond_outside_valence"], expected, atol=1e-8)


@pytest.mark.parametrize("name", ALL)
def test_degenerate_orbitals_have_the_same_shares(name):
    t = data(name)["table"]
    for _, rows in t.groupby("degenerate_group"):
        first = rows["bond_shares"].iloc[0]
        for shares in rows["bond_shares"]:
            assert shares.keys() == first.keys()
            assert all(abs(shares[k] - first[k]) < 1e-12 for k in first)


# ── pi or sigma, for every orbital ──────────────────────────────────────

@pytest.mark.parametrize("name", ALL)
def test_pi_share_agrees_with_the_reflection(name):
    """In a planar molecule <psi|R|psi> = 1 - 2P exactly. Methanamide (heavy atoms 0.08 Bohr out of
    the plane) and Thymine (methyl hydrogens) are not planar; there only the sign is compared, and
    the two documented exceptions are explained."""
    d = data(name)
    mf, t = d["mf"], d["table"]
    _, normal, out_of_plane = molecular_plane(mf.mol)
    r = reflection(mf.mol, mf.mo_coeff, normal)
    p = t["pi_score"].to_numpy()
    flat = max(np.abs(mf.mol.atom_coords() @ normal - (mf.mol.atom_coords() @ normal).mean()))
    if flat < 0.01:
        assert np.abs(r - (1 - 2 * p)).max() < 1e-3
    disagree = {t["name"][k] for k in range(len(t)) if (p[k] > 0.5) != (r[k] < 0)}
    assert disagree == DISAGREE.get(name, set())
    rows = t.set_index("name")
    if name == "Thymine":          # pi symmetric through the plane, but mostly on the methyl hydrogens
        assert r[rows.loc["LUMO+6", "index"]] < -0.9 and rows.loc["LUMO+6", "h_pop"] > 0.5
    if name == "Methanamide":      # on the boundary: P = 0.50 and <psi|R|psi> = 0.0
        k = rows.loc["LUMO+26", "index"]
        assert abs(p[k] - 0.5) < 0.02 and abs(r[k]) < 0.05


@pytest.mark.parametrize("name", ALL)
def test_avas_pi_space_contains_exactly_the_pi_orbitals(name):
    d = data(name)
    mf, t = d["mf"], d["table"]
    _, normal, _ = molecular_plane(mf.mol)
    axis = "xyz"[int(np.argmax(np.abs(normal)))]
    heavy = sorted({mf.mol.atom_symbol(a) for a in range(mf.mol.natm) if mf.mol.atom_charge(a) > 1})
    ncas, nelecas, mo = avas.avas(mf, [f"{s} 2p{axis}" for s in heavy], threshold=0.2, verbose=0)
    ncore = (mf.mol.nelectron - nelecas) // 2
    inside = ((mo[:, ncore:ncore + ncas].T @ mf.get_ovlp() @ mf.mo_coeff) ** 2).sum(0) > 0.5
    is_pi = (t["pi_score"] > 0.5).to_numpy()
    valence_occ = (t["occupied"] & ~t["core"]).to_numpy()
    assert list(inside & valence_occ) == list(is_pi & valence_occ)        # occupied: the same orbitals
    assert not (inside & ~is_pi).any()                                     # virtual: only pi* ones


@pytest.mark.parametrize("name", ["NH2-", "Methanamide", "Uracil"])
def test_labels_and_shares_do_not_depend_on_rounding(name):
    """The same labels and shares with 1, 2 and 8 threads (different rounding in the localization).
    NH2- has a sigma and a pi lone pair on one atom, which mix freely unless localized apart."""
    from pyscf import lib
    results = []
    for threads in (1, 2, 8):
        with lib.with_omp_threads(threads):
            t = classify_orbitals(run_scf(molecules[name], "cc-pVDZ"))
        results.append((list(t["label"]), list(t["bonds"])))
    assert results[0] == results[1] == results[2]
