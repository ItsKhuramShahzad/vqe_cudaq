"""Orbital labels against known chemistry, and the rules that choose active orbitals."""

import numpy as np
import pandas as pd
import pytest
from pyscf import gto, scf

from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules
from vqe_cudaq.orbital_selection import (DegenerateSplitError, classify_orbitals, molecular_plane,
                                         select_active_orbitals, validate_active_orbitals)

# occupied pi orbitals = pi electrons / 2, from the Lewis structures
LEWIS_PI = {"Ethylene": 1, "NH2-": 1, "Methanamide": 2, "Benzene": 3, "Naphthalene": 5, "Uracil": 5,
            "Thymine": 5, "Cytosine": 5, "Adenine": 6, "Guanine": 7}
CORE = {"Ethylene": 2, "NH2-": 1, "Methanamide": 3, "Benzene": 6, "Uracil": 8, "Thymine": 9}


@pytest.fixture(scope="module")
def tables():
    return {}


def table_of(name, tables):
    if name not in tables:
        tables[name] = classify_orbitals(run_scf(molecules[name], "cc-pVDZ"))
    return tables[name]


def fake_table():
    """Occupied 0-5 (0 core), virtual 6-11, with the labels written out."""
    labels = ["core", "s(X-X)", "pi", "s(X-H)", "pi", "pi", "pi*", "s*(X-H)", "s*(X-H)", "pi*", "s*", "pi*"]
    energies = np.array([-11, -1.2, -.6, -.5, -.4, -.3, .1, .15, .2, .25, .3, .4])
    t = pd.DataFrame(dict(index=range(12), occupied=[k < 6 for k in range(12)], energy=energies,
                          label=labels, core=[k == 0 for k in range(12)],
                          degenerate_group=range(12)))
    return t


# ── rules, no chemistry ─────────────────────────────────────────────────

def test_energy_rule_is_the_current_consecutive_choice():
    s = select_active_orbitals(fake_table(), 4, 3)
    assert s.active == [4, 5, 6] and s.tag == "energy" and not s.fallbacks


def test_combined_rule_skips_the_excluded_classes_and_keeps_the_size():
    s = select_active_orbitals(fake_table(), 4, 3, rule="combined", exclude=["xh"])
    assert s.active == [4, 5, 6]
    s = select_active_orbitals(fake_table(), 6, 5, rule="combined", exclude=["xh"])
    assert s.active == [2, 4, 5, 6, 9] and s.tag == "no-xh"


def test_character_rule_takes_classes_in_order_and_records_a_fill_up():
    s = select_active_orbitals(fake_table(), 6, 5, rule="character", include=["pi"])
    assert s.active == [2, 4, 5, 6, 9] and not s.fallbacks
    s = select_active_orbitals(fake_table(), 8, 5, rule="character", include=["pi"])
    assert s.active == [2, 3, 4, 5, 6] and s.fallbacks == ["occupied: 1 of 4 filled by energy"]


def test_core_orbitals_are_never_chosen_automatically():
    s = select_active_orbitals(fake_table(), 10, 6, rule="energy")
    assert 0 not in s.active


def test_manual_list_is_sorted_and_tagged():
    s = select_active_orbitals(fake_table(), 4, 3, rule="manual", active=[6, 5, 3])
    assert s.active == [3, 5, 6] and s.tag == "manual_3-5-6"


@pytest.mark.parametrize("active, nele", [([3, 3, 6], 4), ([5, 6, 99], 4), ([3, 4, 5], 4),
                                          ([5, 6, 7], 4), ([3, 4], 4), ([4, 5, 6], 3)])
def test_invalid_lists_are_refused(active, nele):
    with pytest.raises(ValueError):
        validate_active_orbitals(active, nocc=6, nmo=12, nele_cas=nele)


def test_bad_options_are_refused():
    for kwargs in (dict(rule="nonsense"), dict(rule="character"), dict(rule="combined"),
                   dict(rule="character", include=["sigma"])):
        with pytest.raises(ValueError):
            select_active_orbitals(fake_table(), 4, 3, **kwargs)


def test_splitting_a_degenerate_pair_gives_a_warning():
    t = fake_table()
    t.loc[6:7, "degenerate_group"] = 6
    assert select_active_orbitals(t, 4, 3).warnings == ["degenerate orbitals split: takes [6] of [6, 7]"]
    assert not select_active_orbitals(t, 4, 4).warnings


def test_character_rule_never_splits_a_degenerate_set():
    t = fake_table()
    t.loc[4:5, "degenerate_group"] = 4                 # the two highest pi orbitals are degenerate
    with pytest.raises(DegenerateSplitError, match=r"0 or 2 occupied orbitals keep it whole"):
        select_active_orbitals(t, 2, 2, rule="character", include=["pi"])
    s = select_active_orbitals(t, 4, 3, rule="character", include=["pi"])
    assert s.active == [4, 5, 6] and not s.warnings


def test_extend_degenerate_takes_the_whole_set_and_reports_the_new_size():
    t = fake_table()
    t.loc[4:5, "degenerate_group"] = 4
    s = select_active_orbitals(t, 2, 2, rule="character", include=["pi"], extend_degenerate=True)
    assert s.active == [4, 5, 6] and (s.nele, s.norb) == (4, 3)
    assert "whole degenerate set" in s.warnings[0]


def test_degenerate_virtual_sets_are_kept_whole_too():
    t = fake_table()
    t.loc[7:8, "degenerate_group"] = 7                 # two degenerate sigma* orbitals
    with pytest.raises(DegenerateSplitError, match="virtual"):
        select_active_orbitals(t, 2, 2, rule="combined", exclude=["pi"])     # 1 virtual: half the pair
    assert select_active_orbitals(t, 2, 3, rule="combined", exclude=["pi"]).active == [3, 7, 8]
    s = select_active_orbitals(t, 2, 2, rule="combined", exclude=["pi"], extend_degenerate=True)
    assert s.active == [3, 7, 8] and (s.nele, s.norb) == (2, 3)


def test_energy_rule_still_reproduces_the_published_spaces():
    t = fake_table()
    t.loc[4:5, "degenerate_group"] = 4
    s = select_active_orbitals(t, 2, 2)                 # splits the pair, as the paper did, and warns
    assert s.active == [5, 6] and s.warnings


def test_benzene_pi_space_keeps_its_degenerate_pairs(tables):
    t = table_of("Benzene", tables)
    names = t.set_index("index")["name"]
    s = select_active_orbitals(t, 4, 4, rule="character", include=["pi"])
    assert [names[k] for k in s.active] == ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]
    with pytest.raises(DegenerateSplitError):
        select_active_orbitals(t, 2, 2, rule="character", include=["pi"])


# ── labels against known chemistry ──────────────────────────────────────

@pytest.mark.parametrize("name", sorted(LEWIS_PI))
def test_occupied_pi_count_matches_the_lewis_structure(name, tables):
    t = table_of(name, tables)
    assert t.attrs["planar"]
    assert ((t["label"] == "pi") & t["occupied"]).sum() == LEWIS_PI[name]


@pytest.mark.parametrize("name", ["Benzene", "Naphthalene", "Uracil", "Adenine"])
def test_pi_scores_are_clear_cut(name, tables):
    scores = table_of(name, tables)["pi_score"]
    assert (scores.lt(0.1) | scores.gt(0.9)).all()


def test_benzene_frontier_orbitals(tables):
    t = table_of("Benzene", tables).set_index("name")["label"]
    assert list(t[["HOMO", "HOMO-1", "HOMO-2", "HOMO-3", "HOMO-4"]]) == ["pi", "pi", "s(X-H)", "s(X-H)", "pi"]
    assert list(t[["LUMO", "LUMO+1", "LUMO+2"]]) == ["pi*", "pi*", "s*(X-H)"]


def test_polar_n_h_bonds_are_not_lone_pairs(tables):
    """NH2-: one in-plane lone pair and one pi lone pair; the other two orbitals are N-H bonds."""
    t = table_of("NH2-", tables).set_index("name")["label"]
    assert list(t[["HOMO", "HOMO-1", "HOMO-2", "HOMO-3"]]) == ["pi", "n", "s(X-H)", "s(X-H)"]


def test_thymine_methyl_group_is_not_counted_as_ring_pi(tables):
    t = table_of("Thymine", tables)
    extra = t[t["label"] == "pi(X-H)"]
    assert list(extra["name"]) == ["HOMO-8"] and extra["share_xh"].iloc[0] > 0.5


@pytest.mark.parametrize("name, counts", [("Ethylene", dict(pi=1, xh=4, xx=1, lone=0)),
                                          ("Benzene", dict(pi=3, xh=6, xx=6, lone=0)),
                                          ("Naphthalene", dict(pi=5, xh=8, xx=11, lone=0))])
def test_bond_orbitals_give_the_textbook_bonds(name, counts, tables):
    assert table_of(name, tables).attrs["bond_counts"] == counts


@pytest.mark.parametrize("name", sorted(LEWIS_PI))
def test_bond_orbitals_count_every_bond_to_hydrogen_and_every_pi(name, tables):
    t = table_of(name, tables)
    counts = t.attrs["bond_counts"]
    n_h = dict(Ethylene=4, Benzene=6, Naphthalene=8, Methanamide=3, Uracil=4, Thymine=6,
               Cytosine=5, Adenine=5, Guanine=5)
    n_h["NH2-"] = 2
    assert counts["xh"] == n_h[name] and counts["pi"] == LEWIS_PI[name]
    assert sum(counts.values()) == int(t["occupied"].sum() - t["core"].sum())


def test_ring_nitrogen_lone_pairs_are_found(tables):
    t = table_of("Adenine", tables).set_index("name")
    assert list(t.loc[["HOMO-2", "HOMO-4", "HOMO-5"], "label"]) == ["n", "n", "n"]


def test_shares_of_each_occupied_valence_orbital_add_up_to_one(tables):
    t = table_of("Cytosine", tables)
    v = t[t["occupied"] & ~t["core"]]
    total = v[["share_pi", "share_xh", "share_xx", "share_lone"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-8)


def test_labels_do_not_change_when_recomputed(tables):
    mf = run_scf(molecules["Uracil"], "cc-pVDZ")
    assert list(classify_orbitals(mf)["label"]) == list(classify_orbitals(mf)["label"])


@pytest.mark.parametrize("name, n", sorted(CORE.items()))
def test_core_count(name, n, tables):
    t = table_of(name, tables)
    assert t.attrs["ncore"] == n and t["core"].sum() == n
    assert (t[t["core"]]["energy"] < -10).all() and (t[~t["core"]]["energy"] > -3).all()


def ethylene(rotation=np.eye(3)):
    atoms = [("C", (0, 0, 0.6758)), ("C", (0, 0, -0.6758)), ("H", (0, 0.9366, 1.2497)),
             ("H", (0, -0.9366, 1.2497)), ("H", (0, 0.9366, -1.2497)), ("H", (0, -0.9366, -1.2497))]
    return scf.RHF(gto.M(atom=[(a, tuple(rotation @ np.array(c))) for a, c in atoms],
                         basis="cc-pVDZ", verbose=0)).run()


def test_labels_are_the_same_in_each_coordinate_plane():
    """Stored in the yz plane; the same molecule turned into the xz and xy planes."""
    yz_to_xz = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    yz_to_xy = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    labels = [classify_orbitals(ethylene(r))["label"] for r in (np.eye(3), yz_to_xz, yz_to_xy)]
    assert list(labels[0][:12]) == list(labels[1][:12]) == list(labels[2][:12])


def test_a_flat_molecule_not_along_an_axis_is_refused():
    th = 0.7
    tilted = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    with pytest.raises(ValueError, match="rotate the geometry"):
        classify_orbitals(ethylene(tilted))


def test_pi_score_agrees_with_reflection_through_the_plane():
    """Independent check of mo_comps: pi orbitals change sign in the mirror, sigma orbitals do not."""
    from pyscf import dft
    mf = ethylene()
    table = classify_orbitals(mf)
    grid = dft.gen_grid.Grids(mf.mol)
    grid.level = 1
    grid.build()
    mirrored = grid.coords * np.array([-1, 1, 1])              # the molecule lies in the yz plane
    a = dft.numint.eval_ao(mf.mol, grid.coords) @ mf.mo_coeff
    b = dft.numint.eval_ao(mf.mol, mirrored) @ mf.mo_coeff
    overlap = (grid.weights[:, None] * a * b).sum(0) / (grid.weights[:, None] * a * a).sum(0)
    assert np.allclose((1 - overlap) / 2, table["pi_score"], atol=1e-3)


def test_a_non_planar_molecule_gets_no_pi_label():
    water_dimer_like = [("C", (0, 0, 0)), ("C", (1.5, 0, 0)), ("C", (2.2, 1.3, 0)), ("C", (2.2, 1.3, 1.5)),
                        ("H", (-0.6, 0.9, 0)), ("H", (-0.6, -0.9, 0)), ("H", (-0.4, 0, 1)), ("H", (2, -0.9, 0.3)),
                        ("H", (1.7, -0.4, -1)), ("H", (3.3, 1.3, -0.2)), ("H", (1.7, 2.2, -0.4)),
                        ("H", (3.3, 1.3, 1.7)), ("H", (1.7, 0.4, 1.9)), ("H", (1.7, 2.2, 1.9))]
    mol = gto.M(atom=water_dimer_like, basis="sto-3g", verbose=0)
    assert molecular_plane(mol)[2] > 0.5
    t = classify_orbitals(scf.RHF(mol).run())
    assert not t.attrs["planar"] and "pi" not in set(t["label"])


def test_energy_rule_matches_the_shipped_active_spaces(tables):
    """The default rule gives the first ncore orbitals frozen, as every stored space assumes."""
    t = table_of("Benzene", tables)
    for space in molecules["Benzene"]["valid_active_spaces"]:
        s = select_active_orbitals(t, int(space["nele_cas"]), int(space["norb_cas"]))
        first = int(space["ncore"])
        assert s.active == list(range(first, first + int(space["norb_cas"])))


# ── bond shares by atoms: IBOs for occupied, LIVVOs for virtual orbitals ──

def test_bond_shares_of_ethylene_match_the_textbook_bonds(tables):
    t = table_of("Ethylene", tables).set_index("name")
    assert t.loc["HOMO-5", "bond_shares"]["C–C σ"] == pytest.approx(0.60, abs=0.01)
    assert t.loc["HOMO-5", "bond_shares"]["C–H σ"] == pytest.approx(0.40, abs=0.01)
    assert t.loc["HOMO-5", "bonds"] == "C–C σ 0.60, C–H σ 0.40"
    assert t.loc["HOMO", "bond_shares"]["C–C π"] == pytest.approx(1.0, abs=1e-6)
    assert t.loc["LUMO", "main_bond"] == "C–C π*" and t.loc["LUMO", "bond_shares"]["C–C π*"] > 0.85


def test_bond_names_use_only_the_atoms_of_the_molecule(tables):
    names = " ".join(table_of("Ethylene", tables).attrs["bond_names"]
                     + table_of("Ethylene", tables).attrs["antibond_names"])
    assert "N" not in names and "O" not in names
    t = table_of("Methanamide", tables)
    assert {"N–H σ", "C–H σ", "O lone pair"} <= set(t.attrs["bond_names"])


def test_pi_bonds_are_named_by_their_atoms(tables):
    names = set(table_of("Uracil", tables).attrs["bond_names"])
    assert {"C–C π", "C–O π"} <= names and "π" not in names


@pytest.mark.parametrize("name", ["Ethylene", "Methanamide", "Uracil"])
def test_bond_shares_add_up_to_one(name, tables):
    t = table_of(name, tables)
    shown = t[~t["core"]]
    assert np.allclose([sum(d.values()) for d in shown["bond_shares"]], 1.0, atol=1e-8)


@pytest.mark.parametrize("name", ["Ethylene", "Methanamide"])
def test_virtual_valence_space_has_one_orbital_per_iao_beyond_the_occupied(name, tables):
    t = table_of(name, tables)
    mol = run_scf(molecules[name], "cc-pVDZ").mol
    from pyscf.lo.iao import reference_mol
    n_iao = reference_mol(mol).nao
    valence_part = (1 - t[~t["occupied"]]["no_bond_outside_valence"]).sum()
    assert valence_part == pytest.approx(n_iao - t.attrs["nocc"], abs=1e-6)
