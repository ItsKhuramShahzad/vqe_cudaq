"""The bond-type search (vqe_cudaq.bond_type_search) and its use from the VQE command line.

A size that is not in the result files is searched on the fly (ensure_size). The searched
spaces must be exactly those of the full search of scripts/bond_type_casci.py: the same
spaces, energies and left-out cases, checked against a full search here and, when present,
against the cluster results in analysis/bond_type_casci_hpc.
"""

import glob
import os
import pickle

import pandas as pd
import pytest

from conftest import ROOT
from vqe_cudaq import bond_type_search as bts
from vqe_cudaq.chosen_spaces import bond_type_space
from vqe_cudaq.hamiltonian import run_scf
from vqe_cudaq.molecules import molecules

HPC = os.path.join(ROOT, "analysis", "bond_type_casci_hpc")
NOCC, NMO = 8, 48                       # Ethylene in cc-pVDZ


@pytest.fixture(scope="module")
def ethylene():
    mf = run_scf(molecules["Ethylene"], "cc-pVDZ")
    return mf, bts.typed_orbitals(mf)


def spaces_of(rows, size):
    return {(r["occupied_types"], r["virtual_types"], r["active"], round(r["E_casci"], 9))
            for r in rows if (r["nele"], r["norb"]) == size}


# ── small pieces ──────────────────────────────────────────────────────────────

def test_type_sets_are_unique_and_named():
    sets = bts.type_sets(["C–C π", "C–C σ", "C–H σ"], 2)
    assert set(sets) == {"C–C π", "C–C σ", "C–H σ", "C–C π + C–C σ", "C–C π + C–H σ",
                         "C–C σ + C–H σ", "C–C π + C–C σ + C–H σ"}
    assert len(set(sets.values())) == len(sets)                    # each set of types once


def test_cumulative_never_splits_a_degenerate_set():
    assert bts.cumulative([[7], [5, 6], [4]], 2) == [[7]]
    assert bts.cumulative([[7], [5, 6], [4]], 3) == [[7], [7, 5, 6]]


def test_orbital_types_of_ethylene(ethylene):
    _, table = ethylene
    t = dict(zip(table["name"], table["type"]))
    assert t["HOMO"] == "C–C π" and t["LUMO"] == "C–C π*" and t["HOMO-5"] == "C–C σ"
    assert all(t[f"HOMO-{k}"] == "C–H σ" for k in (1, 2, 3, 4))
    assert list(table["type"][table["core"]]) == ["core", "core"]


# ── one size equals the full search ──────────────────────────────────────────

def test_one_size_gives_the_spaces_of_the_full_search(ethylene):
    mf, table = ethylene
    occ = table[table["occupied"] & ~table["core"]]
    vir = table[~table["occupied"]]
    e_corr = -0.3
    full = bts.type_spaces(mf, table, occ, vir, bts.options(max_electrons=4, max_orbitals=4),
                           {}, e_corr, "Ethylene")
    for size in [(4, 4), (2, 3)]:
        one = bts.type_spaces(mf, table, occ, vir, bts.options(max_electrons=4, max_orbitals=4,
                                                               sizes={size}), {}, e_corr, "Ethylene")
        assert {(r["nele"], r["norb"]) for r in one} == {size}
        assert spaces_of(one, size) == spaces_of(full, size)


# ── ensure_size: compute once, then read ─────────────────────────────────────

def test_missing_size_is_searched_and_saved(ethylene, tmp_path, monkeypatch):
    mf, _ = ethylene
    assert bts.ensure_size(str(tmp_path), "Ethylene", mf, 4, 4, log=lambda *a, **k: None) is True
    main, left = bts.csv_paths(str(tmp_path), "Ethylene")
    assert os.path.exists(main) and os.path.exists(left)
    assert bts.stored_sizes(str(tmp_path), "Ethylene") == {(4, 4)}
    left_rows = pd.read_csv(left, encoding="utf-8-sig")
    assert set(left_rows["left_out"]) <= {"HOMO-1", "HOMO-2", "HOMO-3", "LUMO+1", "LUMO+2", "LUMO+3"}

    def no_search(*a, **k):
        raise AssertionError("searched again")
    monkeypatch.setattr(bts, "typed_orbitals", no_search)          # second call must only read
    assert bts.ensure_size(str(tmp_path), "Ethylene", mf, 4, 4) is False


def test_new_size_is_added_and_old_rows_are_kept(ethylene, tmp_path):
    mf, _ = ethylene
    quiet = dict(log=lambda *a, **k: None)
    bts.ensure_size(str(tmp_path), "Ethylene", mf, 4, 4, **quiet)
    main, _ = bts.csv_paths(str(tmp_path), "Ethylene")
    before = pd.read_csv(main, encoding="utf-8-sig")
    bts.ensure_size(str(tmp_path), "Ethylene", mf, 2, 3, **quiet)
    after = pd.read_csv(main, encoding="utf-8-sig")
    assert bts.stored_sizes(str(tmp_path), "Ethylene") == {(4, 4), (2, 3)}
    pd.testing.assert_frame_equal(after.iloc[:len(before)].reset_index(drop=True), before)
    assert after["ccsd_corr_mHa"].nunique() == 1                    # one CCSD for the molecule


def test_selected_space_is_the_best_of_its_size(ethylene, tmp_path):
    mf, _ = ethylene
    bts.ensure_size(str(tmp_path), "Ethylene", mf, 4, 4, log=lambda *a, **k: None)
    s = bond_type_space(str(tmp_path), "Ethylene", 4, 4, NOCC, NMO, rank=1)
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in bts.csv_paths(str(tmp_path), "Ethylene")]
    assert s["E_casci_bond_type"] == min(f["E_casci"].min() for f in frames)


def test_impossible_size_is_refused(tmp_path):
    mf = run_scf(molecules["NH2-"], "sto-3g")                         # 4 valence occupied orbitals
    with pytest.raises(ValueError, match="no active space"):
        bts.ensure_size(str(tmp_path), "NH2-", mf, 10, 12, log=lambda *a, **k: None)
    assert not os.listdir(tmp_path)                                   # nothing written


@pytest.mark.skipif(not os.path.exists(os.path.join(HPC, "Ethylene_bond_type_casci.csv")),
                    reason="no cluster results in analysis/bond_type_casci_hpc")
@pytest.mark.parametrize("size", [(4, 4), (6, 7)])
def test_on_the_fly_search_reproduces_the_cluster_run(ethylene, tmp_path, size):
    """Same spaces (type sets and left-out cases) and energies as the full HPC search."""
    mf, _ = ethylene
    bts.ensure_size(str(tmp_path), "Ethylene", mf, *size, log=lambda *a, **k: None)
    for new_path, old_path in zip(bts.csv_paths(str(tmp_path), "Ethylene"), bts.csv_paths(HPC, "Ethylene")):
        new = pd.read_csv(new_path, encoding="utf-8-sig")
        old = pd.read_csv(old_path, encoding="utf-8-sig")
        old = old[(old["nele"] == size[0]) & (old["norb"] == size[1])]
        key = ["active"] + (["left_out"] if "left_out" in new else ["occupied_types", "virtual_types"])
        a = new.set_index(key)["E_casci"].sort_index()
        b = old.set_index(key)["E_casci"].sort_index()
        assert list(a.index) == list(b.index)
        assert (a - b).abs().max() < 2e-6


# ── command line ──────────────────────────────────────────────────────────────

def parse(*argv):
    from vqe_cudaq.cli import build_parser
    return build_parser().parse_args(["--molecule", "Ethylene", "--active-space-rule", "bond-type", *argv])


def test_cli_searches_a_size_that_is_not_there(tmp_path):
    from vqe_cudaq.cli import chosen_active_spaces
    s, = chosen_active_spaces(parse("--bond-type-dir", str(tmp_path), "--sizes", "4,4"),
                              "Ethylene", molecules["Ethylene"])
    assert bts.stored_sizes(str(tmp_path), "Ethylene") == {(4, 4)}
    assert (s["nele_cas"], s["norb_cas"]) == (4, 4) and s["rule"] == "bond-type"


def test_cli_never_mode_only_reads(tmp_path):
    from vqe_cudaq.cli import chosen_active_spaces
    with pytest.raises(FileNotFoundError):
        chosen_active_spaces(parse("--bond-type-dir", str(tmp_path), "--sizes", "4,4",
                                   "--bond-type-compute", "never"), "Ethylene", molecules["Ethylene"])
    assert not os.listdir(tmp_path)


@pytest.mark.parametrize("size", ["4,2", "3,4", "0,2"])
def test_cli_refuses_sizes_that_cannot_exist(tmp_path, size):
    from vqe_cudaq.cli import chosen_active_spaces
    with pytest.raises(SystemExit, match="even number of electrons"):
        chosen_active_spaces(parse("--bond-type-dir", str(tmp_path), "--sizes", size),
                             "Ethylene", molecules["Ethylene"])


@pytest.mark.slow
def test_cli_new_user_runs_vqe_without_any_bond_type_files(tmp_path, monkeypatch):
    """Empty --bond-type-dir: the search runs for (4e,3o), then the VQE reaches its CASCI."""
    from vqe_cudaq import config
    from vqe_cudaq.cli import main
    for k, v in {"N_JITTER_RESTARTS": 0, "VQE_MAX_CYCLES": 1, "VQE_CHUNK_MAXITER": 200}.items():
        monkeypatch.setattr(config, k, v)
    main(["--molecule", "Ethylene", "--active-space-rule", "bond-type", "--bond-type-dir", str(tmp_path / "bt"),
          "--sizes", "4,3", "--integrals", str(tmp_path / "integrals"), "--target", "qpp-cpu",
          "--out_dir", str(tmp_path)])
    assert bts.stored_sizes(str(tmp_path / "bt"), "Ethylene") == {(4, 3)}
    path, = glob.glob(str(tmp_path / "*_bond-type_VQE_results.pkl"))
    with open(path, "rb") as fh:
        run, = pickle.load(fh)["Ethylene"]["active_space_runs"]
    assert abs(run["casci"]["E_casci_total"] - run["space"]["E_casci_bond_type"]) < 2e-6
    assert abs(run["compare"]["d_vqe_minus_casci"]) < 1e-6


# ── --sizes all: ratio criterion and the three ways to select ────────────────

def test_all_sizes_up_to_the_limits():
    sizes = bts.all_sizes(8, 10)
    assert len(sizes) == 30 and sizes[0] == (2, 2) and sizes[-1] == (8, 10)   # as the cluster search
    assert (4, 2) not in sizes and all(e % 2 == 0 and o > e // 2 for e, o in sizes)
    assert bts.all_sizes(8, 10, n_valence_occ=2) == [s for s in sizes if s[0] <= 4]


def selection_tables(folder):
    """E_best = -42 (a left-out space); ratio < 0.25 means E_corr < -31.5."""
    main = pd.DataFrame({"nele": [2, 2, 4, 4, 4, 6],
                         "norb": [2, 2, 4, 4, 4, 6],
                         "active": ["7 8", "6 9", "6 7 8 9", "5 7 8 9", "6 7 8 10", "5 6 7 8 9 10"],
                         "E_casci": [-78.048, -78.043, -78.078, -78.073, -78.058, -78.076],
                         "E_corr_mHa": [-10.0, -5.0, -40.0, -35.0, -20.0, -38.0]})
    left = pd.DataFrame({"nele": [4], "norb": [4], "active": ["4 7 8 9"], "E_casci": [-78.080],
                         "E_corr_mHa": [-42.0], "left_out": ["HOMO-1"]})
    main.to_csv(os.path.join(folder, "Ethylene_bond_type_casci.csv"), index=False, encoding="utf-8-sig")
    left.to_csv(os.path.join(folder, "Ethylene_bond_type_casci_left_out.csv"), index=False, encoding="utf-8-sig")


@pytest.mark.parametrize("select, top, expected", [
    ("all", 5, [-42.0, -40.0, -38.0, -35.0]),           # (c) every passing space, best first
    ("best-per-size", 5, [-42.0, -38.0]),               # (a) the best passing (4e,4o) and (6e,6o)
    ("top", 3, [-42.0, -40.0, -38.0]),                  # (b) the 3 best passing spaces
    ("top", 10, [-42.0, -40.0, -38.0, -35.0]),          # fewer pass than asked: all of them
])
def test_three_ways_to_select_passing_spaces(tmp_path, select, top, expected):
    from vqe_cudaq.chosen_spaces import passing_spaces
    selection_tables(tmp_path)
    spaces = passing_spaces(str(tmp_path), "Ethylene", NOCC, NMO, 8, 10, 0.25, select, top)
    assert [s["E_corr_mHa"] for s in spaces] == expected
    assert all(s["E_best_mHa"] == -42.0 and s["ratio"] < 0.25 for s in spaces)
    assert spaces[0]["left_out"] == "HOMO-1" and spaces[0]["ratio"] == 0.0
    assert abs([s for s in spaces if s["E_corr_mHa"] == -40.0][0]["ratio"] - 2 / 42) < 1e-12 \
        if -40.0 in expected else True
    assert [s["selection_rank"] for s in spaces] == list(range(1, len(spaces) + 1))


def test_limits_and_threshold_change_the_selection(tmp_path):
    from vqe_cudaq.chosen_spaces import passing_spaces
    selection_tables(tmp_path)
    up_to_4 = passing_spaces(str(tmp_path), "Ethylene", NOCC, NMO, 4, 4, 0.25, "all")
    assert [s["E_corr_mHa"] for s in up_to_4] == [-42.0, -40.0, -35.0]          # no (6e,6o)
    strict = passing_spaces(str(tmp_path), "Ethylene", NOCC, NMO, 8, 10, 0.05, "all")
    assert [s["E_corr_mHa"] for s in strict] == [-42.0, -40.0]                   # ratio 0 and 0.048
    assert passing_spaces(str(tmp_path), "Ethylene", NOCC, NMO, 2, 2, 0.25, "all")[0]["E_best_mHa"] == -10.0


@pytest.mark.skipif(not os.path.exists(os.path.join(HPC, "Ethylene_bond_type_casci.csv")),
                    reason="no cluster results in analysis/bond_type_casci_hpc")
def test_selection_agrees_with_the_excel_files():
    """Ethylene up to (8e,10o): E_best -39.143 mHa and 74 spaces pass ratio < 0.25 (summary sheet)."""
    from vqe_cudaq.chosen_spaces import passing_spaces
    every = passing_spaces(HPC, "Ethylene", NOCC, NMO, 8, 10, 0.25, "all")
    assert len(every) == 74 and round(every[0]["E_best_mHa"], 3) == -39.143      # Excel shows 3 decimals
    per_size = passing_spaces(HPC, "Ethylene", NOCC, NMO, 8, 10, 0.25, "best-per-size")
    sizes = [(s["nele_cas"], s["norb_cas"]) for s in per_size]
    assert len(sizes) == len(set(sizes)) == len({(s["nele_cas"], s["norb_cas"]) for s in every})
    assert per_size[0]["label"] == "HOMO-4 HOMO-3 HOMO-2 HOMO LUMO LUMO+6 LUMO+8 LUMO+11 LUMO+16 LUMO+18"


def test_cli_sizes_all_searches_every_missing_size_and_selects(tmp_path):
    from vqe_cudaq.cli import chosen_active_spaces
    args = parse("--bond-type-dir", str(tmp_path), "--sizes", "all", "--bond-type-max-electrons", "4",
                 "--bond-type-max-orbitals", "4", "--select", "best-per-size")
    spaces = chosen_active_spaces(args, "Ethylene", molecules["Ethylene"])
    assert bts.stored_sizes(str(tmp_path), "Ethylene") == set(bts.all_sizes(4, 4))
    sizes = [(s["nele_cas"], s["norb_cas"]) for s in spaces]
    assert len(sizes) == len(set(sizes)) and all(s["ratio"] < 0.25 for s in spaces)
    assert [s["E_corr_mHa"] for s in spaces] == sorted(s["E_corr_mHa"] for s in spaces)
    top = chosen_active_spaces(parse("--bond-type-dir", str(tmp_path), "--sizes", "all",
                                     "--bond-type-max-electrons", "4", "--bond-type-max-orbitals", "4",
                                     "--select", "top", "--top", "2", "--bond-type-compute", "never"),
                               "Ethylene", molecules["Ethylene"])
    assert len(top) == 2 and top[0]["E_corr_mHa"] == spaces[0]["E_corr_mHa"]


@pytest.mark.parametrize("argv, message", [
    (("--top", "0", "--select", "top"), "--top must be at least 1"),
    (("--bond-type-max-electrons", "3"), "even number of electrons"),
    (("--bond-type-max-electrons", "8", "--bond-type-max-orbitals", "4"), "more orbitals than electron pairs"),
    (("--ratio-max", "0.0"), "no space passes"),
])
def test_cli_sizes_all_refuses_bad_settings(tmp_path, argv, message):
    from vqe_cudaq.cli import chosen_active_spaces
    selection_tables(tmp_path)
    with pytest.raises(SystemExit, match=message):
        chosen_active_spaces(parse("--bond-type-dir", str(tmp_path), "--sizes", "all",
                                   "--bond-type-compute", "never", *argv), "Ethylene", molecules["Ethylene"])


@pytest.mark.slow
def test_cli_sizes_all_runs_vqe_on_the_selected_spaces(tmp_path, monkeypatch):
    """--sizes all up to (4e,3o), --select top --top 2: two VQE runs, each reaching its CASCI."""
    from vqe_cudaq import config
    from vqe_cudaq.cli import main
    for k, v in {"N_JITTER_RESTARTS": 0, "VQE_MAX_CYCLES": 1, "VQE_CHUNK_MAXITER": 200}.items():
        monkeypatch.setattr(config, k, v)
    main(["--molecule", "Ethylene", "--active-space-rule", "bond-type", "--bond-type-dir", str(tmp_path / "bt"),
          "--sizes", "all", "--bond-type-max-electrons", "4", "--bond-type-max-orbitals", "3",
          "--select", "top", "--top", "2", "--integrals", str(tmp_path / "integrals"),
          "--target", "qpp-cpu", "--out_dir", str(tmp_path)])
    path, = glob.glob(str(tmp_path / "*_bond-type_VQE_results.pkl"))
    with open(path, "rb") as fh:
        runs = pickle.load(fh)["Ethylene"]["active_space_runs"]
    assert len(runs) == 2
    assert [r["space"]["selection_rank"] for r in runs] == [1, 2]
    assert runs[0]["space"]["ratio"] == 0.0 and runs[1]["space"]["ratio"] < 0.25
    for run in runs:
        assert abs(run["compare"]["d_vqe_minus_casci"]) < 1e-6
