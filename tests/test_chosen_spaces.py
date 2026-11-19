"""Active spaces given as a list of orbitals (--active-space-rule manual / bond-type).

The energy rule (published spaces) must not change; a chosen space must give the same
energies as an independent route: PySCF CASCI with sort_mo, the full CI of the saved
integrals, the qubit Hamiltonian's lowest state, the HF reference energy, OpenFermion's
own active-space Hamiltonian and the bond-type CASCI tables. The end-to-end VQE runs at
the bottom are marked slow.
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from pyscf import mcscf

from conftest import INTEGRALS, ROOT, load
from vqe_cudaq import config
from vqe_cudaq.chosen_spaces import (ChosenSpaceError, bond_type_space, bond_type_spaces,
                                     chosen_space, orbital_index, orbital_name, space_key)
from vqe_cudaq.hamiltonian import (IntegralFiles, SpaceDoesNotFit, compute_active_space,
                                   integral_file_name, load_integrals, qubit_hamiltonian,
                                   run_ccsd, run_scf)
from vqe_cudaq.molecules import molecules

NOCC, NMO = 8, 48                       # Ethylene in cc-pVDZ
BOND_TYPE_DIR = os.path.join(ROOT, "analysis", "bond_type_casci_hpc")


# ── names and space dicts ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name, index", [("HOMO", 7), ("HOMO-2", 5), ("homo-7", 0), ("LUMO", 8),
                                         ("LUMO+3", 11), ("12", 12), ("HOMO−1", 6)])
def test_orbital_names_give_mo_numbers(name, index):
    assert orbital_index(name, NOCC) == index


@pytest.mark.parametrize("k", range(20))
def test_orbital_name_and_index_round_trip(k):
    assert orbital_index(orbital_name(k, NOCC), NOCC) == k


@pytest.mark.parametrize("bad", ["HOMO+1", "LUMO-2", "SOMO", "H-1", ""])
def test_wrong_orbital_names_are_refused(bad):
    with pytest.raises(ChosenSpaceError):
        orbital_index(bad, NOCC)


def test_chosen_space_counts_electrons_and_frozen_orbitals():
    s = chosen_space(["LUMO+3", "HOMO", "HOMO-2", "LUMO"], NOCC, NMO)
    assert s["active_orbitals"] == [5, 7, 8, 11]                 # sorted MO numbers
    assert (s["ncore"], s["nele_cas"], s["norb_cas"]) == (6, 4, 4)
    assert s["label"] == "HOMO-2 HOMO LUMO LUMO+3"
    assert s["rule"] == "manual" and s["warnings"] == []


@pytest.mark.parametrize("orbitals", [["HOMO", "HOMO-1"], ["LUMO", "LUMO+1"], ["HOMO", "HOMO", "LUMO"],
                                      ["HOMO", "47", "48"]])
def test_invalid_spaces_are_refused(orbitals):
    with pytest.raises(ChosenSpaceError):
        chosen_space(orbitals, NOCC, NMO)


def test_part_of_a_degenerate_set_gives_a_warning():
    e = np.array([-1.0, -0.5, -0.5, 0.2, 0.2, 0.4])      # HOMO/HOMO-1 and LUMO/LUMO+1 degenerate
    whole = chosen_space(["HOMO", "HOMO-1", "LUMO", "LUMO+1"], 3, 6, mo_energy=e)
    part = chosen_space(["HOMO", "LUMO", "LUMO+1"], 3, 6, mo_energy=e)
    assert whole["warnings"] == []
    assert part["warnings"] == ["HOMO is degenerate with HOMO-1, which is not in the space"]


def test_energy_rule_seed_key_is_unchanged():
    """The published runs seed with (molecule, ncore, nele, norb): no orbitals added for them."""
    assert space_key({"ncore": 6, "nele_cas": 4, "norb_cas": 3}) == (6, 4, 3)
    assert space_key(chosen_space(["HOMO-2", "HOMO", "LUMO"], NOCC, NMO)) == (6, 4, 3, (5, 7, 8))


# ── bond-type tables ──────────────────────────────────────────────────────────

def write_tables(folder):
    """Two small CSV files in the format of scripts/bond_type_casci.py."""
    main = pd.DataFrame({"nele": [4, 4, 4, 2], "norb": [4, 4, 4, 2],
                         "active": ["6 7 8 9", "5 7 8 11", "6 7 8 9", "7 8"],
                         "E_casci": [-78.06, -78.07, -78.06, -78.05],
                         "E_corr_mHa": [-21.6, -31.6, -21.6, -11.6]})
    left = pd.DataFrame({"nele": [4], "norb": [4], "active": ["4 7 8 9"], "E_casci": [-78.08],
                         "E_corr_mHa": [-41.6], "left_out": ["HOMO-1"]})
    main.to_csv(os.path.join(folder, "Ethylene_bond_type_casci.csv"), index=False, encoding="utf-8-sig")
    left.to_csv(os.path.join(folder, "Ethylene_bond_type_casci_left_out.csv"), index=False,
                encoding="utf-8-sig")


def test_bond_type_table_is_best_first_without_repeats(tmp_path):
    write_tables(tmp_path)
    table = bond_type_spaces(str(tmp_path), "Ethylene")
    assert list(table["active"]) == ["4 7 8 9", "5 7 8 11", "6 7 8 9", "7 8"]
    assert list(table["left_out"]) == ["HOMO-1", "", "", ""]


def test_bond_type_space_by_rank(tmp_path):
    write_tables(tmp_path)
    best = bond_type_space(str(tmp_path), "Ethylene", 4, 4, NOCC, NMO, rank=1)
    second = bond_type_space(str(tmp_path), "Ethylene", 4, 4, NOCC, NMO, rank=2)
    assert best["active_orbitals"] == [4, 7, 8, 9] and best["left_out"] == "HOMO-1"
    assert best["E_casci_bond_type"] == -78.08 and best["rule"] == "bond-type"
    assert second["label"] == "HOMO-2 HOMO LUMO LUMO+3"
    with pytest.raises(ChosenSpaceError, match="sizes available"):
        bond_type_space(str(tmp_path), "Ethylene", 6, 8, NOCC, NMO)
    with pytest.raises(FileNotFoundError):
        bond_type_space(str(tmp_path), "Benzene", 4, 4, NOCC, NMO)


# ── integrals for chosen orbitals ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def ethylene():
    mf = run_scf(molecules["Ethylene"], "cc-pVDZ")
    return mf, run_ccsd(mf)


def lowest_state(data):
    """Lowest eigenvalue of the qubit Hamiltonian with nele_cas electrons (exact)."""
    from openfermion import get_sparse_operator
    from openfermion.linalg import jw_number_restrict_operator
    from scipy.sparse.linalg import eigsh
    n = 2 * data["norb_cas"]
    h = jw_number_restrict_operator(get_sparse_operator(qubit_hamiltonian(data), n_qubits=n),
                                    data["nele_cas"], n_qubits=n)
    return float(eigsh(h, k=1, which="SA")[0][0]) if h.shape[0] > 2 else float(np.linalg.eigvalsh(h.toarray())[0])


def hf_energy(data):
    """Energy of the HF determinant from the active-space integrals alone."""
    no, h, g = data["nele_cas"] // 2, data["h1"], data["eri"]
    occ = range(no)
    return (data["e_core"] + 2 * sum(h[i, i] for i in occ)
            + sum(2 * g[i, i, j, j] - g[i, j, j, i] for i in occ for j in occ))


def test_energy_block_as_a_list_is_the_published_space(ethylene):
    """active = list(range(ncore, ncore+norb)) gives exactly the energy-rule integrals."""
    mf, ccsd = ethylene
    old = compute_active_space(mf, 6, 4, 3, ccsd=ccsd)
    new = compute_active_space(mf, 6, 4, 3, ccsd=ccsd, active=[6, 7, 8])
    for k in ("h1", "eri", "t1_active", "t2_active"):
        assert np.allclose(old[k], new[k], atol=1e-12), k
    assert abs(old["e_casci"] - new["e_casci"]) < 1e-12 and abs(old["e_core"] - new["e_core"]) < 1e-12
    assert "active_orbitals" not in old and list(new["active_orbitals"]) == [6, 7, 8]
    ref = load("cc-pVDZ", "Ethylene", 6, 4, 3)                  # the shipped integral file
    assert abs(old["e_casci"] - ref["e_casci"]) < 1e-8


@pytest.mark.parametrize("orbitals", [["HOMO-2", "HOMO", "LUMO"],                         # HOMO-1 skipped
                                      ["HOMO-3", "HOMO", "LUMO", "LUMO+2"],
                                      ["HOMO-2", "HOMO", "LUMO", "LUMO+6", "LUMO+8"]])
def test_chosen_space_energies_agree_four_ways(ethylene, orbitals):
    mf, ccsd = ethylene
    s = chosen_space(orbitals, NOCC, NMO, mo_energy=mf.mo_energy)
    data = compute_active_space(mf, s["ncore"], s["nele_cas"], s["norb_cas"], ccsd=ccsd,
                                active=s["active_orbitals"])
    cas = mcscf.CASCI(mf, s["norb_cas"], s["nele_cas"])
    cas.verbose = 0
    direct = cas.kernel(cas.sort_mo([k + 1 for k in s["active_orbitals"]]))[0]
    assert abs(data["e_casci"] - direct) < 1e-10                  # PySCF CASCI, own setup
    assert abs(lowest_state(data) - data["e_casci"]) < 1e-8       # qubit Hamiltonian, exact
    assert abs(hf_energy(data) - mf.e_tot) < 1e-9                 # frozen core + active HF = E_HF
    assert data["e_casci"] < mf.e_tot


def test_chosen_amplitudes_are_the_right_block(ethylene):
    mf, ccsd = ethylene
    _, t1, t2 = ccsd
    s = chosen_space(["HOMO-3", "HOMO", "LUMO", "LUMO+2"], NOCC, NMO)
    data = compute_active_space(mf, s["ncore"], s["nele_cas"], s["norb_cas"], ccsd=ccsd,
                                active=s["active_orbitals"])
    o, v = [4, 7], [0, 2]                                         # HOMO-3, HOMO; LUMO, LUMO+2
    assert np.array_equal(data["t1_active"], t1[np.ix_(o, v)])
    assert np.array_equal(data["t2_active"], t2[np.ix_(o, o, v, v)])
    import cudaq
    from vqe_cudaq.operators import build_theta0_and_labels_standard
    theta0, _, n = build_theta0_and_labels_standard(data["t1_active"], data["t2_active"], 4, 4)
    assert len(theta0) == n == cudaq.kernels.uccsd_num_parameters(4, 8)


def test_openfermion_hamiltonian_of_chosen_orbitals(ethylene):
    """The geometry route (OpenFermion active-space Hamiltonian) gives the same lowest state."""
    import openfermion
    import openfermionpyscf
    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from vqe_cudaq.xyz import geometry_in_angstrom
    mf, _ = ethylene
    spec = molecules["Ethylene"]
    molecule = openfermionpyscf.run_pyscf(openfermion.MolecularData(
        geometry_in_angstrom(spec), "cc-pVDZ", 1, 0), run_scf=True)
    act = [5, 7, 8]
    occ = [k for k in range(NOCC) if k not in act]
    qop = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian(
        occupied_indices=occ, active_indices=act)))
    data = {"norb_cas": 3, "nele_cas": 4}
    from openfermion import get_sparse_operator
    from openfermion.linalg import jw_number_restrict_operator
    h = jw_number_restrict_operator(get_sparse_operator(qop, n_qubits=6), 4, n_qubits=6)
    cas = mcscf.CASCI(mf, 3, 4)
    cas.verbose = 0
    assert abs(np.linalg.eigvalsh(h.toarray())[0] - cas.kernel(cas.sort_mo([6, 8, 9]))[0]) < 1e-8


def test_wrong_orbital_list_is_refused(ethylene):
    mf, _ = ethylene
    with pytest.raises(SpaceDoesNotFit):
        compute_active_space(mf, 6, 4, 3, active=[5, 6, 7])        # 3 occupied: 6 electrons
    with pytest.raises(SpaceDoesNotFit):
        compute_active_space(mf, 5, 4, 3, active=[5, 7, 8])        # freezes 6, not 5


# ── integral files: chosen and published spaces side by side ─────────────────

def test_chosen_and_energy_files_never_mix(tmp_path):
    """(ncore, nele, norb) = (6, 4, 3) for both [6, 7, 8] (energy) and [5, 7, 8] (chosen)."""
    files = IntegralFiles(str(tmp_path), "Ethylene", molecules["Ethylene"], "cc-pVDZ")
    chosen = files.get(6, 4, 3, active=[5, 7, 8])
    energy = files.get(6, 4, 3)
    names = sorted(os.path.basename(p) for p in files.made)
    assert names == ["No#_00_No(c)_6_Ne(a)_4_No(a)_3_orbs_5-7-8.npz",
                     integral_file_name(5, 6, 4, 3)]              # energy file: published number
    assert chosen["active_orbitals"] == [5, 7, 8] and "active_orbitals" not in energy
    assert abs(energy["e_casci"] - load("cc-pVDZ", "Ethylene", 6, 4, 3)["e_casci"]) < 1e-8
    assert not np.allclose(chosen["h1"], energy["h1"])            # really a different space
    assert abs(chosen["e_casci"] - energy["e_casci"]) > 1e-6
    again = load_integrals(str(tmp_path), "cc-pVDZ", "Ethylene", 6, 4, 3, active=[8, 5, 7])
    assert again["e_casci"] == chosen["e_casci"]                  # order of the list does not matter
    with pytest.raises(FileNotFoundError):
        load_integrals(str(tmp_path), "cc-pVDZ", "Ethylene", 6, 4, 3, active=[4, 7, 8])


def test_published_integral_files_still_load():
    for path in sorted(glob.glob(os.path.join(INTEGRALS, "cc-pVDZ", "Ethylene", "*.npz"))):
        assert "_orbs_" not in os.path.basename(path)
    assert load("cc-pVDZ", "Ethylene", 6, 4, 3)["e_casci"] < load("cc-pVDZ", "Ethylene", 6, 4, 3)["e_hf"]


# ── bond-type results of this repository (skipped when the CSV files are absent) ──

@pytest.mark.skipif(not os.path.exists(os.path.join(BOND_TYPE_DIR, "Ethylene_bond_type_casci.csv")),
                    reason="no bond-type CASCI results in analysis/bond_type_casci_hpc")
@pytest.mark.parametrize("nele, norb", [(4, 3), (4, 4), (6, 8)])
def test_bond_type_spaces_reproduce_their_casci(ethylene, nele, norb):
    mf, _ = ethylene
    s = bond_type_space(BOND_TYPE_DIR, "Ethylene", nele, norb, NOCC, NMO)
    data = compute_active_space(mf, s["ncore"], s["nele_cas"], s["norb_cas"], active=s["active_orbitals"])
    assert abs(data["e_casci"] - s["E_casci_bond_type"]) < 2e-6       # CSV keeps 6 decimals
    best = bond_type_spaces(BOND_TYPE_DIR, "Ethylene")
    same = best[(best["nele"] == nele) & (best["norb"] == norb)]
    assert s["E_casci_bond_type"] == same["E_casci"].min()


# ── command line ──────────────────────────────────────────────────────────────

def parse(*argv):
    from vqe_cudaq.cli import build_parser
    return build_parser().parse_args(["--molecule", "Ethylene", *argv])


def test_cli_manual_rule_gives_the_space():
    from vqe_cudaq.cli import chosen_active_spaces
    args = parse("--active-space-rule", "manual", "--active-orbitals", "HOMO", "HOMO-2", "LUMO", "LUMO+3")
    s, = chosen_active_spaces(args, "Ethylene", molecules["Ethylene"])
    assert s["active_orbitals"] == [5, 7, 8, 11] and (s["ncore"], s["nele_cas"], s["norb_cas"]) == (6, 4, 4)


def test_cli_bond_type_rule_gives_one_space_per_size(tmp_path):
    from vqe_cudaq.cli import chosen_active_spaces
    write_tables(tmp_path)
    args = parse("--active-space-rule", "bond-type", "--bond-type-dir", str(tmp_path), "--sizes", "4,4", "(2e,2o)")
    spaces = chosen_active_spaces(args, "Ethylene", molecules["Ethylene"])
    assert [s["active_orbitals"] for s in spaces] == [[4, 7, 8, 9], [7, 8]]


@pytest.mark.parametrize("argv, message", [
    (("--active-space-rule", "manual"), "needs --active-orbitals"),
    (("--active-space-rule", "bond-type"), "needs --sizes"),
    (("--active-space-rule", "manual", "--active-orbitals", "HOMO", "LUMO", "--space_idx", "2"), "space_idx"),
    (("--active-space-rule", "bond-type", "--sizes", "six"), "electrons,orbitals"),
])
def test_cli_refuses_incomplete_choices(argv, message):
    from vqe_cudaq.cli import chosen_active_spaces
    with pytest.raises(SystemExit, match=message):
        chosen_active_spaces(parse(*argv), "Ethylene", molecules["Ethylene"])


def test_cli_default_is_the_energy_rule():
    assert parse().active_space_rule == "energy"


def test_cli_all_molecules_only_with_the_energy_rule(tmp_path):
    from vqe_cudaq.cli import main
    with pytest.raises(SystemExit, match="one --molecule"):
        main(["--all", "--active-space-rule", "manual", "--active-orbitals", "HOMO", "LUMO",
              "--out_dir", str(tmp_path)])


# ── end-to-end VQE (slow) ─────────────────────────────────────────────────────

@pytest.fixture
def light_cpu(monkeypatch):
    """Double precision CPU, one optimiser chunk of 200 evaluations, no seed search."""
    for k, v in {"TARGET": "qpp-cpu", "BASIS": "cc-pVDZ", "TARGET_PRECISION": "fp64",
                 "OPTIMIZER": "COBYLA", "N_JITTER_RESTARTS": 0, "VQE_MAX_CYCLES": 1,
                 "VQE_CHUNK_MAXITER": 200}.items():
        monkeypatch.setattr(config, k, v)


def chosen_spec(orbitals):
    spec = dict(molecules["Ethylene"])
    spec["valid_active_spaces"] = [chosen_space(orbitals, NOCC, NMO)]
    return spec


@pytest.mark.slow
def test_vqe_on_a_chosen_space_reaches_casci_both_routes(light_cpu, tmp_path):
    """HOMO-2 HOMO LUMO (HOMO-1 skipped), 6 qubits: integral files and geometry give the same run."""
    from vqe_cudaq.driver import run_one_molecule
    fil, = run_one_molecule("Ethylene", chosen_spec(["HOMO-2", "HOMO", "LUMO"]),
                            integrals_dir=str(tmp_path))["active_space_runs"]
    geo, = run_one_molecule("Ethylene", chosen_spec(["HOMO-2", "HOMO", "LUMO"]))["active_space_runs"]
    mf = run_scf(molecules["Ethylene"], "cc-pVDZ")
    cas = mcscf.CASCI(mf, 3, 4)
    cas.verbose = 0
    e_casci = cas.kernel(cas.sort_mo([6, 8, 9]))[0]
    for run in (fil, geo):
        assert run["active_indices"] == {"occupied_indices": [0, 1, 2, 3, 4, 6], "active_indices": [5, 7, 8]}
        assert abs(run["casci"]["E_casci_total"] - e_casci) < 1e-8
        assert abs(run["theta0"]["E_ref_total"] - mf.e_tot) < 1e-8          # theta = 0 is HF
        assert abs(run["vqe"]["E_total"] - e_casci) < 1e-6
        assert run["final_state"]["fidelity"] > 1 - 1e-6
    assert abs(geo["vqe"]["E_total"] - fil["vqe"]["E_total"]) < 1e-8
    assert abs(geo["theta0"]["E_theta0_total"] - fil["theta0"]["E_theta0_total"]) < 1e-8


@pytest.mark.slow
def test_cli_manual_rule_writes_its_own_result_file(light_cpu, tmp_path):
    from vqe_cudaq.cli import main
    main(["--molecule", "Ethylene", "--active-space-rule", "manual", "--active-orbitals", "HOMO-2", "HOMO", "LUMO",
          "--integrals", str(tmp_path / "integrals"), "--target", "qpp-cpu", "--out_dir", str(tmp_path)])
    path, = glob.glob(str(tmp_path / "*_Ethylene_cc-pVDZ_qpp-cpu_COBYLA_manual_VQE_results.pkl"))
    with open(path, "rb") as fh:
        res = pickle.load(fh)["Ethylene"]
    run, = res["active_space_runs"]
    assert res["active_space_rule"] == "manual"
    assert run["space"]["label"] == "HOMO-2 HOMO LUMO" and run["space"]["active_orbitals"] == [5, 7, 8]
    assert abs(run["compare"]["d_vqe_minus_casci"]) < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(BOND_TYPE_DIR, "Ethylene_bond_type_casci.csv")),
                    reason="no bond-type CASCI results in analysis/bond_type_casci_hpc")
def test_cli_bond_type_rule_runs_the_best_space(light_cpu, tmp_path):
    from vqe_cudaq.cli import main
    main(["--molecule", "Ethylene", "--active-space-rule", "bond-type", "--bond-type-dir", BOND_TYPE_DIR,
          "--sizes", "4,3", "--integrals", str(tmp_path / "integrals"), "--target", "qpp-cpu",
          "--out_dir", str(tmp_path)])
    path, = glob.glob(str(tmp_path / "*_bond-type_VQE_results.pkl"))
    with open(path, "rb") as fh:
        run, = pickle.load(fh)["Ethylene"]["active_space_runs"]
    assert abs(run["casci"]["E_casci_total"] - run["space"]["E_casci_bond_type"]) < 2e-6
    assert abs(run["compare"]["d_vqe_minus_casci"]) < 1e-6


def test_cli_old_option_name_still_works():
    """--orbital-rule was the first name of --active-space-rule; commands with it still run."""
    from vqe_cudaq.cli import build_parser
    old = build_parser().parse_args(["--molecule", "Ethylene", "--orbital-rule", "manual"])
    new = build_parser().parse_args(["--molecule", "Ethylene", "--active-space-rule", "manual"])
    assert old.active_space_rule == new.active_space_rule == "manual"


# ── CCSD amplitudes of a chosen space: an independent route ─────────────────────

@pytest.mark.parametrize("orbitals", [["HOMO-2", "HOMO", "LUMO", "LUMO+5"],
                                      ["HOMO-3", "HOMO-2", "HOMO", "LUMO", "LUMO+6", "LUMO+8"]])
def test_amplitudes_equal_a_reordered_ccsd(ethylene, orbitals):
    """Move the chosen MOs into the usual block (frozen occupied, active, other virtual), run CCSD
    again and cut the block: amplitudes, integrals and CASCI must equal the chosen-space mapping."""
    mf, ccsd = ethylene
    s = chosen_space(orbitals, NOCC, NMO)
    act = s["active_orbitals"]
    ours = compute_active_space(mf, s["ncore"], s["nele_cas"], s["norb_cas"], ccsd=ccsd, active=act)
    order = [k for k in range(NOCC) if k not in act] + act + [k for k in range(NOCC, NMO) if k not in act]
    mf2 = run_scf(molecules["Ethylene"], "cc-pVDZ")
    mf2.mo_coeff, mf2.mo_energy, mf2.mo_occ = mf.mo_coeff[:, order], mf.mo_energy[order], mf.mo_occ[order]
    ref = compute_active_space(mf2, s["ncore"], s["nele_cas"], s["norb_cas"], ccsd=run_ccsd(mf2))
    for k in ("t1_active", "t2_active", "h1", "eri"):
        assert np.allclose(ours[k], ref[k], atol=1e-9), k
    assert abs(ours["e_casci"] - ref["e_casci"]) < 1e-10


@pytest.mark.parametrize("orbitals", [["HOMO-2", "HOMO", "LUMO"], ["HOMO-2", "HOMO", "LUMO", "LUMO+5"]])
def test_ccsd_start_lies_between_hf_and_casci(ethylene, orbitals):
    """theta = 0 is the HF state; theta0 (CCSD amplitudes) is below E_HF and above E_CASCI."""
    from conftest import spin_hamiltonian
    from vqe_cudaq.ansatz import energy_expectation
    from vqe_cudaq.operators import build_theta0_and_labels_standard
    mf, ccsd = ethylene
    s = chosen_space(orbitals, NOCC, NMO)
    d = compute_active_space(mf, s["ncore"], s["nele_cas"], s["norb_cas"], ccsd=ccsd, active=s["active_orbitals"])
    c0, h = spin_hamiltonian(d)
    theta0, _, n = build_theta0_and_labels_standard(d["t1_active"], d["t2_active"], s["nele_cas"], s["norb_cas"])
    q = 2 * s["norb_cas"]
    assert abs(c0 + energy_expectation(h, q, s["nele_cas"], np.zeros(n)) - mf.e_tot) < 1e-9
    e0 = c0 + energy_expectation(h, q, s["nele_cas"], theta0)
    assert d["e_casci"] - 1e-9 <= e0 < mf.e_tot
    assert (mf.e_tot - e0) / (mf.e_tot - d["e_casci"]) > 0.9            # the start recovers > 90 %
