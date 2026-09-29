"""End-to-end VQE runs through the driver and the CLI (about a minute each on a laptop).

Run them with ``pytest -m slow``; ``pytest -m "not slow"`` skips them. They use one
optimiser chunk of 200 evaluations and no seed search, which is enough to reach CASCI
for the 8-parameter spaces used here; the production settings are in config.py.
"""

import glob
import os
import pickle

import pytest

from conftest import INTEGRALS, load
from vqe_cudaq import config
from vqe_cudaq.molecules import molecules

pytestmark = pytest.mark.slow


def one_space(name, nele, norb):
    spec = dict(molecules[name])
    spec["valid_active_spaces"] = [s for s in spec["valid_active_spaces"]
                                   if (s["nele_cas"], s["norb_cas"]) == (nele, norb)]
    assert len(spec["valid_active_spaces"]) == 1
    return spec


@pytest.fixture(scope="module", autouse=True)
def light_vqe():
    """Short optimisation for these tests; the config is restored afterwards."""
    light = {"N_JITTER_RESTARTS": 0, "VQE_MAX_CYCLES": 1, "VQE_CHUNK_MAXITER": 200}
    saved = {k: getattr(config, k) for k in light}
    for k, v in light.items():
        setattr(config, k, v)
    yield
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.fixture
def cpu_config(monkeypatch):
    """Double-precision CPU run in cc-pVDZ; restored after the test."""
    monkeypatch.setattr(config, "TARGET", "qpp-cpu")
    monkeypatch.setattr(config, "BASIS", "cc-pVDZ")
    monkeypatch.setattr(config, "TARGET_PRECISION", "fp64")
    monkeypatch.setattr(config, "OPTIMIZER", "COBYLA")
    monkeypatch.setattr(config, "MAX_MEMORY", config.MAX_MEMORY)


@pytest.fixture(scope="module")
def ethylene_from_files(light_vqe):
    config.TARGET, config.BASIS, config.TARGET_PRECISION = "qpp-cpu", "cc-pVDZ", "fp64"
    from vqe_cudaq.driver import run_one_molecule
    return run_one_molecule("Ethylene", one_space("Ethylene", 4, 3), integrals_dir=INTEGRALS)


def test_run_from_integral_files_reaches_casci(ethylene_from_files):
    res = ethylene_from_files
    run, = res["active_space_runs"]
    d = load("cc-pVDZ", "Ethylene", 6, 4, 3)
    assert run["casci"]["E_casci_total"] == d["e_casci"]
    assert abs(run["vqe"]["E_total"] - d["e_casci"]) < 1e-6
    assert run["vqe"]["E_total"] >= d["e_casci"] - 1e-9
    assert run["sizes"] == {"qubits": 6, "uccsd_num_parameters": 8}


def test_run_records_reference_seed_and_provenance(ethylene_from_files):
    res = ethylene_from_files
    run, = res["active_space_runs"]
    d = load("cc-pVDZ", "Ethylene", 6, 4, 3)
    assert abs(run["theta0"]["E_ref_total"] - d["e_hf"]) < 1e-9          # circuit at theta = 0
    assert d["e_casci"] - 1e-9 <= run["theta0"]["E_theta0_total"] < d["e_hf"]
    assert run["theta0"]["source"].startswith("CCSD-sliced (corrected packer")
    assert "from integral file" in run["theta0"]["source"]
    assert res["references"] == {"E_hf_full": d["e_hf"], "E_ccsd_full": d["e_ccsd"]}
    assert "fp64" in str(res["cudaq_precision"]).lower()
    assert res["run_metadata"]["script_name"] == "vqe_cudaq"
    assert res["integrals"]["files_made"] == []                        # nothing recomputed


def test_geometry_route_gives_the_same_run(cpu_config, ethylene_from_files):
    """Without --integrals the driver runs PySCF itself; the energies are the same."""
    from vqe_cudaq.driver import run_one_molecule
    geo, = run_one_molecule("Ethylene", one_space("Ethylene", 4, 3))["active_space_runs"]
    fil, = ethylene_from_files["active_space_runs"]
    assert abs(geo["casci"]["E_casci_total"] - fil["casci"]["E_casci_total"]) < 1e-9
    assert abs(geo["theta0"]["E_theta0_total"] - fil["theta0"]["E_theta0_total"]) < 1e-8
    assert abs(geo["vqe"]["E_total"] - fil["vqe"]["E_total"]) < 1e-8


def test_cli_writes_a_result_file(cpu_config, tmp_path):
    """python -m vqe_cudaq.cli --molecule Ethylene --space_idx 4 --integrals integrals"""
    from vqe_cudaq.cli import main
    main(["--molecule", "Ethylene", "--space_idx", "4", "--integrals", INTEGRALS,
          "--target", "qpp-cpu", "--out_dir", str(tmp_path)])
    path, = glob.glob(str(tmp_path / "*_Ethylene_cc-pVDZ_qpp-cpu_COBYLA_VQE_results.pkl"))
    with open(path, "rb") as fh:
        res = pickle.load(fh)["Ethylene"]
    run, = res["active_space_runs"]
    assert (run["space"]["nele_cas"], run["space"]["norb_cas"]) == (4, 3)
    assert abs(run["compare"]["d_vqe_minus_casci"]) < 1e-6


def test_missing_integral_file_is_made_during_the_run(cpu_config, tmp_path, monkeypatch):
    """--integrals pointing at an empty folder: the file is computed, saved and used."""
    monkeypatch.setattr(config, "BASIS", "sto-3g")
    from vqe_cudaq.driver import run_one_molecule
    res = run_one_molecule("NH2-", one_space("NH2-", 2, 3), integrals_dir=str(tmp_path))
    made = res["integrals"]["files_made"]
    assert [os.path.basename(p) for p in made] == ["No#_08_No(c)_4_Ne(a)_2_No(a)_3.npz"]
    run, = res["active_space_runs"]
    ref = load("sto-3g", "NH2-", 4, 2, 3)
    assert abs(run["casci"]["E_casci_total"] - ref["e_casci"]) < 1e-7
    assert abs(run["vqe"]["E_total"] - ref["e_casci"]) < 1e-6
