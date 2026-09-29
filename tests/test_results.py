"""The reference CUDA-Q results in results/cpu and results/gpu."""

import glob
import os
import pickle

import pytest

from conftest import RESULTS, load

OLD_NAMES = {"Benzaanthracene": "Tetracene"}
PKLS = sorted(glob.glob(os.path.join(RESULTS, "*", "*.pkl")))


def read(path):
    with open(path, "rb") as fh:
        (name, res), = pickle.load(fh).items()
    return OLD_NAMES.get(name, name), res


def runs(res):
    return [r for r in res["active_space_runs"] if not r.get("skipped")]


def test_one_file_per_molecule_on_each_backend():
    from vqe_cudaq.molecules import molecules
    for backend in ("cpu", "gpu"):
        names = [read(p)[0] for p in glob.glob(os.path.join(RESULTS, backend, "*.pkl"))]
        assert sorted(names) == sorted(molecules), backend


@pytest.mark.parametrize("path", PKLS, ids=lambda p: os.path.relpath(p, RESULTS))
def test_every_run_used_the_corrected_packer_and_cc_pvdz(path):
    _, res = read(path)
    assert res["basis"] == "cc-pVDZ" and res["optimizer"] == "COBYLA"
    assert res["target"] == ("qpp-cpu" if "/cpu/" in path else "nvidia")
    assert runs(res)
    for r in runs(res):
        assert r["theta0"]["source"].startswith("CCSD-sliced (corrected packer")
        assert r["theta0"]["theta_scale"] == 1.0


@pytest.mark.parametrize("path", PKLS, ids=lambda p: os.path.relpath(p, RESULTS))
def test_vqe_is_within_1_6_mha_of_casci_and_not_below_it(path):
    _, res = read(path)
    for r in runs(res):
        d = r["vqe"]["E_total"] - r["casci"]["E_casci_total"]
        assert -1e-9 <= d < 1.6e-3, r["space"]


@pytest.mark.parametrize("path", PKLS, ids=lambda p: os.path.relpath(p, RESULTS))
def test_reference_energies_match_the_integral_files(path):
    """CASCI and HF in the results equal those in integrals/cc-pVDZ, so the files
    describe the same molecules and orbitals the results were computed with."""
    name, res = read(path)
    checked = 0
    for r in runs(res):
        s = r["space"]
        try:
            d = load("cc-pVDZ", name, int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"]))
        except FileNotFoundError:     # Ethylene (8,5) and Pentacene (4,6): no longer benchmarked
            continue
        assert abs(d["e_casci"] - r["casci"]["E_casci_total"]) < 1e-8
        assert abs(d["e_hf"] - res["references"]["E_hf_full"]) < 1e-8
        checked += 1
    assert checked >= 8


@pytest.mark.parametrize("path", PKLS, ids=lambda p: os.path.relpath(p, RESULTS))
def test_recorded_precision_is_fp64(path):
    """Checked where the file records it; the May 2026 files predate that field."""
    _, res = read(path)
    precision = res.get("cudaq_precision")
    if precision is not None:
        assert "fp64" in str(precision).lower()


def test_cpu_and_gpu_agree_on_the_hamiltonian():
    """Same molecule, same space: same CASCI energy and constant c0 on both backends."""
    cpu = dict(read(p) for p in glob.glob(os.path.join(RESULTS, "cpu", "*.pkl")))
    gpu = dict(read(p) for p in glob.glob(os.path.join(RESULTS, "gpu", "*.pkl")))
    pairs = 0
    for name in cpu:
        by_space = {(r["space"]["nele_cas"], r["space"]["norb_cas"]): r for r in runs(gpu[name])}
        for r in runs(cpu[name]):
            g = by_space.get((r["space"]["nele_cas"], r["space"]["norb_cas"]))
            if g is None:
                continue
            assert abs(r["casci"]["E_casci_total"] - g["casci"]["E_casci_total"]) < 1e-8
            assert abs(r["hamiltonian"]["c0"] - g["hamiltonian"]["c0"]) < 1e-8
            pairs += 1
    assert pairs == 108 + 2          # 12 x 9 benchmark spaces, plus Ethylene (8,5) and Pentacene (4,6)
