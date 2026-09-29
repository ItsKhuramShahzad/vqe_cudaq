"""The active-space integral files in integrals/ and the functions that make and read them."""

import glob
import os
import re
import shutil

import numpy as np
import pytest
from openfermion import get_sparse_operator, jw_get_ground_state_at_particle_number
from pyscf import fci

from conftest import BASES, INTEGRALS, load
from vqe_cudaq import hamiltonian as H
from vqe_cudaq.molecules import molecules

NAME = re.compile(r"^No#_(\d+)_No\(c\)_(\d+)_Ne\(a\)_(\d+)_No\(a\)_(\d+)\.npz$")
ALL_FILES = sorted(glob.glob(os.path.join(INTEGRALS, "*", "*", "*.npz")))


def test_file_count():
    """12 molecules x 9 spaces in cc-pVDZ and 6-31G; STO-3G is too small for 4 NH2- spaces."""
    count = {b: len(glob.glob(os.path.join(INTEGRALS, b, "*", "*.npz"))) for b in BASES}
    assert count == {"cc-pVDZ": 108, "6-31g": 108, "sto-3g": 104}


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: os.path.relpath(p, INTEGRALS))
def test_every_file_is_consistent_with_its_name(path):
    """Each file loads, passes the checks in load_active_space, carries CCSD amplitudes,
    and is the molecule, basis and active space its folder and name say."""
    m = NAME.match(os.path.basename(path))
    assert m, f"unexpected file name {os.path.basename(path)}"
    index, ncore, nele, norb = map(int, m.groups())
    basis, molecule = path.split(os.sep)[-3:-1]
    d = H.load_active_space(path)
    assert (d["ncore"], d["nele_cas"], d["norb_cas"]) == (ncore, nele, norb)
    assert str(d["molecule"]) == molecule and str(d["basis"]) == basis
    assert "t1_active" in d and "t2_active" in d
    space = molecules[molecule]["valid_active_spaces"][index - 1]
    assert (space["ncore"], space["nele_cas"], space["norb_cas"]) == (ncore, nele, norb)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(INTEGRALS, "cc-pVDZ", "*", "*.npz"))),
                         ids=lambda p: os.path.relpath(p, INTEGRALS))
def test_integrals_reproduce_the_stored_casci_energy(path):
    d = H.load_active_space(path)
    e = fci.direct_spin1.kernel(d["h1"], d["eri"], d["norb_cas"], d["nele_cas"])[0] + d["e_core"]
    assert abs(e - d["e_casci"]) < 1e-8


@pytest.mark.parametrize("molecule, ncore, nele, norb", [
    ("Ethylene", 6, 4, 3), ("NH2-", 4, 2, 4), ("Benzene", 19, 4, 4), ("Uracil", 28, 2, 3),
])
def test_qubit_hamiltonian_ground_state_is_casci(molecule, ncore, nele, norb):
    """The Jordan-Wigner Hamiltonian built from the file has the CASCI energy as its
    lowest eigenvalue in the right particle-number sector."""
    d = load("cc-pVDZ", molecule, ncore, nele, norb)
    sparse = get_sparse_operator(H.qubit_hamiltonian(d), n_qubits=2 * norb)
    e0, _ = jw_get_ground_state_at_particle_number(sparse, nele)
    assert abs(e0 - d["e_casci"]) < 1e-9


def test_load_integrals_finds_by_active_space_and_checks_the_request(tmp_path):
    d = load("cc-pVDZ", "Ethylene", 6, 4, 3)
    assert d["path"].endswith("No#_05_No(c)_6_Ne(a)_4_No(a)_3.npz")
    with pytest.raises(FileNotFoundError):
        load("cc-pVDZ", "Ethylene", 6, 4, 9)
    # a file placed under the wrong molecule is refused
    wrong = tmp_path / "cc-pVDZ" / "Benzene"
    wrong.mkdir(parents=True)
    shutil.copy(d["path"], wrong / os.path.basename(d["path"]))
    with pytest.raises(ValueError):
        H.load_integrals(str(tmp_path), "cc-pVDZ", "Benzene", 6, 4, 3)


def test_space_that_does_not_fit_is_reported():
    mf = H.run_scf(molecules["NH2-"], "sto-3g")      # only 7 orbitals in STO-3G
    with pytest.raises(H.SpaceDoesNotFit):
        H.compute_active_space(mf, 2, 6, 7)
    with pytest.raises(H.SpaceDoesNotFit):
        H.compute_active_space(mf, 2, 4, 3)          # wrong electron count


def test_missing_file_is_made_saved_and_reused(tmp_path, monkeypatch):
    """IntegralFiles computes a missing file once (SCF + CCSD), saves it under the
    repository naming, and reads it back without any further PySCF run."""
    spec = molecules["NH2-"]
    files = H.IntegralFiles(str(tmp_path), "NH2-", spec, "sto-3g")
    d = files.get(4, 2, 3)
    assert os.path.basename(d["path"]) == "No#_08_No(c)_4_Ne(a)_2_No(a)_3.npz"
    assert files.made == [d["path"]]
    ref = load("sto-3g", "NH2-", 4, 2, 3)
    for key in ("e_hf", "e_casci", "e_core", "e_ccsd"):
        assert abs(d[key] - ref[key]) < 1e-7, key

    def no_pyscf(*args, **kwargs):
        raise AssertionError("PySCF must not run again")
    monkeypatch.setattr(H, "run_scf", no_pyscf)
    again = H.IntegralFiles(str(tmp_path), "NH2-", spec, "sto-3g").get(4, 2, 3)
    assert again["path"] == d["path"] and abs(again["e_casci"] - d["e_casci"]) == 0.0


def test_dump_script_rewrites_the_shipped_files(tmp_path):
    """scripts/dump_integrals.py writes the same files (names and energies) as in integrals/."""
    from scripts.dump_integrals import dump_molecule
    dump_molecule("NH2-", "sto-3g", str(tmp_path))
    new = sorted(os.listdir(tmp_path / "sto-3g" / "NH2-"))
    shipped = sorted(os.listdir(os.path.join(INTEGRALS, "sto-3g", "NH2-")))
    assert new == shipped and len(new) == 5
    for name in new:
        a = H.load_active_space(str(tmp_path / "sto-3g" / "NH2-" / name))
        b = H.load_active_space(os.path.join(INTEGRALS, "sto-3g", "NH2-", name))
        for key in ("e_hf", "e_casci", "e_core", "e_ccsd"):
            assert abs(a[key] - b[key]) < 1e-7, (name, key)
        # orbitals are fixed only up to sign, so compare what does not depend on it
        assert np.allclose(np.linalg.eigvalsh(a["h1"]), np.linalg.eigvalsh(b["h1"]), atol=1e-6)
