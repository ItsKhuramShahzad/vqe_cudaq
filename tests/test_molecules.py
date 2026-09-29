"""The molecule database: the 12 benchmark molecules and their active spaces."""

import os

import pytest
from pyscf import gto

from conftest import ROOT
from vqe_cudaq.molecules import molecules
from vqe_cudaq.xyz import geometry_in_angstrom

BENCHMARK = {"Ethylene", "Benzene", "Naphthalene", "Tetracene", "Pentacene", "NH2-",
             "Methanamide", "Guanine", "Cytosine", "Adenine", "Thymine", "Uracil"}
SPACES = [(6, 4), (6, 5), (6, 6), (6, 7), (4, 3), (4, 4), (4, 5), (2, 3), (2, 4)]


def test_exactly_the_12_benchmark_molecules():
    assert set(molecules) == BENCHMARK


@pytest.mark.parametrize("name", sorted(BENCHMARK))
def test_closed_shell(name):
    assert int(molecules[name]["multiplicity"]) == 1


@pytest.mark.parametrize("name", sorted(BENCHMARK))
def test_same_nine_active_spaces_in_the_same_order(name):
    spaces = [(s["nele_cas"], s["norb_cas"]) for s in molecules[name]["valid_active_spaces"]]
    assert spaces == SPACES


@pytest.mark.parametrize("name", sorted(BENCHMARK))
def test_electron_counts(name):
    """2*ncore + nele_cas is the molecule's electron count, which the geometry confirms."""
    m = molecules[name]
    mol = gto.M(atom=geometry_in_angstrom(m), unit="Angstrom", basis="sto-3g",
                charge=int(m["charge"]), spin=0, verbose=0)
    assert mol.nelectron == int(m["Total Electrons"])
    for s in m["valid_active_spaces"]:
        assert 2 * int(s["ncore"]) + int(s["nele_cas"]) == mol.nelectron


@pytest.mark.parametrize("name", sorted(BENCHMARK))
def test_cc_pvdz_orbital_count(name):
    """'Total Spatial Orbitals' is the number of cc-pVDZ basis functions."""
    m = molecules[name]
    mol = gto.M(atom=geometry_in_angstrom(m), unit="Angstrom", basis="cc-pVDZ",
                charge=int(m["charge"]), spin=0, verbose=0)
    assert mol.nao_nr() == int(m["Total Spatial Orbitals"])


@pytest.mark.parametrize("name", sorted(BENCHMARK))
def test_source_xyz_file_and_picture(name):
    assert molecules[name].get("source") in ("NIST CCCBDB", "PubChem")
    folder = os.path.join(ROOT, "geometries( xyz_files)")
    assert os.path.isfile(os.path.join(folder, f"{name}.xyz"))
    assert os.path.isfile(os.path.join(folder, "images", f"{name}.png"))
