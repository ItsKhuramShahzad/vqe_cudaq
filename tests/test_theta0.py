"""The CCSD starting point theta0: packing order and values, and the energy it gives."""

import glob
import os
import pickle

import numpy as np
import pytest

import cudaq
from conftest import RESULTS, load, spin_hamiltonian
from vqe_cudaq.ansatz import energy_expectation
from vqe_cudaq.operators import build_theta0_and_labels_standard as pack
from vqe_cudaq.operators import slice_ccsd_to_active

SPACES = [(6, 4), (6, 5), (6, 6), (6, 7), (4, 3), (4, 4), (4, 5), (2, 3), (2, 4)]


def random_amplitudes(nele, norb, seed=0):
    o, v = nele // 2, norb - nele // 2
    rng = np.random.default_rng(seed)
    return rng.normal(size=(o, v)), rng.normal(size=(o, o, v, v))


@pytest.mark.parametrize("nele, norb", SPACES)
def test_packer_gives_exactly_the_kernel_parameter_count(nele, norb):
    t1, t2 = random_amplitudes(nele, norb)
    theta, labels, n = pack(t1, t2, nele, norb)
    assert n == cudaq.kernels.uccsd_num_parameters(nele, 2 * norb)
    assert len(theta) == len(labels) == n
    assert ("PAD",) not in labels


def test_block_order_is_the_cuda_q_order():
    """[singlesAlpha | singlesBeta | doublesMixed | doublesAlpha | doublesBeta]."""
    t1, t2 = random_amplitudes(4, 5)          # o = 2, v = 3
    _, labels, _ = pack(t1, t2, 4, 5)
    blocks = [lab[0] for lab in labels]
    sizes = {"singlesAlpha": 6, "singlesBeta": 6, "doublesMixed": 36,
             "doublesAlpha": 3, "doublesBeta": 3}
    expected = [name for name, k in sizes.items() for _ in range(k)]
    assert blocks == expected
    # mixed doubles loop: occ alpha i, occ beta j, virtual beta b, then virtual alpha a
    mixed = [lab[1:] for lab in labels if lab[0] == "doublesMixed"]
    assert mixed[:4] == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0)]


def test_packed_values():
    """Singles t1; mixed doubles -2*t2[i,j,a,b]; same-spin doubles 2*(t2[ijab] - t2[jiab])."""
    t1, t2 = random_amplitudes(4, 5, seed=3)
    theta, labels, _ = pack(t1, t2, 4, 5)
    for value, lab in zip(theta, labels):
        kind = lab[0]
        if kind in ("singlesAlpha", "singlesBeta"):
            i, a = lab[1:]
            assert value == t1[i, a]
        elif kind == "doublesMixed":
            i, j, b, a = lab[1:]
            assert value == -2.0 * t2[i, j, a, b]
        else:
            i, j, a, b = lab[1:]
            assert np.isclose(value, 2.0 * (t2[i, j, a, b] - t2[j, i, a, b]))


def test_zero_amplitudes_and_scale():
    t1, t2 = random_amplitudes(6, 5)
    zero, _, _ = pack(np.zeros_like(t1), np.zeros_like(t2), 6, 5)
    assert not zero.any()
    one, _, _ = pack(t1, t2, 6, 5)
    half, _, _ = pack(t1, t2, 6, 5, scale=0.5)
    assert np.allclose(half, 0.5 * one)


def test_slicing_picks_the_active_block():
    nocc, nmo = 5, 9
    t1 = np.arange(nocc * (nmo - nocc), dtype=float).reshape(nocc, nmo - nocc)
    t2 = np.random.default_rng(1).normal(size=(nocc, nocc, nmo - nocc, nmo - nocc))
    occ, vir, t1a, t2a = slice_ccsd_to_active(t1, t2, nocc, nmo, active_orbs=[3, 4, 5, 6])
    assert occ == [3, 4] and vir == [0, 1]
    assert np.array_equal(t1a, t1[3:5, 0:2])
    assert np.array_equal(t2a, t2[3:5, 3:5, 0:2, 0:2])


def seed_energy(d):
    c0, ham = spin_hamiltonian(d)
    ne, no = d["nele_cas"], d["norb_cas"]
    theta, _, _ = pack(d["t1_active"], d["t2_active"], ne, no)
    return c0 + energy_expectation(ham, 2 * no, ne, theta)


@pytest.mark.parametrize("space", [(5, 6, 4), (5, 6, 5), (5, 6, 6), (5, 6, 7), (6, 4, 3),
                                   (6, 4, 4), (6, 4, 5), (7, 2, 3), (7, 2, 4)],
                         ids=lambda s: f"{s[1]}e{s[2]}o")
def test_seed_energy_lies_between_casci_and_hf(space):
    """At theta0 the circuit is already below E_HF and never below E_CASCI (Ethylene)."""
    d = load("cc-pVDZ", "Ethylene", *space)
    e = seed_energy(d)
    assert d["e_casci"] - 1e-9 <= e < d["e_hf"]


def test_seed_reproduces_the_published_cpu_run():
    """Ethylene (6,7) has 204 parameters, so it skips the seed search and the first energy
    the optimiser evaluated is the energy at theta0. The integral file must give it back."""
    path, = glob.glob(os.path.join(RESULTS, "cpu", "*_Ethylene_*.pkl"))
    with open(path, "rb") as fh:
        res = pickle.load(fh)["Ethylene"]
    run, = [r for r in res["active_space_runs"]
            if (r["space"]["nele_cas"], r["space"]["norb_cas"]) == (6, 7)]
    assert run["seed_search"] is None
    published = run["hamiltonian"]["c0"] + run["vqe"]["energy_convergence"][0]
    d = load("cc-pVDZ", "Ethylene", 5, 6, 7)
    assert abs(seed_energy(d) - published) < 1e-9


def test_seed_energy_does_not_depend_on_orbital_signs():
    """A fresh SCF + CCSD fixes the orbital signs its own way, yet gives the same theta0
    energy as the stored file, because amplitudes and integrals share one set of orbitals."""
    from vqe_cudaq import hamiltonian as H
    from vqe_cudaq.molecules import molecules
    mf = H.run_scf(molecules["NH2-"], "sto-3g")
    fresh = H.compute_active_space(mf, 3, 4, 4, ccsd=H.run_ccsd(mf))
    stored = load("sto-3g", "NH2-", 3, 4, 4)
    assert abs(seed_energy(fresh) - seed_energy(stored)) < 1e-8


# ── theta0 comes only from CCSD: no zero, padded or random start ──────

def test_packer_refuses_amplitudes_of_another_active_space():
    """Amplitudes for (4,5) packed as (4,4): wrong count, so an error, never padding or cutting."""
    t1, t2 = random_amplitudes(4, 5)
    with pytest.raises(ValueError, match="expects"):
        pack(t1, t2, 4, 4)


def test_open_shell_molecule_stops_before_any_calculation(monkeypatch):
    """Only closed shell is supported: no HEA, no random start, and no SCF is even run."""
    import openfermionpyscf
    from vqe_cudaq.driver import run_one_molecule
    from vqe_cudaq.molecules import molecules

    def no_scf(*args, **kwargs):
        raise AssertionError("SCF must not run for an open-shell molecule")
    monkeypatch.setattr(openfermionpyscf, "run_pyscf", no_scf)
    triplet = dict(molecules["Ethylene"], multiplicity=3)
    for integrals_dir in (None, "integrals"):
        with pytest.raises(ValueError, match="closed-shell"):
            run_one_molecule("Ethylene", triplet, integrals_dir=integrals_dir)


def test_unconverged_ccsd_stops_the_run(monkeypatch):
    """CCSD that does not converge raises instead of giving a starting point."""
    import pyscf.cc.ccsd
    from vqe_cudaq import hamiltonian as H
    from vqe_cudaq.molecules import molecules
    mf = H.run_scf(molecules["NH2-"], "sto-3g")
    monkeypatch.setattr(pyscf.cc.ccsd.CCSD, "max_cycle", 1)
    with pytest.raises(RuntimeError, match="CCSD did not converge"):
        H.run_ccsd(mf)


def test_geometry_route_stops_when_ccsd_fails(monkeypatch):
    """Without integral files, a CCSD failure stops the molecule; it is not swallowed."""
    from vqe_cudaq import config, driver
    from vqe_cudaq.molecules import molecules

    def failing_ccsd(*args, **kwargs):
        raise RuntimeError("CCSD did not converge")
    monkeypatch.setattr(driver, "run_ccsd", failing_ccsd)
    monkeypatch.setattr(config, "BASIS", "sto-3g")
    with pytest.raises(RuntimeError, match="CCSD did not converge"):
        driver.run_one_molecule("NH2-", molecules["NH2-"])


def test_no_switch_to_turn_ccsd_off():
    from vqe_cudaq import config
    assert not hasattr(config, "RUN_CCSD_REFERENCE")
