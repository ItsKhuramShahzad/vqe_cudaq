"""The CUDA-Q kernels: initial filling, parameter counts, and how much each parameter rotates."""

from math import comb

import numpy as np
import pytest

import cudaq
from conftest import load, spin_hamiltonian
from vqe_cudaq.ansatz import (energy_expectation, hea_kernel_openshell, hea_num_parameters,
                              uccsd_kernel_interleaved)

SPACES = [(6, 4), (6, 5), (6, 6), (6, 7), (4, 3), (4, 4), (4, 5), (2, 3), (2, 4)]


def n_uccsd(nele, norb):
    """Singles (alpha + beta), mixed-spin doubles, same-spin doubles (alpha + beta)."""
    o, v = nele // 2, norb - nele // 2
    return 2 * o * v + (o * v) ** 2 + 2 * comb(o, 2) * comb(v, 2)


@pytest.mark.parametrize("nele, norb", SPACES)
def test_uccsd_parameter_count(nele, norb):
    assert cudaq.kernels.uccsd_num_parameters(nele, 2 * norb) == n_uccsd(nele, norb)


def test_parameter_counts_of_the_benchmark():
    """8 to 204 parameters; (6,7) is the only space above HEAVY_PARAM_THRESHOLD = 150."""
    from vqe_cudaq import config
    counts = {s: n_uccsd(*s) for s in SPACES}
    assert counts[(6, 4)] == 15 and counts[(6, 7)] == 204 and counts[(4, 3)] == 8
    assert [s for s, n in counts.items() if n > config.HEAVY_PARAM_THRESHOLD] == [(6, 7)]


@pytest.mark.parametrize("nele, norb", [(2, 3), (4, 3), (6, 4), (4, 5)])
def test_theta_zero_is_the_hartree_fock_determinant(nele, norb):
    """Interleaved filling: alpha on even qubits, beta on odd ones, lowest orbitals first."""
    n = cudaq.kernels.uccsd_num_parameters(nele, 2 * norb)
    bits = cudaq.sample(uccsd_kernel_interleaved, 2 * norb, nele, [0.0] * n, shots_count=100)
    assert dict(bits.items()) == {"1" * nele + "0" * (2 * norb - nele): 100}


# 2 electrons in 2 orbitals (4 qubits): parameters are
#   0 singlesAlpha (qubit 0 -> 2), 1 singlesBeta (1 -> 3), 2 doublesMixed (0,1 -> 2,3).
# State index = sum of 2**qubit, so HF |q0 q1> is index 3.
@pytest.mark.parametrize("index, excited", [(0, 0b0110), (1, 0b1001), (2, 0b1100)],
                         ids=["singleAlpha", "singleBeta", "doubleMixed"])
def test_each_parameter_rotates_by_half_its_value(index, excited):
    """cudaq.kernels.uccsd applies exp(theta/2 * G): a parameter theta moves amplitude
    sin(theta/2) into the excited determinant. This holds for the doubles, which is
    why the packer writes 2*t2 (operators.build_theta0_and_labels_standard), and on
    this CUDA-Q version it holds for the singles too, which the packer does NOT double.
    If a CUDA-Q upgrade changes this, the packing has to be revisited."""
    t = 0.7
    theta = [0.0, 0.0, 0.0]
    theta[index] = t
    state = np.array(cudaq.get_state(uccsd_kernel_interleaved, 4, 2, theta))
    amp = np.abs(state)
    assert abs(amp[0b0011] - abs(np.cos(t / 2))) < 1e-10
    assert abs(amp[excited] - abs(np.sin(t / 2))) < 1e-10
    assert abs(np.linalg.norm(state) - 1.0) < 1e-12


def test_random_parameters_keep_the_state_normalised_and_the_energy_variational(ethylene_43):
    """Any theta gives a normalised state whose energy is not below E_CASCI."""
    c0, ham = spin_hamiltonian(ethylene_43)
    n = cudaq.kernels.uccsd_num_parameters(4, 6)
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.normal(0.0, 0.5, n)
        state = np.array(cudaq.get_state(uccsd_kernel_interleaved, 6, 4, list(theta)))
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        assert c0 + energy_expectation(ham, 6, 4, theta) >= ethylene_43["e_casci"] - 1e-10


def test_hea_for_odd_electron_counts():
    """Open-shell path: 3 Ry layers (3*qubits parameters), first nele qubits filled.
    At theta = 0 only the X filling and the two CNOT ladders act."""
    assert hea_num_parameters(6) == 18
    q = [1, 1, 1, 0, 0, 0]
    for _ in range(2):
        for i in range(5):
            q[i + 1] ^= q[i]
    bits = cudaq.sample(hea_kernel_openshell, 6, 3, [0.0] * 18, shots_count=100)
    assert dict(bits.items()) == {"".join(map(str, q)): 100}


def test_energy_expectation_picks_the_kernel_by_shell():
    """open_shell=True with odd nele uses the HEA; otherwise UCCSD is used."""
    z0 = cudaq.spin.z(0)
    # HEA: X then Ry(a) on qubit 0, and the CNOTs it controls leave Z0 alone: <Z0> = -cos(a)
    a = 0.9
    theta = np.zeros(18)
    theta[0] = a
    assert abs(energy_expectation(z0, 6, 3, theta, open_shell=True) + np.cos(a)) < 1e-10
    # UCCSD at theta = 0: qubit 0 occupied, <Z0> = -1
    n = cudaq.kernels.uccsd_num_parameters(4, 6)
    assert abs(energy_expectation(z0, 6, 4, np.zeros(n)) + 1.0) < 1e-12
