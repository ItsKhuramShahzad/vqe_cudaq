"""From integrals to the CUDA-Q observable: qubit operator, real part, constant split."""

import numpy as np
import pytest
from openfermion import QubitOperator, count_qubits, get_sparse_operator

from conftest import load, spin_hamiltonian
from vqe_cudaq.ansatz import energy_expectation
from vqe_cudaq.hamiltonian import qubit_hamiltonian
from vqe_cudaq.operators import make_qubitop_real, split_constant

SMALL = [("Ethylene", 6, 4, 3), ("Ethylene", 7, 2, 4), ("NH2-", 3, 4, 4),
         ("Benzene", 20, 2, 3), ("Guanine", 36, 6, 4)]


def test_make_qubitop_real_drops_tiny_imaginary_parts():
    q = QubitOperator("Z0", 0.5 + 1e-9j) + QubitOperator("X0 X1", -0.25)
    r = make_qubitop_real(q)
    assert r.terms == {((0, "Z"),): 0.5, ((0, "X"), (1, "X")): -0.25}
    assert all(isinstance(c, float) for c in r.terms.values())


def test_make_qubitop_real_refuses_a_non_hermitian_operator():
    with pytest.raises(ValueError):
        make_qubitop_real(QubitOperator("Y0", 0.1j))


def test_split_constant():
    q = QubitOperator((), -7.5) + QubitOperator("Z0", 0.5) + QubitOperator("Z0 Z1", 0.1)
    c0, rest = split_constant(q)
    assert c0 == -7.5
    assert () not in rest.terms
    assert rest + QubitOperator((), c0) == q


@pytest.mark.parametrize("case", SMALL, ids=lambda c: f"{c[0]}-{c[2]}e{c[3]}o")
def test_hamiltonian_is_real_hermitian_on_2_norb_qubits(case):
    d = load("cc-pVDZ", *case)
    qop = qubit_hamiltonian(d)
    assert count_qubits(qop) == 2 * d["norb_cas"]
    real = make_qubitop_real(qop)          # raises if any coefficient has an imaginary part
    m = get_sparse_operator(real, n_qubits=2 * d["norb_cas"])
    assert abs(m - m.getH()).max() < 1e-12


@pytest.mark.parametrize("case", SMALL, ids=lambda c: f"{c[0]}-{c[2]}e{c[3]}o")
def test_circuit_at_theta_zero_gives_the_hartree_fock_energy(case):
    """c0 + <HF|H|HF> on the CUDA-Q simulator is E_HF: the Hamiltonian, the
    Jordan-Wigner qubit order and the kernel's initial filling all agree."""
    import cudaq
    d = load("cc-pVDZ", *case)
    c0, ham = spin_hamiltonian(d)
    nq, ne = 2 * d["norb_cas"], d["nele_cas"]
    n = cudaq.kernels.uccsd_num_parameters(ne, nq)
    assert abs(c0 + energy_expectation(ham, nq, ne, np.zeros(n)) - d["e_hf"]) < 1e-9


def test_file_hamiltonian_equals_the_geometry_route(ethylene_43):
    """The Hamiltonian built from the integral file equals the one the driver builds from
    the geometry with OpenFermion-PySCF (get_molecular_hamiltonian), and E_HF agrees."""
    import openfermion
    import openfermionpyscf
    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from vqe_cudaq.molecules import molecules
    from vqe_cudaq.xyz import geometry_in_angstrom
    spec = molecules["Ethylene"]
    mol = openfermionpyscf.run_pyscf(openfermion.MolecularData(
        geometry_in_angstrom(spec), "cc-pVDZ", 1, 0), run_scf=True)
    ref = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian(
        occupied_indices=range(6), active_indices=range(6, 9))))
    diff = ref - qubit_hamiltonian(ethylene_43)
    assert max((abs(c) for c in diff.terms.values()), default=0.0) < 1e-9
    assert abs(float(mol.hf_energy) - ethylene_43["e_hf"]) < 1e-9
