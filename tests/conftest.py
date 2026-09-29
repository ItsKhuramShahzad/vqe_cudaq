"""Shared fixtures for the vqe_cudaq tests.

All tests run on the CUDA-Q ``qpp-cpu`` target (double precision), so they need
no GPU. They read the integral files in ``integrals/`` and the reference results
in ``results/`` that ship with the repository.
"""

import os
import sys

import pytest

# The test circuits have at most 14 qubits; one OpenMP thread is faster for them than
# many (set before CUDA-Q loads; an OMP_NUM_THREADS from the shell is kept).
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

INTEGRALS = os.path.join(ROOT, "integrals")
RESULTS = os.path.join(ROOT, "results")
BASES = ("cc-pVDZ", "6-31g", "sto-3g")


@pytest.fixture(scope="session", autouse=True)
def cpu_target():
    """Every test runs on the double-precision CPU simulator."""
    import cudaq
    from vqe_cudaq import config
    cudaq.set_target("qpp-cpu")
    config.TARGET = "qpp-cpu"
    yield


def spin_hamiltonian(data):
    """(c0, SpinOperator without the constant) for one integral file, as the driver builds it."""
    import cudaq
    from vqe_cudaq.hamiltonian import qubit_hamiltonian
    from vqe_cudaq.operators import make_qubitop_real, split_constant
    c0, qop_nc = split_constant(make_qubitop_real(qubit_hamiltonian(data)))
    return c0, cudaq.SpinOperator(qop_nc)


def load(basis, molecule, ncore, nele_cas, norb_cas):
    from vqe_cudaq.hamiltonian import load_integrals
    return load_integrals(INTEGRALS, basis, molecule, ncore, nele_cas, norb_cas)


@pytest.fixture(scope="session")
def ethylene_43():
    """Ethylene cc-pVDZ, 4 electrons in 3 orbitals (6 qubits): the smallest real case."""
    return load("cc-pVDZ", "Ethylene", 6, 4, 3)
