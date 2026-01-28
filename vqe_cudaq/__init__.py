"""
vqe_cudaq
=========

Variational Quantum Eigensolver (VQE) framework for molecular systems
using CUDA-Q, OpenFermion, and PySCF.

This package provides tools to:
- build molecular and qubit Hamiltonians
- define quantum ansätze
- run VQE optimizations
- benchmark CPU vs GPU backends
"""

# Core API exports (public interface)

from .hamiltonian import molecularHamiltonian
from .molecules import molecules
from .runner import RUN as run_vqe

__all__ = [
    "molecularHamiltonian",
    "molecules",
    "run_vqe",
]
