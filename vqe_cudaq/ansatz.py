"""CUDA-Q ansatz kernel, energy evaluation and final-state diagnostics.

Closed-shell molecules only (even ``nele_cas``): the UCCSD circuit of
``cudaq.kernels.uccsd`` applied to the Hartree-Fock determinant in the interleaved
convention (qubits 2k and 2k+1 are the alpha and beta spin orbitals of active
orbital k). See docs/METHODS.md, Section 6.
"""

import cudaq


@cudaq.kernel
def uccsd_kernel_interleaved(qubit_num: int, electron_num: int, thetas: list[float]):
    """Hartree-Fock determinant (alpha, beta, alpha, beta, ...) followed by UCCSD."""
    q = cudaq.qvector(qubit_num)
    filled = 0
    orb = 0
    while filled < electron_num:
        x(q[2 * orb]); filled += 1
        if filled < electron_num:
            x(q[2 * orb + 1]); filled += 1
        orb += 1
    cudaq.kernels.uccsd(q, thetas, electron_num, qubit_num)


def energy_expectation(spin_ham_nc, qubit_count, nele_cas, theta):
    """<psi(theta)|H'|psi(theta)> for the Hamiltonian without its constant term."""
    r = cudaq.observe(uccsd_kernel_interleaved, spin_ham_nc, qubit_count, nele_cas, theta)
    return float(r.expectation())


def final_state_diagnostics(qubit_ham, theta, qubit_count, nele_cas, E_vqe):
    """Final UCCSD state and how close it is to the exact ground state (closed shell).

    qubit_ham: the real Jordan-Wigner Hamiltonian including its constant
    (OpenFermion ``QubitOperator``); theta: the optimised parameters; E_vqe: the
    final VQE energy from ``cudaq.observe``.

    The state is returned in CUDA-Q order (qubit k is bit k of the index). OpenFermion
    uses the opposite order (qubit 0 is the most significant bit), so the state is
    reordered before OpenFermion operators act on it; ``E_check`` = <psi|H|psi> must
    then equal ``E_vqe``. The state must be evaluated with the same CUDA-Q version
    that optimised theta: other versions can give a different state for the same theta.
    """
    import numpy as np
    import scipy.sparse.linalg
    from openfermion import (get_sparse_operator, jordan_wigner, jw_number_indices,
                             s_squared_operator)

    n = int(qubit_count)
    psi = np.array(cudaq.get_state(uccsd_kernel_interleaved, n, nele_cas,
                                   [float(t) for t in theta]), dtype=complex)
    psi_of = psi.reshape([2] * n).transpose(list(range(n - 1, -1, -1))).reshape(-1)

    H = get_sparse_operator(qubit_ham, n_qubits=n).tocsc()
    S2 = get_sparse_operator(jordan_wigner(s_squared_operator(n // 2)), n_qubits=n).tocsc()
    E_check = float(np.real(np.vdot(psi_of, H @ psi_of)))
    s2 = float(np.real(np.vdot(psi_of, S2 @ psi_of)))

    # exact ground state(s) in the sector with nele_cas electrons
    idx = jw_number_indices(nele_cas, n)
    H_sec = H[idx, :][:, idx]
    if len(idx) <= 400:
        vals, vecs = np.linalg.eigh(H_sec.toarray())
    else:
        vals, vecs = scipy.sparse.linalg.eigsh(H_sec, k=4, which="SA")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    E_exact = float(vals[0])
    ground = vecs[:, vals < E_exact + 1e-8]          # every degenerate ground state
    psi_sec = psi_of[idx]
    S2_sec = S2[idx, :][:, idx]

    return {
        "psi": psi,                                              # CUDA-Q order
        "E_check": E_check,                                      # <psi|H|psi>
        "E_check_minus_E_vqe": float(E_check - E_vqe),          # ~1e-10: order and state are right
        "S2": s2,                                                # 0 for a pure singlet
        "weight_in_N_sector": float(np.sum(np.abs(psi_sec) ** 2)),   # 1: right electron count
        "E_exact": E_exact,                                      # equals E_CASCI
        "ground_degeneracy": int(ground.shape[1]),
        "S2_exact": float(np.real(np.vdot(ground[:, 0], S2_sec @ ground[:, 0]))),
        "fidelity": float(np.sum(np.abs(ground.conj().T @ psi_sec) ** 2)),  # |<exact|VQE>|^2
    }
