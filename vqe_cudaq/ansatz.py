"""CUDA-Q ansatz kernels and energy evaluation.

Closed-shell (even ``nele_cas``): UCCSD -- ``cudaq.kernels.uccsd``.
Open-shell   (odd  ``nele_cas``): Hardware-Efficient Ansatz (HEA), because
``cudaq.kernels.uccsd`` hard-crashes for odd electron counts. The HEA uses
Ry rotations + CNOT entanglement with exactly ``nele_cas`` electrons -- no
padding, no dummy electrons.
"""

import cudaq


# Closed-shell kernel -- interleaved filling (even electrons)
@cudaq.kernel
def uccsd_kernel_interleaved(qubit_num: int, electron_num: int, thetas: list[float]):
    q = cudaq.qvector(qubit_num)
    filled = 0
    orb = 0
    while filled < electron_num:
        x(q[2 * orb]); filled += 1
        if filled < electron_num:
            x(q[2 * orb + 1]); filled += 1
        orb += 1
    cudaq.kernels.uccsd(q, thetas, electron_num, qubit_num)


# Open-shell HEA kernel -- 3 layers of Ry + CNOT, works for any electron count
@cudaq.kernel
def hea_kernel_openshell(qubit_num: int, electron_num: int, thetas: list[float]):
    qubits = cudaq.qvector(qubit_num)
    for i in range(electron_num):
        x(qubits[i])
    for i in range(qubit_num):
        ry(thetas[i], qubits[i])
    for i in range(qubit_num - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    for i in range(qubit_num):
        ry(thetas[qubit_num + i], qubits[i])
    for i in range(qubit_num - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    for i in range(qubit_num):
        ry(thetas[2 * qubit_num + i], qubits[i])


def hea_num_parameters(qubit_count: int) -> int:
    return 3 * qubit_count


def energy_expectation(spin_ham_nc, qubit_count, nele_cas, theta, open_shell=False):
    if open_shell and (nele_cas % 2 != 0):
        r = cudaq.observe(hea_kernel_openshell, spin_ham_nc, qubit_count, nele_cas, theta)
    else:
        r = cudaq.observe(uccsd_kernel_interleaved, spin_ham_nc, qubit_count, nele_cas, theta)
    return float(r.expectation())
