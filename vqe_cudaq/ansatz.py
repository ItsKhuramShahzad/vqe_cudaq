# ansatz.py
import cudaq

@cudaq.kernel
def uccsd_kernel(qubit_num: int, electron_num: int, thetas: list[float]):
    qubits = cudaq.qvector(qubit_num)
    for i in range(electron_num):
        x(qubits[i])
    cudaq.kernels.uccsd(qubits, thetas, electron_num, qubit_num)



def uccsd_num_parameters(electron_num: int, qubit_num: int) -> int:
    return cudaq.kernels.uccsd_num_parameters(electron_num, qubit_num)
