# runner.py
import timeit
import numpy as np
from scipy.optimize import minimize
import cudaq

from .hamiltonian import molecularHamiltonian
from .ansatz import uccsd_kernel, uccsd_num_parameters

def RUN(
    name,
    geometry,
    basis,
    multiplicity,
    charge,
    ncore=0,
    nele_cas=0,
    norb_cas=0,
    ac=True,
    x0=None,
    method="COBYLA",
    maxiter=500,
    target="nvidia",
):
    info = molecularHamiltonian(
        geometry, basis, multiplicity, charge, ncore, nele_cas, norb_cas, ac
    )

    # spin_ham = info["spin_ham"]
    # qubit_count = info["qubit_count"]
    # n_orbitals = info["n_orbitals"]
    # n_electrons = info["n_electrons"]
    # hf_energy = info["hf_energy"]
    # ccsd_energy = info["ccsd_energy"]
    molecular_hamiltonian, qubit_hamiltonian, spin_ham, qubit_count, n_orbitals, n_electrons, hf_energy, ccsd_energy = molecularHamiltonian(
        geometry, basis, multiplicity, charge, ncore, nele_cas, norb_cas, ac
    )
    electron_count = nele_cas if ac else n_electrons
    parameter_count = uccsd_num_parameters(electron_count, qubit_count)

    if x0 is None:
        x0 = np.random.uniform(-0.1, 0.1, parameter_count)

    cudaq.set_target(target)

    quantum_times = []
    exp_vals = []

    def cost(theta):
        start_q = timeit.default_timer()
        try:
            result = cudaq.observe(uccsd_kernel, spin_ham, qubit_count, electron_count, theta)
            val = result.expectation()
            return val
        except Exception as e:
            print(f"[COST ERROR] {e}")
            return 0.0
        finally:
            end_q = timeit.default_timer()
            quantum_times.append(end_q - start_q)

    def callback(xk):
        exp_vals.append(cost(xk))

    start_time = timeit.default_timer()
    result = minimize(
        cost,
        x0,
        method=method,
        callback=callback,
        options={"maxiter": maxiter},
    )
    end_time = timeit.default_timer()
    runtime = end_time - start_time
    qtime = float(np.sum(quantum_times))

    return {
        "name": name,
        "basis_set": basis,
        "hf_energy": hf_energy,
        "ccsd_energy": ccsd_energy,
        "vqe_energy": float(result.fun),
        "n_orbitals": n_orbitals,
        "n_electrons": n_electrons,
        "ncore": ncore,
        "nele_cas": nele_cas,
        "norb_cas": norb_cas,
        "multiplicity": multiplicity,
        "qubit_count": qubit_count,
        "parameter_count": parameter_count,
        "active_s": f"ncore={ncore}, nele_cas={nele_cas}, norb_cas={norb_cas}" if ac else "Complete Hamiltonian",
        "target": target,
        "method": method,
        "runtime": runtime,
        "simulated_quantum_runtime": round(qtime, 2),
        "optimizer_runtime": round(runtime - qtime, 2),
        "quantum_times": quantum_times,
        "energy_convergence": exp_vals,
        "success": bool(result.success),
        "message": str(result.message),
    }

def run_molecule_all_spaces(mol_name, mol, basis="cc-pVDZ", target="nvidia", maxiter=500, method="COBYLA"):
    results = []
    for i, space in enumerate(mol.get("valid_active_spaces", []), start=1):
        try:
            res = RUN(
                name=f"{mol_name}_run_{i}",
                geometry=mol["geometry"],
                basis=basis,
                multiplicity=mol["multiplicity"],
                charge=mol["charge"],
                ncore=space["ncore"],
                nele_cas=space["nele_cas"],
                norb_cas=space["norb_cas"],
                ac=True,
                method=method,
                maxiter=maxiter,
                target=target,
            )
            results.append(res)
        except Exception as e:
            print(f"[Error] {mol_name} | run {i} space {space} failed: {e}")
    return results
