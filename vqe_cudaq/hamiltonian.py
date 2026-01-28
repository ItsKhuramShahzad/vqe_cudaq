import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
import cudaq


def molecularHamiltonian(geometry, basis, multiplicity, charge, ncore=0, nele_cas=0, norb_cas=0, ac=True):
    # Run PySCF via OpenFermion to get molecular data for given geometry and basis
    """
    Build molecular and qubit Hamiltonians using OpenFermion + PySCF.

    Returns:
        molecular_hamiltonian
        qubit_hamiltonian
        spin_ham (CUDA-Q SpinOperator)
        qubit_count
        n_orbitals
        n_electrons
        hf_energy
        ccsd_energy (placeholder)
    """
    molecule = openfermionpyscf.run_pyscf(openfermion.MolecularData(geometry, basis, multiplicity, charge)) 

    total_orbitals = molecule.n_orbitals
    total_electrons = molecule.n_electrons


    ccsd_energy = 0
    # Active space approximation branch
    if ac:
        # Define frozen core orbital indices (first ncore orbitals)
        frozen_orbitals = range(ncore)
        # Define active space orbital indices (next norb_cas orbitals after frozen core)
        active_orbitals = range(ncore, ncore + norb_cas)
        # Remaining orbitals are virtual orbitals (excluded from active space)
        virtual_orbitals = range(ncore + norb_cas, total_orbitals)

        # Calculate number of electrons in each partition
        frozen_electrons = 2 * len(frozen_orbitals)     # Assume frozen core orbitals are fully occupied (2 electrons each)
        active_electrons = nele_cas                      # Active electrons as provided by user
        virtual_electrons = 0                            # Virtual orbitals usually unoccupied
        qubit_count = 2 * norb_cas
   
        # Get molecular Hamiltonian for the active space only
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=frozen_orbitals,
            active_indices=active_orbitals
        )
       
        # molecular_data = get_mol_hamiltonian(xyz=H2O_mol, spin=0, charge=0, basis='631g', nele_cas=4, norb_cas=4,
        #                              MP2=True, natorb=True, casscf=True, integrals_casscf=True, verbose=True)

    else:
        # No active space: all orbitals treated as active
        frozen_orbitals = []
        active_orbitals = range(total_orbitals)
        virtual_orbitals = []
        active_s = f'all orbital'
        # Assign electrons accordingly: no frozen or virtual orbitals
        frozen_electrons = 0
        active_electrons = total_electrons
        virtual_electrons = 0

        qubit_count = 2 * molecule.n_orbitals
        # Get full molecular Hamiltonian without approximation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()


    # Convert fermionic Hamiltonian to qubit Hamiltonian using Jordan-Wigner transform
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

    # Wrap qubit Hamiltonian for CUDA Quantum backend
    spin_ham = cudaq.SpinOperator(qubit_hamiltonian)

    # Extract Hartree-Fock energy and other properties
    hf_energy = molecule.hf_energy
    n_orbitals = molecule.n_orbitals
    n_electrons = molecule.n_electrons

    # Return all relevant objects and properties for further use
    return molecular_hamiltonian, qubit_hamiltonian, spin_ham, qubit_count, n_orbitals, n_electrons, molecule.hf_energy, ccsd_energy
