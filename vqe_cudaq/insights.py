"""Lightweight PySCF / OpenFermion diagnostics captured per molecule."""

import numpy as np


def collect_pyscf_insights(mf, molecule_of):
    """Snapshot SCF/MO info (energies, occupations) plus OpenFermion sizes."""
    mol = mf.mol
    return {
        "pyscf": {
            "nelec": int(mol.nelectron),
            "charge": int(mol.charge),
            "spin_2S": int(mol.spin),
            "basis": str(mol.basis),
            "nao_nr": int(mol.nao_nr()),
            "energy_scf_total": float(mf.e_tot),
            "converged": bool(getattr(mf, "converged", False)),
            "mo_energy": np.array(mf.mo_energy, dtype=float),
            "mo_occ": np.array(mf.mo_occ, dtype=float),
            "mo_coeff_shape": tuple(mf.mo_coeff.shape),
        },
        "openfermion": {
            "hf_energy": float(molecule_of.hf_energy),
            "n_orbitals": int(molecule_of.n_orbitals),
            "n_electrons": int(molecule_of.n_electrons),
        }
    }
