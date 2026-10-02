# Active-space integrals

One- and two-electron integrals for the active spaces used in the VQE benchmark,
for 12 closed-shell molecules in three basis sets. With these files a VQE (or CASCI)
calculation can be run without the geometry and without the full-molecule integral
transformation, which is too large for the biggest molecules.

## Layout

```
integrals/<basis>/<molecule>/No#_<NN>_No(c)_<C>_Ne(a)_<E>_No(a)_<O>.npz
```

For example `integrals/cc-pVDZ/Ethylene/No#_05_No(c)_6_Ne(a)_4_No(a)_3.npz`:
Ethylene, cc-pVDZ, 6 frozen core orbitals, 4 electrons in 3 active orbitals (6 qubits).

`NN` is only the position of the space in the list in `vqe_cudaq/molecules.py`.
Identify a space by `No(c)` (ncore), `Ne(a)` (active electrons) and `No(a)`
(active orbitals), not by `NN`.

- **Basis sets:** `sto-3g`, `6-31g`, `cc-pVDZ`
- **Molecules:** NH2-, Ethylene, Methanamide, Benzene, Naphthalene, Tetracene,
  Pentacene, Uracil, Cytosine, Thymine, Adenine, Guanine
- **Geometries:** from `vqe_cudaq/molecules.py` (NIST CCCBDB or PubChem)
- **Active spaces:** 9 per molecule, 6 to 14 qubits

Some active spaces do not exist in sto-3g because the basis has too few orbitals
(NH2- has only 7 in sto-3g, so 4 of its 9 spaces are missing there).

**Status:** complete. All 12 molecules in all three basis sets, 320 files, every one with
the CCSD amplitudes (`t1_active`, `t2_active`, `e_ccsd`).

## What is in each file

| Key | Type | Meaning |
|---|---|---|
| `h1` | `(norb, norb)` | effective one-electron integrals of the active space |
| `eri` | `(norb, norb, norb, norb)` | two-electron integrals `(pq|rs)` of the active space |
| `e_core` | float | constant energy: nuclear repulsion plus the frozen core |
| `ncore` | int | number of frozen core orbitals (doubly occupied) |
| `nele_cas` | int | number of active electrons |
| `norb_cas` | int | number of active orbitals; qubits = 2 x `norb_cas` |
| `n_electrons` | int | total electrons of the molecule |
| `nmo` | int | total number of orbitals in this basis |
| `e_hf` | float | Hartree-Fock energy of the whole molecule |
| `e_casci` | float | CASCI energy of this active space, the exact answer VQE should reach |
| `molecule`, `basis`, `charge`, `multiplicity` | | what the file is |
| `pyscf_version` | str | PySCF version that produced it (2.14.0) |
| `t1_active` | `(nocc_a, nvir_a)` | CCSD single amplitudes inside the active space |
| `t2_active` | `(nocc_a, nocc_a, nvir_a, nvir_a)` | CCSD double amplitudes inside the active space |
| `e_ccsd` | float | CCSD energy of the whole molecule |

`nocc_a = nele_cas / 2` active occupied orbitals, `nvir_a = norb_cas - nocc_a` active virtual
ones.

All energies are in Hartree.

### Conventions

- **Orbitals:** canonical restricted Hartree-Fock orbitals. The active orbitals are
  the `norb_cas` orbitals after the first `ncore`.
- **`h1`:** the one-electron integrals with the frozen core folded in (PySCF
  `CASCI.get_h1eff`). It is not the bare one-electron integral.
- **`eri`:** chemists' notation `(pq|rs)`, full four-index array, spatial orbitals
  (PySCF `CASCI.get_h2eff`, unpacked).
- **CCSD amplitudes:** restricted CCSD on the whole molecule (all orbitals correlated),
  PySCF convention `t1[i,a]`, `t2[i,j,a,b]`, then cut to the active orbitals. Occupied
  indices count from 0; virtual indices count from 0 after the last occupied orbital.
  They are the VQE starting point, not a CCSD of the active space alone.
- **Total energy:** `E = e_core + E_active`, where `E_active` is the energy of the
  active-space Hamiltonian built from `h1` and `eri`.

The active-space Hamiltonian is

```
H = e_core + sum_pq h1[p,q] a+_p a_q + 1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q
```

with the sums over spin included.

## Loading

### With NumPy only

```python
import numpy as np

d = np.load("integrals/cc-pVDZ/Ethylene/No#_05_No(c)_6_Ne(a)_4_No(a)_3.npz")
h1, eri, e_core = d["h1"], d["eri"], float(d["e_core"])
nele, norb = int(d["nele_cas"]), int(d["norb_cas"])
print(float(d["e_hf"]), float(d["e_casci"]))
```

### Check a file: rebuild the CASCI energy with PySCF

```python
from pyscf import fci

e = fci.direct_spin1.kernel(h1, eri, norb, nele)[0] + e_core
print(e - float(d["e_casci"]))     # about 1e-10 or smaller
```

### Qubit Hamiltonian (OpenFermion, Jordan-Wigner)

```python
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial

one, two = spinorb_from_spatial(h1, np.asarray(eri.transpose(0, 2, 3, 1), order="C"))
qubit_ham = jordan_wigner(get_fermion_operator(InteractionOperator(e_core, one, 0.5 * two)))
```

This is the same Hamiltonian that `molecule.get_molecular_hamiltonian(...)` gives
through OpenFermion-PySCF, to about 1e-13. It can be used directly with CUDA-Q
(`cudaq.SpinOperator(qubit_ham)`) or converted for MIMIQ.

### With this repository

```python
from vqe_cudaq.hamiltonian import load_integrals, qubit_hamiltonian

data = load_integrals("integrals", "cc-pVDZ", "Ethylene", 6, 4, 3)   # ncore, nele, norb
qubit_ham = qubit_hamiltonian(data)
```

`load_integrals` finds the file by `ncore`, `nele` and `norb`, checks that it is the
molecule and basis asked for, that `2 * ncore + nele_cas == n_electrons`, and that the
array shapes (integrals and amplitudes) match the active space, and refuses the file if not.

The VQE run uses the files directly:

```bash
python -m vqe_cudaq.cli --molecule Ethylene --space_idx 4 --integrals integrals
```

The starting point is the stored `t1_active` / `t2_active`, packed into the CUDA-Q UCCSD
parameter order by `build_theta0_and_labels_standard` in `vqe_cudaq/operators.py` (the
same packer as the geometry route).

The same files are used by the MIMIQ version of the benchmark
([mimiq-vqe](https://github.com/ItsKhuramShahzad/mimiq-vqe)), so both simulators solve
identical Hamiltonians from identical CCSD starting points.

## How they were made

```bash
python scripts/dump_integrals.py --basis sto-3g 6-31g cc-pVDZ --molecule <name>
```

Restricted Hartree-Fock with PySCF, then PySCF CASCI for each active space. Each file was
checked when it was written: the CASCI energy rebuilt from the saved `h1`, `eri` and
`e_core` must equal the CASCI energy, otherwise the file is not saved.

## Checks

- All files load, pass the consistency checks, and rebuild their stored CASCI energy to
  better than 1e-9 Ha.
- The cc-pVDZ files reproduce the CASCI and Hartree-Fock energies of independent
  earlier CUDA-Q runs (different machine and PySCF version) to 4e-12 Ha.
- VQE run on MIMIQ and on CUDA-Q from a file alone reaches the stored CASCI energy.
- The files with amplitudes were regenerated on a different machine; their integrals and
  energies equal the earlier files without amplitudes to 4e-12 Ha.
- `tests/test_integrals.py` repeats these checks on every file
  (`python -m pytest tests/test_integrals.py`).
