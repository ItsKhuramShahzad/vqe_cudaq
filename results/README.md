# CUDA-Q reference results

VQE results for the 12 closed-shell molecules of the benchmark in the cc-pVDZ basis,
one PKL per molecule and backend, computed with this code before the final benchmark
run (May–June 2026):

- `cpu/`: CUDA-Q `qpp-cpu` target (CPU cluster)
- `gpu/`: CUDA-Q `nvidia` target (NVIDIA H100 GPU)

All runs start from the CCSD amplitudes with the corrected packer
(`theta0.source = "CCSD-sliced (corrected packer, x2 doubles)"`, `theta_scale = 1.0`).
They are the reference against which the MIMIQ version of the benchmark
([mimiq-vqe](https://github.com/ItsKhuramShahzad/mimiq-vqe)) is compared: every
active-space CASCI and Hartree–Fock energy here equals the one computed from the files
in `integrals/` to about 1e-11 Ha.

## Limitations

These runs predate the final benchmark settings, so they are **not** a like-for-like
CPU/GPU timing comparison:

- **Precision:** `gpu/15_MAY_2026_Pentacene` ran in fp32 (no precision was passed and
  the `nvidia` target defaults to fp32). All other GPU runs are fp64; CPU runs are always
  double precision.
- **Missing space:** `gpu/28_JUN_2026_Cytosine` has 8 of the 9 active spaces; (6,7) is
  missing.
- **Extra spaces:** the Ethylene and Pentacene files also contain one active space that
  is no longer part of the benchmark, (8,5) for Ethylene and (4,6) for Pentacene. The
  benchmark uses the 9 spaces common to all molecules (`vqe_cudaq/molecules.py`).
- **Environments:** CPU and GPU runs used different software environments (CUDA-Q 0.12,
  PySCF 2.11, SciPy 1.13 on the CPU cluster; CUDA-Q 0.11, PySCF 2.6.2, SciPy 1.16 on the
  GPU cluster), and the CPU runs did not all use the same number of cores.
- **Tetracene:** the two `Tetracene` files store the molecule under its former key
  `Benzaanthracene` inside the PKL (the geometry is tetracene). The analysis scripts in
  `analysis/` accept both names.

Energies are unaffected by these points. Timings and evaluation counts should be compared
per energy evaluation, not as total runtime.

## Loading

```python
import pickle

with open("results/cpu/12_MAY_2026_Benzene_cc-pVDZ_qpp-cpu_COBYLA_VQE_results.pkl", "rb") as fh:
    data = pickle.load(fh)
res = data["Benzene"]
for run in res["active_space_runs"]:
    s, v = run["space"], run["vqe"]
    print(s["nele_cas"], s["norb_cas"], v["E_total"], run["casci"]["E_casci_total"])
```
