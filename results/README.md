# CUDA-Q reference results

VQE results for the 12 closed-shell molecules of the benchmark in the cc-pVDZ basis,
one PKL per molecule and backend, computed with this code (May–June 2026):

- `cpu/`: CUDA-Q `qpp-cpu` target (CPU cluster)
- `gpu/`: CUDA-Q `nvidia` target (NVIDIA H100 GPU)

All runs start from the CCSD amplitudes with the corrected packer
(`theta0.source = "CCSD-sliced (corrected packer, x2 doubles)"`, `theta_scale = 1.0`).
They are the reference against which the MIMIQ version of the benchmark
([mimiq-vqe](https://github.com/ItsKhuramShahzad/mimiq-vqe)) is compared: every
active-space CASCI and Hartree–Fock energy here equals the one computed from the files
in `integrals/` to about 1e-11 Ha.

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
