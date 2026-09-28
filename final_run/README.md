# Final benchmark run

The script and data used for the final CPU/GPU benchmark runs. One script, one set of
settings, on both backends:

| File | What it is |
|---|---|
| `2026_SEP_Optimized_VQE.py` | the VQE run: one molecule (all or one active space) per call, one PKL out |
| `molecules_data.py` | the 12 closed-shell molecules, geometries and their 9 active spaces |
| `integrals.py` | active-space integrals and CCSD amplitudes: make, save, load |
| `dump_active_integrals.py` | makes the files in `../integrals/` |

## Environment

Python 3.11.13, CUDA-Q 0.11.0, PySCF 2.6.2, SciPy 1.16.0, NumPy 1.26.4,
OpenFermion 1.6.1, OpenFermion-PySCF 0.5. The script refuses CUDA-Q versions whose
UCCSD kernel has not been checked against the parameter packing (0.11, 0.12, 0.14).

## Running

Run from this folder, so the script finds `molecules_data.py` and `integrals.py`.

From the geometry (SCF, CCSD and CASCI with PySCF in the job):

```bash
python 2026_SEP_Optimized_VQE.py --molecule Ethylene --target qpp-cpu
python 2026_SEP_Optimized_VQE.py --molecule Ethylene --target nvidia --space_idx 4
```

From the saved integral files (no SCF or CCSD in the job):

```bash
python 2026_SEP_Optimized_VQE.py --molecule Ethylene --target qpp-cpu --integrals ../integrals
```

With `--integrals`, the active-space Hamiltonian, the Hartree-Fock, CCSD and CASCI
energies and the CCSD amplitudes for the starting point are read from
`<folder>/<basis>/<molecule>/space_*.npz`. A missing file, or one without CCSD amplitudes,
is computed (SCF and CCSD once per molecule) and saved there for later runs;
`--max-memory` (MB) sets the PySCF limit for that. The two routes give the same
Hamiltonian, reference energies and starting point, so the results are the same; the
file route avoids the full-molecule integral transformation, which needs several hundred
GB for the largest molecules.

Other options: `--basis` (default `cc-pVDZ`), `--precision` (default `fp64`; the script
stops if the simulator is not in double precision), `--optimizer` (default `COBYLA`),
`--out_dir`, `--space_idx` (index into the molecule's `valid_active_spaces`).

## Settings

Every closed-shell run starts from the CCSD amplitudes of the whole molecule, cut to the
active space and packed into the CUDA-Q UCCSD parameter order (factor 2 on the doubles,
see `build_theta0_and_labels_standard`); the script stops if any other starting point
would be used. Random numbers are seeded per molecule and active space
(`SEED = 12345` plus a SHA-256 hash), so CPU and GPU runs start identically.

Each PKL also records the software versions, host, SLURM job, core count and GPU
(`run_metadata`), and the energy of the circuit at the Hartree-Fock point and at the
CCSD starting point (`theta0.E_ref_total`, `theta0.E_theta0_total`).
