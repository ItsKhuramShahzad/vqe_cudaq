# Acceleration of molecular ground-state energy estimation with the variational quantum eigensolver using CUDA-Q

A **Variational Quantum Eigensolver (VQE)** framework for molecular
ground-state energies using **CUDA-Q**, **OpenFermion**, and **PySCF**, plus a
self-contained **analysis / reporting** suite (energy tables, scatter plots,
publication-ready LaTeX figures).

It targets closed-shell **active-space Hamiltonians** with a **UCCSD** ansatz, and
runs on both **GPU (`nvidia`)** and
**CPU (`qpp-cpu`)** CUDA-Q backends for systematic CPU-vs-GPU benchmarking
across:

- the 12 benchmark molecules (hydrocarbons, nitrogen/amide systems, nucleobases)
- multiple active-space selections `(ncore, nele_cas, norb_cas)`
- CPU vs GPU backends and precisions

---

## 🧪 Benchmark molecules

12 closed-shell molecules, cc-pVDZ basis. Geometries are from NIST CCCBDB or PubChem, as
recorded in `vqe_cudaq/molecules.py` (also as XYZ files in `geometries( xyz_files)/`).

**Hydrocarbons**

<table>
  <tr>
    <td align="center"><img src="geometries(%20xyz_files)/images/Ethylene.png" height="90" alt="Ethylene"><br><b>Ethylene</b><br><sub>C<sub>2</sub>H<sub>4</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Benzene.png" height="90" alt="Benzene"><br><b>Benzene</b><br><sub>C<sub>6</sub>H<sub>6</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Naphthalene.png" height="90" alt="Naphthalene"><br><b>Naphthalene</b><br><sub>C<sub>10</sub>H<sub>8</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Tetracene.png" height="90" alt="Tetracene"><br><b>Tetracene</b><br><sub>C<sub>18</sub>H<sub>12</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Pentacene.png" height="90" alt="Pentacene"><br><b>Pentacene</b><br><sub>C<sub>22</sub>H<sub>14</sub></sub></td>
  </tr>
</table>

**Nitrogen and amide systems**

<table>
  <tr>
    <td align="center"><img src="geometries(%20xyz_files)/images/NH2-.png" height="90" alt="Amide anion"><br><b>Amide anion</b><br><sub>NH<sub>2</sub><sup>&minus;</sup></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Methanamide.png" height="90" alt="Methanamide"><br><b>Methanamide</b><br><sub>HCONH<sub>2</sub></sub></td>
  </tr>
</table>

**Nucleobases**

<table>
  <tr>
    <td align="center"><img src="geometries(%20xyz_files)/images/Guanine.png" height="90" alt="Guanine"><br><b>Guanine</b><br><sub>C<sub>5</sub>H<sub>5</sub>N<sub>5</sub>O</sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Cytosine.png" height="90" alt="Cytosine"><br><b>Cytosine</b><br><sub>C<sub>4</sub>H<sub>5</sub>N<sub>3</sub>O</sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Adenine.png" height="90" alt="Adenine"><br><b>Adenine</b><br><sub>C<sub>5</sub>H<sub>5</sub>N<sub>5</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Thymine.png" height="90" alt="Thymine"><br><b>Thymine</b><br><sub>C<sub>5</sub>H<sub>6</sub>N<sub>2</sub>O<sub>2</sub></sub></td>
    <td align="center"><img src="geometries(%20xyz_files)/images/Uracil.png" height="90" alt="Uracil"><br><b>Uracil</b><br><sub>C<sub>4</sub>H<sub>4</sub>N<sub>2</sub>O<sub>2</sub></sub></td>
  </tr>
</table>

| Molecule | Formula | Electrons | Orbitals (cc-pVDZ) | Geometry |
|---|---|---:|---:|---|
| Ethylene | C<sub>2</sub>H<sub>4</sub> | 16 | 48 | NIST CCCBDB |
| Benzene | C<sub>6</sub>H<sub>6</sub> | 42 | 114 | NIST CCCBDB |
| Naphthalene | C<sub>10</sub>H<sub>8</sub> | 68 | 180 | NIST CCCBDB |
| Tetracene | C<sub>18</sub>H<sub>12</sub> | 120 | 312 | NIST CCCBDB |
| Pentacene | C<sub>22</sub>H<sub>14</sub> | 146 | 378 | PubChem |
| Amide anion | NH<sub>2</sub><sup>&minus;</sup> | 10 | 24 | NIST CCCBDB |
| Methanamide | HCONH<sub>2</sub> | 24 | 57 | NIST CCCBDB |
| Guanine | C<sub>5</sub>H<sub>5</sub>N<sub>5</sub>O | 78 | 179 | PubChem |
| Cytosine | C<sub>4</sub>H<sub>5</sub>N<sub>3</sub>O | 58 | 137 | PubChem |
| Adenine | C<sub>5</sub>H<sub>5</sub>N<sub>5</sub> | 70 | 165 | PubChem |
| Thymine | C<sub>5</sub>H<sub>6</sub>N<sub>2</sub>O<sub>2</sub> | 66 | 156 | PubChem |
| Uracil | C<sub>4</sub>H<sub>4</sub>N<sub>2</sub>O<sub>2</sub> | 58 | 132 | PubChem |

Each molecule has the same 9 active spaces, from 6 to 14 qubits, written as
(active electrons, active orbitals):

| Qubits | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| Active spaces | (2,3), (4,3) | (2,4), (4,4), (6,4) | (4,5), (6,5) | (6,6) | (6,7) |

---

## ✨ Features

- Molecular / active-space Hamiltonians via **OpenFermion + PySCF**
- Jordan–Wigner mapping; **UCCSD** kernel (`cudaq.kernels.uccsd`)
- **CCSD-amplitude seeding** of the UCCSD parameters (CUDA-Q-exact packing,
  version-pinned) for a high-quality starting point
- **CASCI** and full-system **CCSD** references computed per molecule
- **Multi-cycle** COBYLA optimization with jittered restarts and convergence control
- **Deterministic seeding** (`hashlib`-based) so CPU and GPU start identically —
  making the speedup comparison meaningful
- Rich per-molecule **PKL** output (HF / CCSD / CASCI / VQE energies, qubit &
  parameter counts, convergence traces, per-cycle and quantum/optimizer timing)
- **Analysis suite**: energy CSV, static & interactive scatter plots, and a
  full PGFPlots/LaTeX report generator
- **Molecular-orbital inspection**: all-MO Molden/CSV export, frontier cube
  files, MO energy diagrams, and positive/negative Jupyter isosurfaces

---

## 📁 Repository structure

```
vqe_cudaq/
├── README.md
├── pyproject.toml            # package metadata
├── requirements.txt
├── LICENSE
├── vqe_cudaq/                # the VQE engine (import as `vqe_cudaq`)
│   ├── __init__.py           # lazy exports; data/config import without CUDA-Q
│   ├── config.py             # run settings (CLI overrides mutate these)
│   ├── molecules.py          # the 12 molecules: geometry, source, 9 active spaces
│   ├── active_space.py       # CAS generation, validation, and analysis
│   ├── xyz.py                # validated XYZ serialization and batch export
│   ├── utils.py              # logging, filenames, pickling, stable hashing, run metadata
│   ├── backend.py            # CUDA-Q target selection, fp64 check, version tripwire
│   ├── operators.py          # qubit-op helpers + CCSD→UCCSD θ₀ packing
│   ├── hamiltonian.py        # integral files: make, save, load; qubit Hamiltonian
│   ├── ansatz.py             # UCCSD kernel + energy expectation
│   ├── vqe.py                # single-chunk / multi-cycle / jitter optimizers
│   ├── insights.py           # PySCF/OpenFermion diagnostics
│   ├── visualization.py      # interactive 3D molecular geometry views
│   ├── orbitals.py           # MO calculation, export, diagrams, 3D isosurfaces
│   ├── driver.py             # run_one_molecule / run_all_molecules
│   └── cli.py                # command-line entry point
├── analysis/                 # reporting tools (no CUDA-Q needed); outputs stay in here
│   ├── energy_csv.py         # per-config energy table -> vqe_energy_table.csv
│   ├── scatter_plots.py      # static scatter plots -> figures/scatter/, tex_out/
│   ├── scatter_interactive.py# interactive hover plots (Plotly HTML)
│   └── latex_report.py       # full PGFPlots/LaTeX figure report -> tex_out/
├── scripts/
│   └── dump_integrals.py     # writes the integral files from the geometries
├── notebooks/
│   └── Molecular_Orbital_Visualization.ipynb
├── integrals/                # active-space integrals + CCSD amplitudes (see its README)
│   ├── cc-pVDZ/<molecule>/No#_01_No(c)_..._No(a)_4.npz   # 9 files per molecule
│   ├── 6-31g/
│   └── sto-3g/
├── geometries( xyz_files)/   # <molecule>.xyz for the 12 molecules
│   └── images/               # <molecule>.png, shown in this README
└── results/                  # CUDA-Q reference results (see its README)
    ├── cpu/                  # 12 PKLs, qpp-cpu target
    └── gpu/                  # 12 PKLs, nvidia target
                              # other run outputs here are git-ignored
```

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
pip install -e .                     # editable install of `vqe_cudaq`
# or: pip install -r requirements.txt
pip install -e ".[interactive]"      # + plotly for the interactive plots
pip install -e ".[visualization]"    # + Jupyter 3D molecular visualization
```

> **CUDA-Q** must be installed separately (see NVIDIA's instructions); it is not
> pulled in by `pip` so installs succeed on non-NVIDIA machines. Use `nvidia`
> for GPU and `qpp-cpu` for CPU-only runs.

---

## 🧬 Interactive 3D molecular geometries

From a Jupyter notebook:

```python
from vqe_cudaq.molecules import molecules
from vqe_cudaq.visualization import visualize_all, visualize_one

visualize_one("Adenine", molecules)
visualize_all(molecules, names=["Ethylene", "Benzene", "Adenine"])
```

Each molecule records its source coordinate unit explicitly. Bonds in these
views are inferred from covalent radii for visualization only; they are not
used by the VQE calculation.

### Export XYZ geometry files

XYZ files are always written in Ångström; source geometries stored in Bohr are
converted automatically.

```bash
# Export all molecules
python -m vqe_cudaq.cli --export-xyz --xyz-dir xyz_files

# Export one molecule
python -m vqe_cudaq.cli --export-xyz --molecule Adenine --xyz-dir xyz_files
```

The same operation is available from Python:

```python
from vqe_cudaq import write_all_xyz, write_xyz
from vqe_cudaq.molecules import molecules

write_xyz("Adenine", molecules["Adenine"], output_dir="xyz_files")
write_all_xyz(molecules, output_dir="xyz_files")
```

### Generate and analyze active spaces

Generate configurations from explicit system dimensions:

```bash
python -m vqe_cudaq.cli --generate-active-spaces \
    --total-orbitals 40 --total-electrons 15 --max-configs 30
```

Generate from one molecule's stored metadata, or analyze its curated spaces:

```bash
python -m vqe_cudaq.cli --generate-active-spaces --molecule Ethylene
python -m vqe_cudaq.cli --analyze-active-spaces --molecule Ethylene
python -m vqe_cudaq.cli --analyze-active-spaces --molecule Ethylene --space_idx 0
```

From Python:

```python
from vqe_cudaq import analyze_active_space, generate_valid_active_spaces

spaces = generate_valid_active_spaces(
    total_orbitals=40,
    total_electrons=15,
    max_configs=30,
)
summary, accounted_electrons = analyze_active_space(
    total_orbitals=40,
    **spaces[0],
    expected_total_electrons=15,
)
```

Frozen-core orbitals are assumed doubly occupied. Generation preserves at
least one unoccupied active orbital and one external virtual orbital. Molecule
orbital counts are basis-dependent; use explicit `--total-orbitals` and
`--total-electrons` values whenever the calculation basis differs from the
metadata source.

### Inspect molecular orbitals before choosing an active space

A complete ready-to-run Jupyter workflow is provided in
[`notebooks/Molecular_Orbital_Visualization.ipynb`](notebooks/Molecular_Orbital_Visualization.ipynb).
It exports every orbital and gives you a dropdown for interactive 3D viewing.

Calculate one molecule and export all canonical MOs to Molden, a complete
energy/occupation table, an energy-level diagram, and cube files around the
HOMO/LUMO frontier:

```bash
python -m vqe_cudaq.cli --export-mos --molecule Ethylene \
    --basis cc-pVDZ --mo-window 3 --mo-grid 80
```

The default cube selection is HOMO-3 through HOMO and LUMO through LUMO+3.
Use zero-based explicit indices when needed, or avoid the relatively large
cube grids:

```bash
python -m vqe_cudaq.cli --export-mos --molecule Adenine \
    --basis 6-31g --mo-indices 18 19 20 21 --mo-grid 100
python -m vqe_cudaq.cli --export-mos --molecule Adenine \
    --basis 6-31g --no-mo-cubes
```

The Molden file contains **every** orbital and can be opened in Molden,
Avogadro, or Jmol. From Jupyter, calculate and inspect the table or render a
cube's positive (blue) and negative (red) phases:

```python
from vqe_cudaq import (
    export_orbital_bundle,
    orbital_table,
    run_orbital_calculation,
    view_orbital_cube,
)
from vqe_cudaq.molecules import molecules

calc = run_orbital_calculation(
    "Ethylene", molecules["Ethylene"], basis="cc-pVDZ"
)
display(orbital_table(calc))

files = export_orbital_bundle(calc, occupied_below=3, virtual_above=3)
view_orbital_cube(calc, files["cubes"][0], isovalue=0.03).show()
```

For active-space selection, inspect orbital energy, occupation, spatial shape,
symmetry/localization, and chemical character together. Do not select solely
by proximity to the HOMO-LUMO gap: include near-degenerate orbitals and both
members of chemically important bonding/antibonding pairs. Generated MO data
are ignored by Git because cube files can be very large.

---

## 🚀 Running the VQE

### Quick start: from the saved integral files (recommended)

The repository ships the active-space integrals **and** the CCSD starting amplitudes
for all 12 molecules and 9 active spaces in `integrals/` (cc-pVDZ, 6-31G and STO-3G).
With `--integrals integrals` a run reads everything it needs from these files: no
Hartree–Fock, no CCSD and no integral transformation of the whole molecule. This is
the easiest way to run the benchmark, and it works on a laptop even for the largest
molecules, whose full-molecule integrals need a compute node with hundreds of GB of
memory. The files give the same Hamiltonians and energies as the geometry route
(checked to 3e-13 Ha).

Run the commands from the repository root:

```bash
# Smallest test: Ethylene, one active space (4 electrons in 3 orbitals, 6 qubits), CPU
python -m vqe_cudaq.cli --molecule Ethylene --space_idx 4 --target qpp-cpu --integrals integrals

# All 9 active spaces of one molecule, CPU
python -m vqe_cudaq.cli --molecule Benzene --target qpp-cpu --integrals integrals

# The same on an NVIDIA GPU (double precision is the default and is enforced)
python -m vqe_cudaq.cli --molecule Benzene --target nvidia --integrals integrals

# All 12 molecules
python -m vqe_cudaq.cli --all --target qpp-cpu --integrals integrals

# Another basis set (folder names in integrals/: cc-pVDZ, 6-31g, sto-3g)
python -m vqe_cudaq.cli --molecule Uracil --basis 6-31g --target qpp-cpu --integrals integrals
```

`--space_idx` selects one active space; the numbering is the same for every molecule:

| `--space_idx` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| (electrons, orbitals) | (6,4) | (6,5) | (6,6) | (6,7) | (4,3) | (4,4) | (4,5) | (2,3) | (2,4) |
| qubits | 8 | 10 | 12 | 14 | 6 | 8 | 10 | 6 | 8 |

The 6- and 8-qubit spaces finish in minutes on a laptop; the 14-qubit space (6,7) takes
hours on a CPU node and much less on a GPU. In STO-3G a few active spaces do not fit the
smaller basis and are skipped (for example 4 of the 9 for NH<sub>2</sub><sup>&minus;</sup>).

**Output.** One PKL per molecule in `--out_dir` (default `pkl_results/`), named
`<date>_<molecule>_<basis>_<target>_COBYLA_VQE_results.pkl`. Runs of different
`--space_idx` of the same molecule on the same day have the same file name, so give each
its own `--out_dir`. For every active space the PKL holds the HF, CCSD, CASCI and final
VQE energies, the circuit energy at the CCSD starting point (`theta0.E_theta0_total`) and
at theta = 0 (`theta0.E_ref_total`, equal to E_HF), the convergence trace, the timings
and the run metadata (package versions, host, SLURM job, GPU, hash of the `vqe_cudaq`
code). A successful run reaches the CASCI energy of each active space (see
`compare.d_vqe_minus_casci`).

If a file is missing (for instance for a new molecule), it is computed with PySCF and
saved in `integrals/` on the fly; `--max-memory` sets the PySCF memory limit in MB.
The file format is described in [`integrals/README.md`](integrals/README.md); the files
are made with `scripts/dump_integrals.py`.

### From the geometry

Without `--integrals`, each run starts from the geometry in `vqe_cudaq/molecules.py`:
Hartree–Fock and CCSD of the whole molecule, then CASCI and the Hamiltonian of each
active space. Use this for a new molecule or basis set; for the large molecules it needs
a compute node with a lot of memory.

```bash
python -m vqe_cudaq.cli --molecule Ethylene --target qpp-cpu --space_idx 4
python -m vqe_cudaq.cli --molecule Ethylene --target nvidia --basis cc-pVDZ
python -m vqe_cudaq.cli --all --target qpp-cpu --optimizer COBYLA
```

### From Python

```python
from vqe_cudaq import run_one_molecule
from vqe_cudaq.molecules import molecules

result = run_one_molecule("Ethylene", molecules["Ethylene"], integrals_dir="integrals")
# without integrals_dir, the run starts from the geometry
```

### Settings

Defaults live in `vqe_cudaq/config.py` and can be overridden on the CLI (`--basis`,
`--target`, `--precision`, `--optimizer`, `--out_dir`, `--integrals`, `--max-memory`).
Every run uses the same settings on CPU and GPU: fp64 is enforced (a run that is not
fp64 stops), and closed-shell runs must start from the CCSD amplitudes with the corrected
CUDA-Q packing.

---

## 📊 Analysis & reporting

The analysis tools read the CPU and GPU results in `results/cpu` and `results/gpu`
and write everything under `analysis/`. They work from any folder:

```bash
# 1) Energy table -> analysis/vqe_energy_table.csv (CPU_DIR/GPU_DIR at the top)
python analysis/energy_csv.py

# 2) Static scatter plots -> analysis/figures/scatter, analysis/tex_out
python analysis/scatter_plots.py

# 3) Interactive hover plots -> analysis/figures/scatter/scatter_interactive.html
python analysis/scatter_interactive.py

# 4) Full LaTeX / PGFPlots figure report -> analysis/tex_out
python analysis/latex_report.py
# another run: --cpu_dir DIR --gpu_dir DIR --out DIR
```

Each per-molecule PKL contains HF / CCSD / CASCI / VQE energies, qubit &
parameter counts, convergence history, and CPU/GPU runtime breakdowns.

---

## 👤 Authors

**Khuram Shahzad**<sup>1,\*</sup> and **[Rosa Di Felice](https://dornsife.usc.edu/profile/rosa-di-felice/)**<sup>2,3,\*</sup>

1. Department of Physics, Informatics and Mathematics, University of Modena and Reggio Emilia, 41125 Modena, Italy
2. Departments of Physics and Astronomy, and Quantitative and Computational Biology, University of Southern California, Los Angeles, CA 90089, USA
3. CNR Institute of Nanoscience, 41125 Modena, Italy

\* Corresponding authors
