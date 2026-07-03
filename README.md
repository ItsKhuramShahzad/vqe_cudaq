# Quantum Benchmarking of Molecular Ground-State Energy Estimation

A **Variational Quantum Eigensolver (VQE)** framework for molecular
ground-state energies using **CUDA-Q**, **OpenFermion**, and **PySCF**, plus a
self-contained **analysis / reporting** suite (energy tables, scatter plots,
publication-ready LaTeX figures).

It targets **active-space Hamiltonians** and a **UCCSD** ansatz (with an
open-shell **HEA** fallback), and runs on both **GPU (`nvidia`)** and
**CPU (`qpp-cpu`)** CUDA-Q backends for systematic CPU-vs-GPU benchmarking
across:

- multiple molecules (15 built in)
- multiple active-space selections `(ncore, nele_cas, norb_cas)`
- CPU vs GPU backends and precisions

---

## ✨ Features

- Molecular / active-space Hamiltonians via **OpenFermion + PySCF**
- Jordan–Wigner mapping; **UCCSD** kernel (closed-shell) and **HEA** (open-shell)
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
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── vqe_cudaq/                 # the VQE engine (import as `vqe_cudaq`)
│   ├── __init__.py            # lazy exports; data/config import without CUDA-Q
│   ├── config.py             # run settings (CLI overrides mutate these)
│   ├── molecules.py          # molecule database (geometry + valid active spaces)
│   ├── active_space.py       # CAS generation, validation, and analysis
│   ├── xyz.py                # validated XYZ serialization and batch export
│   ├── utils.py              # logging, filenames, pickling, stable hashing
│   ├── backend.py            # CUDA-Q target selection + version tripwire
│   ├── operators.py          # qubit-op helpers + CCSD→UCCSD θ₀ packing
│   ├── hamiltonian.py        # standalone active-space builder (for dump_integrals)
│   ├── ansatz.py             # UCCSD / HEA kernels + energy expectation
│   ├── vqe.py                # single-chunk / multi-cycle / jitter optimizers
│   ├── insights.py           # PySCF/OpenFermion diagnostics
│   ├── visualization.py      # interactive 3D molecular geometry views
│   ├── orbitals.py           # MO calculation, export, diagrams, 3D isosurfaces
│   ├── driver.py             # run_one_molecule / run_all_molecules
│   └── cli.py                # command-line entry point
├── analysis/                 # reporting tools (no CUDA-Q needed)
│   ├── energy_csv.py         # per-config energy table (HF/CCSD/CASCI/VQE, speedup)
│   ├── scatter_plots.py      # static scatter plots (PNG + PDF + PGFPlots .tex)
│   ├── scatter_interactive.py# interactive hover plots (Plotly HTML)
│   └── latex_report.py       # full PGFPlots/LaTeX figure + report generator
├── scripts/
│   └── dump_integrals.py
├── notebooks/
│   └── Molecular_Orbital_Visualization.ipynb
├── integrals/                # cached active-space integrals (.npz)
├── geometries( xyz_files)/
└── results/                  # PKL outputs (git-ignored)
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
visualize_all(molecules, names=["Methylene", "Benzene", "Adenine"])
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
python -m vqe_cudaq.cli --generate-active-spaces --molecule Methylene
python -m vqe_cudaq.cli --analyze-active-spaces --molecule Methylene
python -m vqe_cudaq.cli --analyze-active-spaces --molecule Methylene --space_idx 0
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
python -m vqe_cudaq.cli --export-mos --molecule Methylene \
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
    "Methylene", molecules["Methylene"], basis="cc-pVDZ"
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

```bash
# Single molecule, GPU
python -m vqe_cudaq.cli --molecule Methylene --target nvidia --basis cc-pVDZ

# Single molecule, CPU, one active space only
python -m vqe_cudaq.cli --molecule Benzene --target qpp-cpu --space_idx 0

# All molecules
python -m vqe_cudaq.cli --all --target qpp-cpu --optimizer COBYLA
```

Or from Python:

```python
from vqe_cudaq import run_one_molecule
from vqe_cudaq.molecules import molecules

result = run_one_molecule("Methylene", molecules["Methylene"])
```

Default run settings live in `vqe_cudaq/config.py` and can be overridden on the
CLI (`--basis`, `--target`, `--precision`, `--optimizer`, `--out_dir`).

---

## 📊 Analysis & reporting

Run the CPU and GPU benchmarks, then point the analysis tools at the two result
folders. From the repo root:

```bash
# 1) Build the per-configuration energy table (edit CPU_DIR/GPU_DIR at the top)
python analysis/energy_csv.py

# 2) Static scatter plots (PNG + PDF + PGFPlots .tex)
python analysis/scatter_plots.py

# 3) Interactive hover plots (Plotly HTML)
python analysis/scatter_interactive.py

# 4) Full LaTeX / PGFPlots figure report
python analysis/latex_report.py \
    --cpu_dir results/pkl_results/cpu_pkl_results \
    --gpu_dir results/pkl_results/gpu_pkl_results \
    --out analysis/tex_out
```

Each per-molecule PKL contains HF / CCSD / CASCI / VQE energies, qubit &
parameter counts, convergence history, and CPU/GPU runtime breakdowns.

---

## 👤 Author

**Khuram Shahzad** — PhD Researcher, Quantum Computing for Quantum Chemistry
