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
│   ├── xyz.py                # validated XYZ serialization and batch export
│   ├── utils.py              # logging, filenames, pickling, stable hashing
│   ├── backend.py            # CUDA-Q target selection + version tripwire
│   ├── operators.py          # qubit-op helpers + CCSD→UCCSD θ₀ packing
│   ├── hamiltonian.py        # standalone active-space builder (for dump_integrals)
│   ├── ansatz.py             # UCCSD / HEA kernels + energy expectation
│   ├── vqe.py                # single-chunk / multi-cycle / jitter optimizers
│   ├── insights.py           # PySCF/OpenFermion diagnostics
│   ├── visualization.py      # interactive 3D molecular geometry views
│   ├── driver.py             # run_one_molecule / run_all_molecules
│   └── cli.py                # command-line entry point
├── analysis/                 # reporting tools (no CUDA-Q needed)
│   ├── energy_csv.py         # per-config energy table (HF/CCSD/CASCI/VQE, speedup)
│   ├── scatter_plots.py      # static scatter plots (PNG + PDF + PGFPlots .tex)
│   ├── scatter_interactive.py# interactive hover plots (Plotly HTML)
│   └── latex_report.py       # full PGFPlots/LaTeX figure + report generator
├── scripts/
│   └── dump_integrals.py
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
