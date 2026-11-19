# GPU acceleration of molecular ground-state energy estimation with the variational quantum eigensolver using CUDA-Q

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

The methods of the paper, with every equation mapped to the code that evaluates it, are
described in [`docs/METHODS.md`](docs/METHODS.md).

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
- **Active-space selection by bond type**: every MO labelled by the bond it belongs to
  (IBO/LIVVO), active spaces built from bond types (levels may be skipped), scored with CASCI
  and selected with a ratio criterion; usable directly in the VQE (`--active-space-rule`)

---

## 📁 Repository structure

```
vqe_cudaq/
├── README.md
├── pyproject.toml            # package metadata + pytest settings
├── requirements.txt          # direct dependencies, pinned
├── environment/
│   └── vqe_final_pip.txt     # complete package list of the published runs
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
│   ├── orbital_selection.py  # orbital character (π, σ, lone pair); selection rules
│   ├── chosen_spaces.py      # active spaces as orbital lists: manual names, bond-type results
│   ├── bond_type_search.py   # bond-type search: orbital types, type sets, CASCI; one size on the fly
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
│   ├── dump_integrals.py     # writes the integral files from the geometries
│   ├── hf_orbitals.py        # HF level diagrams and orbital isosurfaces, no core
│   ├── compare_active_spaces.py # CASCI energy of each orbital-selection rule
│   ├── bond_type_casci.py    # CASCI by bond type, HOMO-k/LUMO+k left out, selection ratio vs best
│   ├── check_bond_type_casci.py # independent checks of the bond-type CASCI files (fresh CASCI too)
│   ├── bond_type_overview.py # one row per molecule from the summary sheets
│   ├── bond_type_figure.py   # figure: energy order vs bond type, MOs as sums of IBOs/LIVVOs
│   ├── bond_type_paper_figure.py # paper version of that figure (single column) + captions
│   ├── bond_type_manuscript_figure.py # manuscript figure: two molecules, levels split by bond shares
│   ├── pi_plus_sigma_casci.py # CASCI in the π space plus any σ/σ* choice (skipping allowed)
│   ├── run_bond_type_casci.sh # SLURM array: bond-type CASCI + HOMO/LUMO left out, all molecules
│   ├── run_cpu_final.sh      # SLURM array for the final benchmark, CPU node
│   ├── run_gpu_final.sh      # SLURM array for the final benchmark, GPU node
│   └── final_run_common.sh   # shared part: version and commit checks, molecule list
├── tests/                    # pytest suite, runs on a CPU (see "Tests" below)
├── docs/
│   └── METHODS.md            # the paper's methods, equation by equation, mapped to the code
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

All published results were computed in the conda environment `vqe_final`, identical on the
CPU and GPU clusters. To rebuild it exactly:

```bash
conda create -n vqe_final python=3.11.13
conda activate vqe_final
pip install -r environment/vqe_final_pip.txt
```

`environment/vqe_final_pip.txt` lists every package with its version; `requirements.txt` pins the
direct dependencies only: CUDA-Q 0.11.0, PySCF 2.6.2, OpenFermion 1.6.1, OpenFermion-PySCF
0.5, NumPy 1.26.4, SciPy 1.16.0, pandas 2.3.3, Matplotlib 3.11.2 and pytest 9.1.1. Use
these versions to reproduce the results: another CUDA-Q release can give a different energy
for the same parameters, and the final-run job scripts stop if the versions differ.

The package runs from the repository root without installation
(`python -m vqe_cudaq.cli ...`); `pip install -e .` also works and keeps the same pinned
versions. Optional extras, not part of `vqe_final`:

```bash
pip install plotly                              # analysis/scatter_interactive.py
pip install ipython ipywidgets py3Dmol rdkit    # 3D molecule views in Jupyter
```

CUDA-Q runs on Linux; use the `nvidia` target on a GPU and `qpp-cpu` on a CPU.
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
are ignored by Git because cube files can be very large. A systematic way to do
this is the bond-type selection described in the next section.

---

## 🔗 Active-space selection by bond type

The published runs take the active orbitals nearest the HOMO–LUMO gap (energy order).
The bond-type selection chooses them by the **bond they belong to** instead, so an active
space may skip orbitals (for example HOMO and HOMO−2 without HOMO−1).

**The idea, in five steps** (molecule by molecule):

1. **Orbitals.** Restricted Hartree–Fock; the 1s core orbitals are frozen.
2. **Bond type of each MO.** For planar molecules σ and π are separated first. Occupied MOs
   are written in intrinsic bond orbitals (IBOs; Knizia, *J. Chem. Theory Comput.* 9, 4834, 2013),
   virtual MOs in localized valence virtual orbitals (LIVVOs; PySCF `lo.vvo`,
   doi:10.1021/acs.jctc.7b00493). The share of each bond is the squared overlap; the largest
   share gives the type (C–C π, C–H σ, N lone pair, C–C π\*, …). A virtual MO with no
   antibond share ≥ 0.1 is "no bond (outside valence)": it is made of the basis functions
   beyond the minimal basis. Degenerate MOs get one label and are always taken together.
3. **Candidate spaces.** For occupied and virtual MOs separately: each type, pairs of types,
   families (all π, all X–H σ, all lone pairs, all π\*, …), pairs of families and all types;
   every occupied set × every virtual set; inside a set the MOs nearest the gap first; every
   size up to (8e,10o). The same is repeated with one of HOMO−1, −2, −3 or LUMO+1, +2, +3
   left out.
4. **Score.** CASCI of every space, E_corr = (E_CASCI − E_HF) × 1000 mHa, and frozen-core CCSD
   of the whole molecule for reference.
5. **Select.** E_best = most negative E_corr of all spaces of the molecule;
   ratio = (E_corr − E_best) / |E_best|; a space passes when ratio < 0.25 (it recovers at
   least 75 % of E_best).

Ethylene (cc-pVDZ): 1988 different spaces up to (8e,10o), 74 pass; the best is (8e,10o)
with HOMO−1 left out, E_corr = −39.1 mHa, against −29.3 mHa for the energy-ordered (8e,10o).
With T bond types and F families per side, at most S = T + C(T,2) + F + C(F,2) + 2 sets per
side are built and one space per set pair and size, i.e. N ≤ (1+L)·S_occ·S_vir·N_size
(L left-out orbitals, N_size = 30 sizes up to (8e,10o)); a free choice of orbitals would give
4.3 × 10⁹ spaces for ethylene.

### 1. Look at the orbitals (optional)

```bash
# level diagram coloured by bond type, bond shares and isosurfaces of every MO (PDF per molecule)
python scripts/hf_orbitals.py --molecule Ethylene
```

### 2. Run the search

```bash
# one molecule on a laptop: all sizes up to (8e,10o), then the left-out cases
python scripts/bond_type_casci.py --molecule Ethylene --outdir analysis/bond_type_casci \
    --leave-out 1 2 3 --leave-out-virtual 1 2 3

# all 12 molecules on the cluster (SLURM array, one task per molecule; see the script header)
sbatch scripts/run_bond_type_casci.sh
```

| Option | Meaning (default) |
|---|---|
| `--max-electrons`, `--max-orbitals` | largest active space (8, 10) |
| `--max-virtual` | largest number of virtual MOs (same as `--max-orbitals`) |
| `--max-union` | bond types mixed on one side (2) |
| `--types` | use only these bond types, e.g. `"C–C π" "C–C π*"` (all) |
| `--no-outside-valence` | do not use the "no bond (outside valence)" virtual MOs |
| `--leave-out`, `--leave-out-virtual` | HOMO−k / LUMO+k to leave out, e.g. `1 2 3` (none) |
| `--ratio-max` | starting threshold written into the Excel file (0.25) |
| `--summary-only` | only rebuild the Excel file from the CSV files (no calculation) |
| `--max-memory` | PySCF memory in MB (16000) |

A molecule whose main CSV file already exists skips the main search, so an interrupted
cluster job can be restarted. The heavy molecules (tetracene, pentacene) need a compute node:
pentacene took 7 h and 52 GB with 8 cores.

### 3. Results

All in the `--outdir` folder (results are not tracked by Git):

| File | Contents |
|---|---|
| `<molecule>_bond_type_casci.csv` | every CASCI of the type search (raw, one row per type set and size) |
| `<molecule>_bond_type_casci_left_out.csv` | the new spaces with a HOMO−k or LUMO+k left out |
| `<molecule>_bond_type_casci_summary.xlsx` | the file to read, see below |
| `<molecule>_bond_type_casci_run.json` | computer, threads, git commit, settings and time of the run |

The Excel file has four sheets:

| Sheet | Contents |
|---|---|
| `summary` | molecule, E_HF, CCSD, number of spaces, best space, selection criterion, checks, run time |
| `all spaces` | every different space: orbitals, types with counts (e.g. `C–C π (1) + C–H σ (2)`), E_CASCI, E_corr, % of CCSD, gain over energy order, ratio, passes, left-out and skipped orbitals |
| `passes ratio < 0.25` | the passing spaces, best first |
| `VQE (paper spaces)` | the published VQE spaces next to their CASCI |

E_best, the ratio and the pass column are **Excel formulas**: change the threshold cell in the
`summary` sheet and they update (the `passes` sheet is a list made at the threshold of the run;
for another threshold filter `all spaces` on `passes = yes`).

```bash
# Excel files again from the CSV files (e.g. after a cluster run without openpyxl)
python scripts/bond_type_casci.py --molecule Ethylene --outdir analysis/bond_type_casci_hpc --summary-only

# one overview file, one row per molecule
python scripts/bond_type_overview.py --outdir analysis/bond_type_casci_hpc

# independent checks: row counts, formulas, degenerate sets, left-out orbitals, fresh CASCI
python scripts/check_bond_type_casci.py --outdir analysis/bond_type_casci_hpc --molecule Ethylene Benzene
```

### 4. Use the selected spaces in the VQE

```bash
# the best space of each size that passes the ratio criterion
python -m vqe_cudaq.cli --molecule Ethylene --integrals integrals \
    --active-space-rule bond-type --bond-type-dir analysis/bond_type_casci_hpc --sizes all --select best-per-size
```

All options (`--sizes`, `--rank`, `--select`, `--top`, `--ratio-max`, `--bond-type-compute`) are in
[Choosing the active orbitals](#choosing-the-active-orbitals-energy-bond-type-or-by-hand). The VQE
command does not need step 2: a size that is not in `--bond-type-dir` is searched on the spot.
The CCSD amplitudes that start the VQE are cut to the chosen orbitals; this is checked against
CCSD rerun with the chosen orbitals moved into a block (equal to 1e-14).

### 5. Figures

```bash
# working figure: energy order vs bond type at two sizes, and where the types come from
python scripts/bond_type_figure.py --molecule Ethylene --size 6,8
# paper version: single-column level diagram, orbital pictures and suggested captions
python scripts/bond_type_paper_figure.py --molecule Ethylene --size 6,8
# manuscript figure: two molecules side by side (double column), or stacked (single column)
python scripts/bond_type_manuscript_figure.py --molecules Ethylene Cytosine --sizes 6,8 8,10
python scripts/bond_type_manuscript_figure.py --molecules Ethylene Cytosine --stack
```

They write to `analysis/figures/bond_type_selection/`; the first two need scikit-image
(`pip install -e ".[visualization]"`).

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
(main VQE in `vqe.runtime`, seed search in `vqe.seed_search_runtime`) and the run
metadata (package versions, host, SLURM job, GPU, git commit, hash of the `vqe_cudaq`
code). `final_state` holds the final VQE state vector (`psi`, CUDA-Q qubit order), its
spin `S2`, the exact ground-state energy of the active space (`E_exact`) and the
`fidelity` of the VQE state with the exact ground state. A successful run reaches the
CASCI energy of each active space (see `compare.d_vqe_minus_casci`).

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

### Choosing the active orbitals: energy, bond type or by hand

`--active-space-rule` sets how the orbitals of an active space are chosen. The default,
`energy`, takes the block nearest the HOMO–LUMO gap and runs the molecule's published
spaces exactly as before. The other two rules run one molecule at a time; every occupied
orbital that is not chosen stays doubly occupied (frozen).

| `--active-space-rule` | Orbitals | Extra options |
|---|---|---|
| `energy` (default) | nearest the gap, the published spaces | `--space_idx` |
| `bond-type` | the best space of each size by bond type (the search of `scripts/bond_type_casci.py`) | `--sizes`, `--rank`, `--bond-type-dir`, `--bond-type-compute` |
| `manual` | the orbitals you list | `--active-orbitals` |

```bash
# by hand: HOMO-1 left out (4 electrons in 4 orbitals)
python -m vqe_cudaq.cli --molecule Ethylene --target qpp-cpu --integrals integrals \
    --active-space-rule manual --active-orbitals HOMO-2 HOMO LUMO LUMO+3

# the best (6e,8o) and (4e,4o) spaces of the bond-type search (rank 2 = second best)
python -m vqe_cudaq.cli --molecule Ethylene --target qpp-cpu --integrals integrals \
    --active-space-rule bond-type --bond-type-dir analysis/bond_type_casci_hpc --sizes 6,8 4,4 --rank 1
```

The bond-type rule reads the search results in `--bond-type-dir`. A size that is not there
yet (a new molecule, or a size outside the earlier search) is searched on the spot, only that
size: orbital types, every type set and the HOMO-1..-3 / LUMO+1..+3 left-out cases, CASCI of
each, and the frozen-core CCSD of the molecule once. The rows are added to the CSV files in
that folder, so the next run only reads them. `--bond-type-compute never` only reads the files.
`--rank 1` (default) takes the best E_corr of the size, `--rank 2` the second best, and so on.

With `--sizes all` every size up to `--bond-type-max-electrons` / `--bond-type-max-orbitals`
(default 8 and 10) is searched (missing sizes only), and the spaces are chosen with the
selection criterion of the Excel files: E_best = best E_corr of all sizes,
ratio = (E_corr − E_best) / |E_best|, a space passes when ratio < `--ratio-max` (0.25).
`--select` sets which passing spaces get a VQE run:

| `--select` | VQE runs |
|---|---|
| `best-per-size` (default) | the best passing space of each size |
| `top` | the `--top N` best passing spaces overall (default 5) |
| `all` | every passing space |

```bash
python -m vqe_cudaq.cli --molecule Ethylene --integrals integrals --active-space-rule bond-type \
    --sizes all --ratio-max 0.25 --select top --top 5
```

Orbitals are named `HOMO`, `HOMO-k`, `LUMO`, `LUMO+k` (or MO numbers from 0). A chosen
space gets its own integral file, with the orbitals in the name
(`..._No(a)_4_orbs_5-7-8-11.npz`), and its own result file (`..._manual_VQE_results.pkl`,
`..._bond-type_VQE_results.pkl`), so the published files are never touched. The PKL keeps
the orbitals in `space["active_orbitals"]` and `space["label"]`. A bond-type run stops if its
CASCI energy differs from the bond-type table, and a manual space that takes only part of a
degenerate set is run with a warning (`space["warnings"]`).

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
fp64 stops). Only closed-shell molecules are supported, and every active space starts from
the CCSD amplitudes with the corrected CUDA-Q packing; there is no fallback to zero or random
parameters: if the CCSD amplitudes cannot be obtained (CCSD fails or does not converge) or
packed, the run stops with an error.

### Final benchmark on the clusters

`scripts/run_cpu_final.sh` and `scripts/run_gpu_final.sh` run the 12 molecules as a SLURM
array (one molecule per task, its 9 active spaces one after another, one PKL per
molecule) from the integral files, with identical settings, into
`results/final_2026/cpu` and `results/final_2026/gpu`; SLURM logs go to `logs/`. A CPU task
uses 1 core and 8 GB, a GPU task one GPU, 1 core and 16 GB (the largest active space,
14 qubits, needs 4.4 GB including the final-state analysis). Before starting, each task
checks that the environment has exactly the benchmark versions (CUDA-Q 0.11.0, PySCF
2.6.2, SciPy 1.16.0, NumPy 1.26.4, OpenFermion 1.6.1, Python 3.11.13) and that the
package code is the checked-out commit. The same commit must be used on both clusters:
the UCCSD kernel can give a different state for the same parameters in another CUDA-Q
version.

First time on a cluster, clone the repository (public, no GitHub login needed); after
that, `git pull` in the same folder brings every update:

```bash
git clone https://github.com/ItsKhuramShahzad/vqe_cudaq.git   # first time only
cd vqe_cudaq
git pull                                              # later: same commit on both clusters
conda activate vqe_final
python -m pytest tests/test_ansatz.py                 # quick check of the CUDA-Q version
SPACE_IDX=7 sbatch --array=3 scripts/run_cpu_final.sh # short test: Benzene (2,3)
sbatch scripts/run_cpu_final.sh                       # all 12 molecules (GPU: run_gpu_final.sh)
```

Run the scripts from the repository root, without installing the package into the
environment (`python -m vqe_cudaq.cli` needs no install). Do not edit the code on the
cluster: a task stops if `vqe_cudaq/` differs from the commit; change it elsewhere, push,
and pull.

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

## ✅ Tests

The tests check every step of the pipeline against the integral files and reference
results shipped in this repository. They run on the CPU simulator (`qpp-cpu`), so no GPU
is needed. From the repo root:

```bash
pip install pytest
python -m pytest -m "not slow"     # everything except the VQE runs, about 3 minutes
python -m pytest -m slow           # 5 end-to-end VQE runs, about 4 minutes
python -m pytest                   # all 680 tests
```

| File | What it checks |
|---|---|
| `test_molecules.py` | the 12 molecules, their 9 active spaces, electron and orbital counts, XYZ files and pictures |
| `test_integrals.py` | all 320 integral files load, match their names, reproduce CASCI; missing files are made and reused; `dump_integrals.py` rewrites the same files |
| `test_hamiltonian.py` | qubit Hamiltonian is real and Hermitian, its ground state is CASCI, the circuit at θ = 0 gives E_HF |
| `test_ansatz.py` | UCCSD parameter counts, Hartree–Fock filling, rotation angle of each parameter, final-state diagnostics (CPU and GPU) |
| `test_theta0.py` | CCSD → θ₀ packing (order, values, count), θ₀ energy between CASCI and HF, reproduces a published run |
| `test_backend.py` | CUDA-Q version check, fp64 enforcement, run metadata (including the git commit), machine-independent seeds |
| `test_results.py` | reference PKLs: corrected packer, VQE within 1.6 mHa of CASCI, energies match the integral files, CPU and GPU agree |
| `test_orbital_selection.py` | orbital labels against Lewis structures, selection rules, invalid orbital lists |
| `test_bond_type_search.py` | the bond-type search: one size equals the full search and the cluster run, sizes searched once and added, CLI with no result files; slow: VQE for a new user without any files |
| `test_chosen_spaces.py` | chosen orbital lists: names, bond-type tables, integrals checked four ways, files never mixed with the published ones, CLI; slow: VQE on a chosen space reaches CASCI |
| `test_bond_character.py` | bond type and character of every orbital of all 12 molecules: bonds of the geometry, Lewis π counts, lone pairs, reflection, PySCF `vvo`, AVAS |
| `test_analysis.py` | analysis scripts read `results/`, keep the molecule order, and write the tables and LaTeX report |
| `test_driver.py` (slow) | full runs from integral files, from the geometry and from the CLI reach CASCI |

---

## 👤 Authors

**[Khuram Shahzad](https://www.linkedin.com/in/itskhuramshahzad/)**<sup>1,\*</sup> and **[Rosa Di Felice](https://dornsife.usc.edu/profile/rosa-di-felice/)**<sup>2,3,\*</sup>

1. Department of Physics, Informatics and Mathematics, University of Modena and Reggio Emilia, 41125 Modena, Italy
2. Departments of Physics and Astronomy, and Quantitative and Computational Biology, University of Southern California, Los Angeles, CA 90089, USA
3. CNR Institute of Nanoscience, 41125 Modena, Italy

\* Corresponding authors
