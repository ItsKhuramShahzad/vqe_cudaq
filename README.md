# Quantum Benchmarking Of Molecular Ground-State Energy Estimation

This repository provides a **Variational Quantum Eigensolver (VQE)** implementation for molecular ground-state energy estimation using **CUDA-Q**, **OpenFermion**, and **PySCF**.

The framework supports **active-space Hamiltonians**, a **UCCSD ansatz**, and execution on both **GPU (NVIDIA)** and **CPU simulators**. It is designed for research workflows and systematic benchmarking across:
- multiple molecules
- multiple active-space selections
- CPU vs GPU backends

---

## ✨ Features

- Molecular Hamiltonian construction via **OpenFermion + PySCF**
- Active-space reduction using (`ncore`, `nele_cas`, `norb_cas`)
- Fermion-to-qubit mapping via **Jordan–Wigner**
- **UCCSD ansatz** using CUDA-Q kernels
- Classical optimization using **SciPy** (e.g., COBYLA)
- Execution modes:
  - interactive selection mode
  - single molecule with all active spaces
  - all molecules with all defined active spaces
- Outputs include:
  - HF energy
  - VQE energy
  - qubit count
  - parameter count
  - convergence history
  - runtime breakdown (quantum vs optimizer)

---

## 📁 Repository Structure

```
vqe-cudaq/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── vqe_cudaq/
│       ├── __init__.py
│       ├── molecules.py
│       ├── hamiltonian.py
│       ├── ansatz.py
│       ├── runner.py
│       └── cli.py
└── results/
```

---

## ⚙️ Installation

### 1) Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate     # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** CUDA-Q requires an NVIDIA CUDA-capable system for `target=nvidia`.  
> For CPU-only runs, use the `qpp` backend.

---

##  Usage

### Interactive mode

```bash
python -m vqe_cudaq.cli --interactive
```

### Run a single molecule

```bash
python -m vqe_cudaq.cli --molecule Methylene --target nvidia --basis cc-pVDZ --maxiter 500
```

### Run all molecules

```bash
python -m vqe_cudaq.cli --all --target nvidia --basis cc-pVDZ --maxiter 200
```

---

## 📊 Output

Results are stored as `.pkl` files containing:
- HF energy
- VQE energy
- qubit count
- runtime information
- convergence history

---

## 👤 Author

**Khuram Shahzad**  
PhD Researcher - Quantum Computing for Quantum Chemistry
