# ============================================================
# 2026_SEP_Optimized_VQE.py  (created 18 Sep 2026 from 2026_JUNE_Optimized_VQE.py)
# Purpose: final, consistent CPU/GPU rerun for the paper
#          (same code, same env, same settings on both backends).
#
# Environment: conda env "vqe_final", identical on lyra (GPU) and fe02 (CPU):
#   Python 3.11.13, CUDA-Q 0.11.0, PySCF 2.6.2, SciPy 1.16.0,
#   NumPy 1.26.4, OpenFermion 1.6.1, OpenFermion-PySCF 0.5
#
# Changes vs JUNE version:
#  - [2a] fp64 enforced:
#       * TARGET_PRECISION default "fp64" (global and --precision default)
#       * configure_cudaq_target() raises if the active precision is not fp64
#         (nvidia defaults to fp32; two earlier GPU runs had no precision set)
#  - [2e] (done) assert corrected CCSD packer + THETA_SCALE = 1.0 for closed shell
#  - [2b] (done) run_metadata(): versions, host, SLURM ids, cores, GPU, script hash
#  - [2c] (done) store E(theta=0) and E(theta0) for every active space
#  - [2d] (todo) time the seed search separately (seed_search_runtime)
#  - [2g] (done) --integrals DIR: read the Hamiltonian, E_HF, E_CASCI and the CCSD
#         amplitudes from saved integral files (integrals.py), no SCF/CCSD in the job.
#         Missing files are computed and saved there. Same files as the MIMIQ runs,
#         so both backends solve identical Hamiltonians. Without --integrals the
#         geometry route is unchanged.
#
# molecules_data.py: (done) 9 spaces per molecule; Benzaanthracene renamed Tetracene
#                    (the geometry is tetracene: linear, D2h)
# ============================================================
#
# ---- JUNE 2026 changes (previous version) ----
#  - CORRECTED build_theta0_and_labels_standard:
#       * uses CUDA-Q's exact block order [Sa | Sb | Dmix | Daa | Dbb]
#       * applies the empirically-verified factor of 2 on all doubles
#       * antisymmetrizes same-spin doubles as t[i,j,a,b] - t[j,i,a,b]
#    Verified to recover ~99.5% of the correlation energy at the seed
#    point on H2 (CCSD=FCI) and stretched H4. The old packing put the
#    starting energy *above* HF.
#  - THETA_SCALE bumped from 0.3 to 1.0 (the 0.3 was compensating for
#    the broken packer; with the correct packer we want full strength).
#  - Version tripwire on cudaq right after import (pin: 0.11/0.12/0.14).
#  - hash() replaced with hashlib.sha256 in local_rng derivation so the
#    seed is identical across Python sessions / machines without needing
#    PYTHONHASHSEED.
#
# Open-shell branch (HEA path) is unchanged on purpose; Option B (custom
# spin-UCCSD kernel) is the follow-up work for open-shell molecules.
# ============================================================


import os
import re
import pickle
import time
import timeit
import hashlib
import platform
import socket
import subprocess
import numpy as np
from scipy.optimize import minimize
import scipy

import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.ops import QubitOperator
import cudaq

# ── VERSION TRIPWIRE ──────────────────────────────────────────────────
# The factor-of-2 packing relies on the specific gate decomposition
# inside cudaq.kernels.uccsd's double_excitation_opt (8x rz(0.125*theta)).
# Verified identical across 0.11.x, 0.12.x, 0.14.x. If you upgrade to a
# new version, redo the H2 sanity check before trusting the energies.
_CUDAQ_VER = cudaq.__version__
if not any(v in _CUDAQ_VER for v in ("0.11", "0.12", "0.14")):
    raise RuntimeError(
        f"Packing verified against cudaq 0.11/0.12/0.14; got {_CUDAQ_VER}"
    )

# ──────────────────────────────────────────────────────────────────────

from pyscf import cc, mcscf
import pyscf
from molecules_data import molecules
from integrals import IntegralFiles, SpaceDoesNotFit, qubit_hamiltonian

# -----------------------------
# Global Settings (edit these)
# -----------------------------
TAG = time.strftime("%d_%b_%Y").upper()  # e.g., "23_FEB_2026"
BASIS = "cc-pVDZ"
TARGET = "qpp-cpu"        # e.g., "nvidia" or "qpp-cpu"
OPTIMIZER = "COBYLA"
TARGET_PRECISION = "fp64" # None = default, "fp32" or "fp64" for nvidia
RUN_CCSD_REFERENCE = True

SEED = 12345
rng_global = np.random.default_rng(SEED)

# ----- optimizer settings -----
TOL = 1e-10
COBYLA_RHOBEG = 0.2

# Full strength now that the packing is correct. Previously 0.3, which
# was a band-aid for the broken packer.
THETA_SCALE = 1.0

N_JITTER_RESTARTS = 3
JITTER_SCALE = 5e-3

DIAG_MAX_PARAMS = 0
DIAG_EPS = 1e-3

VQE_EPS_E = 1e-6
VQE_PATIENCE = 3
VQE_MAX_CYCLES = 25
VQE_CHUNK_MAXITER = 600
VQE_JITTER_BETWEEN_CYCLES = True
VQE_JITTER_BETWEEN_SCALE = 5e-4

HEAVY_PARAM_THRESHOLD = 150
HEAVY_RESTARTS = 0
HEAVY_RHOBEG = 0.05

VERBOSE = False
PRINT_EVERY_CYCLE = False

JW_IMAG_TOL = 1e-6

MAX_MEMORY = 16000   # MB, PySCF limit when a missing integral file has to be made [2g]

# -----------------------------
# Utilities
# -----------------------------
def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-\+]", "", name)
    return name

def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _stable_hash(obj) -> int:
    """Deterministic hash across Python sessions / machines.

    Python's built-in hash() randomizes string hashing per-process unless
    PYTHONHASHSEED is fixed before interpreter start, which means
    `hash((mol_name, ...))` differs between runs and between CPU/GPU
    machines. hashlib.sha256 is deterministic by construction.
    """
    return int(hashlib.sha256(repr(obj).encode()).hexdigest(), 16) % (2**31)

def configure_cudaq_target():
    """
    Set CUDA-Q target and print/check backend precision.

    For qpp-cpu:
        default is double precision.

    For nvidia:
        default is usually fp32 unless option="fp64" is given.
    """
    if TARGET == "nvidia" and TARGET_PRECISION is not None:
        cudaq.set_target(TARGET, option=TARGET_PRECISION)
    else:
        cudaq.set_target(TARGET)

    print("CUDA-Q target:", cudaq.get_target().name, flush=True)

    try:
        precision = str(cudaq.get_target().get_precision())
    except Exception as e:
        precision = f"unknown ({e})"

    print("CUDA-Q precision:", precision, flush=True)
    if "fp64" not in precision.lower():
        raise RuntimeError(f"Expected fp64 simulation precision, got: {precision}")
    return precision
def run_metadata():
    """[2b] Record software versions, machine, SLURM job and script hash."""
    meta = {
        "python": platform.python_version(),
        "cudaq": cudaq.__version__,
        "pyscf": pyscf.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "openfermion": openfermion.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "os_cpu_count": os.cpu_count(),
        "script_name": os.path.basename(__file__),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        meta["gpu"] = out.stdout.strip() or None
    except Exception:
        meta["gpu"] = None
    return meta


def make_qubitop_real(qop: QubitOperator, tol: float = 1e-6) -> QubitOperator:
    qreal = QubitOperator()
    for term, coeff in qop.terms.items():
        c = complex(coeff)
        if abs(c.imag) > tol:
            raise ValueError(f"Imag coeff too big (>{tol}): {c} on term {term}")
        re = float(c.real)
        if abs(re) > 0.0:
            qreal += QubitOperator(term, re)
    return qreal

def split_constant(qop: QubitOperator):
    c0 = float(np.real(qop.terms.get((), 0.0)))
    qnc = QubitOperator()
    for term, coeff in qop.terms.items():
        if term == ():
            continue
        qnc += QubitOperator(term, float(np.real(coeff)))
    return c0, qnc

def slice_ccsd_to_active(t1, t2, nocc, nmo, active_orbs):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    active_occ = [p for p in active_orbs if p < nocc]
    active_vir = [p - nocc for p in active_orbs if p >= nocc]

    nocc_act = len(active_occ)
    nvir_act = len(active_vir)

    if nocc_act == 0 or nvir_act == 0:
        t1_act = np.zeros((nocc_act, nvir_act))
        t2_act = np.zeros((nocc_act, nocc_act, nvir_act, nvir_act))
        return active_occ, active_vir, t1_act, t2_act

    t1_act = t1[np.ix_(active_occ, active_vir)]
    t2_act = t2[np.ix_(active_occ, active_occ, active_vir, active_vir)]
    return active_occ, active_vir, t1_act, t2_act


def build_theta0_and_labels_standard(t1_act, t2_act, nele_cas, norb_cas, scale=1.0):
    """
    Pack restricted CCSD amplitudes into the EXACT CUDA-Q UCCSD parameter
    order, for the closed-shell / even-electron path.

    CUDA-Q's order (verified against cudaq/kernels/uccsd.py 0.11.x-0.14.x,
    lines ~437-464 in the consumer kernel):
        [ singlesAlpha | singlesBeta | doublesMixed | doublesAlpha | doublesBeta ]

    Loop conventions (verified against the source):
        singlesAlpha:  for i in occ_alpha,   for a in vir_alpha
        singlesBeta:   for i in occ_beta,    for a in vir_beta
        doublesMixed:  for i in occ_alpha,   for j in occ_beta,
                       for r in vir_beta,    for s in vir_alpha
                       (so the inner virtual axis is BETA-then-ALPHA)
        doublesAlpha:  for i<j in occ_alpha, for a<b in vir_alpha
        doublesBeta:   for i<j in occ_beta,  for a<b in vir_beta

    RCCSD-amplitude formulas (verified empirically against E_CCSD on H2
    and stretched H4 -- recovers ~99.5% of the correlation energy AT THE
    SEED POINT before any optimizer iteration):

        Sa(i,a)        =       scale * t1[i,a]
        Sb(i,a)        =       scale * t1[i,a]
        Dmix(i,j,b,a)  = -2.0 * scale * t2[i,j,a,b]
        Daa(i,j,a,b)   = +2.0 * scale * ( t2[i,j,a,b] - t2[j,i,a,b] )
        Dbb(i,j,a,b)   = +2.0 * scale * ( t2[i,j,a,b] - t2[j,i,a,b] )

    Factor of 2 on doubles:
        cudaq.kernels.uccsd's double_excitation_opt applies the 8 Pauli
        terms of the JW-mapped double excitation generator using
        rz(0.125 * theta) per term, instead of the rz(0.25 * theta) that
        the abstract operator exponentiation calls for. Net effect: the
        kernel applies exp(theta/2 * G) when you write theta. To get the
        physical UCCSD rotation amplitude t, you must therefore pack 2*t.
        (Same kernel behavior in cudaq 0.11, 0.12, 0.14.)
    """
    t1_act = np.asarray(t1_act, dtype=float)
    t2_act = np.asarray(t2_act, dtype=float)

    nocc_act, nvir_act = t1_act.shape
    assert t2_act.shape == (nocc_act, nocc_act, nvir_act, nvir_act), \
        f"t2_act shape mismatch: {t2_act.shape} vs expected " \
        f"({nocc_act},{nocc_act},{nvir_act},{nvir_act})"

    labels = []
    theta = []

    # -------- block 1: singlesAlpha --------
    for i in range(nocc_act):
        for a in range(nvir_act):
            labels.append(("singlesAlpha", i, a))
            theta.append(scale * float(t1_act[i, a]))

    # -------- block 2: singlesBeta --------
    for i in range(nocc_act):
        for a in range(nvir_act):
            labels.append(("singlesBeta", i, a))
            theta.append(scale * float(t1_act[i, a]))

    # -------- block 3: doublesMixed --------
    # CUDA-Q loop: (alpha-occ i, beta-occ j, beta-virt b, alpha-virt a)
    # Value:  -2 * t2[i, j, a, b]
    for i in range(nocc_act):
        for j in range(nocc_act):
            for b in range(nvir_act):           # beta-virtual outer
                for a in range(nvir_act):       # alpha-virtual inner
                    labels.append(("doublesMixed", i, j, b, a))
                    theta.append(-2.0 * scale * float(t2_act[i, j, a, b]))

    # -------- block 4: doublesAlpha --------
    # i<j (occupied alpha), a<b (virtual alpha); antisymmetrized RCCSD
    for i in range(nocc_act - 1):
        for j in range(i + 1, nocc_act):
            for a in range(nvir_act - 1):
                for b in range(a + 1, nvir_act):
                    val = float(t2_act[i, j, a, b] - t2_act[j, i, a, b])
                    labels.append(("doublesAlpha", i, j, a, b))
                    theta.append(2.0 * scale * val)

    # -------- block 5: doublesBeta --------
    # Same numerical values as doublesAlpha for restricted CCSD amplitudes
    for i in range(nocc_act - 1):
        for j in range(i + 1, nocc_act):
            for a in range(nvir_act - 1):
                for b in range(a + 1, nvir_act):
                    val = float(t2_act[i, j, a, b] - t2_act[j, i, a, b])
                    labels.append(("doublesBeta", i, j, a, b))
                    theta.append(2.0 * scale * val)

    theta = np.asarray(theta, dtype=float)
    qubit_count = 2 * norb_cas
    expected = int(cudaq.kernels.uccsd_num_parameters(nele_cas, qubit_count))

    # Defensive: with correct inputs len(theta) == expected. Anything else
    # signals an active-space size or version-skew issue.
    if len(theta) > expected:
        theta0 = theta[:expected].copy()
        labels0 = labels[:expected]
    elif len(theta) < expected:
        theta0 = np.zeros(expected, dtype=float)
        theta0[:len(theta)] = theta
        labels0 = labels + [("PAD",)] * (expected - len(theta))
    else:
        theta0 = theta.copy()
        labels0 = labels

    return theta0, labels0, expected

# -----------------------------
# CUDA-Q ansatz kernels
# Closed-shell (even nele_cas): UCCSD — original unchanged
# Open-shell   (odd  nele_cas): Hardware Efficient Ansatz (HEA)
#   cudaq.kernels.uccsd hard-crashes for odd electrons.
#   HEA uses Ry rotations + CNOT entanglement with exact nele_cas electrons.
#   No padding, no dummy electrons — exact physics.
# -----------------------------

# Closed-shell kernel — original interleaved filling (even electrons)
@cudaq.kernel
def uccsd_kernel_interleaved(qubit_num: int, electron_num: int, thetas: list[float]):
    q = cudaq.qvector(qubit_num)
    filled = 0
    orb = 0
    while filled < electron_num:
        x(q[2 * orb]); filled += 1
        if filled < electron_num:
            x(q[2 * orb + 1]); filled += 1
        orb += 1
    cudaq.kernels.uccsd(q, thetas, electron_num, qubit_num)

# Open-shell HEA kernel — 3 layers of Ry + CNOT, works for any electron count
@cudaq.kernel
def hea_kernel_openshell(qubit_num: int, electron_num: int, thetas: list[float]):
    qubits = cudaq.qvector(qubit_num)
    for i in range(electron_num):
        x(qubits[i])
    for i in range(qubit_num):
        ry(thetas[i], qubits[i])
    for i in range(qubit_num - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    for i in range(qubit_num):
        ry(thetas[qubit_num + i], qubits[i])
    for i in range(qubit_num - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    for i in range(qubit_num):
        ry(thetas[2 * qubit_num + i], qubits[i])

def hea_num_parameters(qubit_count: int) -> int:
    return 3 * qubit_count

def energy_expectation(spin_ham_nc, qubit_count, nele_cas, theta, open_shell=False):
    if open_shell and (nele_cas % 2 != 0):
        r = cudaq.observe(hea_kernel_openshell, spin_ham_nc, qubit_count, nele_cas, theta)
    else:
        r = cudaq.observe(uccsd_kernel_interleaved, spin_ham_nc, qubit_count, nele_cas, theta)
    return float(r.expectation())

# -----------------------------
# VQE one chunk
# -----------------------------
def optimize_vqe_one_chunk(spin_ham_nc, qubit_count, nele_cas, x0,
                           method="COBYLA", tol=1e-10, maxiter=600, rhobeg=0.2,
                           open_shell=False):
    quantum_times = []
    energy_convergence = []

    def cost(theta):
        t0 = timeit.default_timer()
        if open_shell and (nele_cas % 2 != 0):
            r = cudaq.observe(hea_kernel_openshell, spin_ham_nc, qubit_count, nele_cas, theta)
        else:
            r = cudaq.observe(uccsd_kernel_interleaved, spin_ham_nc, qubit_count, nele_cas, theta)
        quantum_times.append(timeit.default_timer() - t0)
        e = float(r.expectation())
        energy_convergence.append(e)
        return e

    t_start = timeit.default_timer()
    if method.upper() == "COBYLA":
        res = minimize(cost, x0, method="COBYLA",
                       options={"maxiter": int(maxiter), "tol": float(tol), "rhobeg": float(rhobeg)})
    else:
        res = minimize(cost, x0, method=method, options={"maxiter": int(maxiter), "tol": float(tol)})
    t_end = timeit.default_timer()

    runtime_total = float(t_end - t_start)
    runtime_quantum_sum = float(np.sum(quantum_times))
    runtime_optimizer = float(runtime_total - runtime_quantum_sum)

    return {
        "E_nc_opt": float(res.fun),
        "theta_opt": np.array(res.x, dtype=float),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "runtime_total": runtime_total,
        "runtime_quantum_sum": runtime_quantum_sum,
        "runtime_optimizer": runtime_optimizer,
        "quantum_times": quantum_times,
        "energy_convergence": energy_convergence,
    }

# -----------------------------
# VQE until converged
# -----------------------------
def vqe_until_converged(
    spin_ham_nc, qubit_count, nele_cas, theta_start, rng,
    eps_E=1e-6, patience=3, max_cycles=25,
    chunk_maxiter=600,
    jitter_between_cycles=True, jitter_scale=5e-4,
    method="COBYLA", tol=1e-10, rhobeg=0.2,
    verbose_cycles=False, open_shell=False
):
    best_theta = np.array(theta_start, dtype=float)
    best_E = energy_expectation(spin_ham_nc, qubit_count, nele_cas, best_theta,
                                open_shell=open_shell)

    all_quantum_times = []
    all_energy_convergence = []
    best_energy_per_cycle = []
    cycle_summaries = []

    total_quantum = 0.0
    total_time = 0.0
    total_optimizer = 0.0
    total_nit = 0
    total_nfev = 0
    last_message = "init"
    any_success = False

    no_improve = 0
    converged = False

    for cyc in range(1, max_cycles + 1):
        x0 = best_theta.copy()
        if jitter_between_cycles and cyc > 1:
            x0 = x0 + rng.normal(0.0, jitter_scale, size=len(x0))

        out = optimize_vqe_one_chunk(
            spin_ham_nc, qubit_count, nele_cas, x0,
            method=method, tol=tol, maxiter=chunk_maxiter, rhobeg=rhobeg,
            open_shell=open_shell
        )

        all_quantum_times.extend(out["quantum_times"])
        all_energy_convergence.extend(out["energy_convergence"])

        total_quantum += out["runtime_quantum_sum"]
        total_time += out["runtime_total"]
        total_optimizer += out["runtime_optimizer"]
        total_nit += max(0, out["nit"])
        total_nfev += max(0, out["nfev"])
        last_message = out["message"]
        any_success = any_success or out["success"]

        E_new = float(out["E_nc_opt"])
        theta_new = np.array(out["theta_opt"], dtype=float)

        dE = best_E - E_new
        if dE > 1e-15:
            best_E = E_new
            best_theta = theta_new

        best_energy_per_cycle.append(best_E)
        cycle_summaries.append({
            "cycle": int(cyc),
            "E_nc_opt_cycle": float(E_new),
            "best_E_nc_after_cycle": float(best_E),
            "dE_vs_prev_best": float(dE),
            "success": bool(out["success"]),
            "nit": int(out["nit"]),
            "nfev": int(out["nfev"]),
            "runtime_total": float(out["runtime_total"]),
            "runtime_quantum_sum": float(out["runtime_quantum_sum"]),
            "runtime_optimizer": float(out["runtime_optimizer"]),
        })

        if verbose_cycles:
            print(f"[VQE] cycle={cyc:02d} best_E_nc={best_E:+.12f} dE={dE:.3e} success={out['success']}", flush=True)

        if dE < eps_E:
            no_improve += 1
        else:
            no_improve = 0

        if no_improve >= patience:
            converged = True
            break

    return {
        "E_nc_opt": float(best_E),
        "theta_opt": np.array(best_theta, dtype=float),
        "success": bool(any_success),
        "message": f"vqe_until_converged: {last_message}",
        "nit": int(total_nit),
        "nfev": int(total_nfev),
        "runtime_total": float(total_time),
        "runtime_quantum_sum": float(total_quantum),
        "runtime_optimizer": float(total_optimizer),
        "cycles": int(cyc),
        "converged": bool(converged),
        "quantum_times": all_quantum_times,
        "energy_convergence": all_energy_convergence,
        "best_energy_per_cycle": best_energy_per_cycle,
        "cycle_summaries": cycle_summaries,
    }

# -----------------------------
# Best-of-jitters seed finder
# -----------------------------
def best_of_jitters_one_chunk(spin_ham_nc, qubit_count, nele_cas, theta0, rng,
                             n_restarts=3, jitter_scale=5e-3,
                             chunk_maxiter=600, method="COBYLA", tol=1e-10, rhobeg=0.2,
                             open_shell=False):
    candidates = [theta0]
    for _ in range(int(n_restarts)):
        candidates.append(theta0 + rng.normal(0.0, jitter_scale, size=len(theta0)))

    best = None
    best_out = None
    best_idx = None

    for idx, x0 in enumerate(candidates):
        out = optimize_vqe_one_chunk(
            spin_ham_nc, qubit_count, nele_cas, x0,
            method=method, tol=tol, maxiter=chunk_maxiter, rhobeg=rhobeg,
            open_shell=open_shell
        )
        if (best is None) or (out["E_nc_opt"] < best):
            best = out["E_nc_opt"]
            best_out = out
            best_idx = idx

    best_out = dict(best_out)
    best_out["best_init_index"] = int(best_idx)
    return best_out

# -----------------------------
# Minimal PySCF insights
# -----------------------------
def collect_pyscf_insights(mf, molecule_of):
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

# -----------------------------
# Run one molecule
# -----------------------------
def run_one_molecule(mol_name: str, spec: dict, integrals_dir: str = None):
    mol_name_clean = sanitize_name(mol_name)

    geometry = spec["geometry"]
    charge = int(spec["charge"])
    multiplicity = int(spec["multiplicity"])

    active_spaces = spec["valid_active_spaces"]

    files = None
    if integrals_dir is not None:
        # [2g] integral-file mode: no SCF/CCSD here, everything comes from the files
        if multiplicity != 1:
            raise ValueError(f"{mol_name}: integral files are closed-shell only")
        files = IntegralFiles(integrals_dir, mol_name, spec, BASIS, max_memory=MAX_MEMORY,
                              all_spaces=molecules.get(mol_name, spec)["valid_active_spaces"])
        first = None
        for s in active_spaces:
            try:
                first = files.get(int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"]))
                break
            except SpaceDoesNotFit:
                continue
        if first is None:
            raise ValueError(f"no active space of {mol_name} fits the {BASIS} basis")

        molecule = mf = None
        t1amp = t2amp = None
        nmo = int(first["nmo"])
        nelec = int(first["n_electrons"])
        nocc = nelec // 2
        nvir = nmo - nocc
        HF_FULL = float(first["e_hf"])
        is_open_shell = False
        E_CCSD_FULL = float(first["e_ccsd"])
        ccsd_block = {"computed": True, "E_ccsd_total": E_CCSD_FULL,
                      "E_ccsd_corr": float(E_CCSD_FULL - HF_FULL),
                      "t1_norm": None, "t2_norm": None,
                      "note": "read from integral files; amplitudes stored per active space."}
        pyscf_info = {"source": f"integral files: {integrals_dir}"}
        t0 = t1 = 0.0                    # the timing entry is set from files.seconds below
    else:
        moldata = openfermion.MolecularData(geometry, BASIS, multiplicity, charge)
        t0 = time.time()
        molecule = openfermionpyscf.run_pyscf(moldata, run_scf=True, run_fci=False)
        t1 = time.time()

        mf = molecule._pyscf_data["scf"]
        pyscf_info = collect_pyscf_insights(mf, molecule)

        # ── CHANGE 1: open-shell skip block REMOVED ───────────────────────────
        # Previously returned skipped dict here if mf.mol.spin != 0
        # Now open-shell molecules continue and run normally

        nmo = mf.mo_coeff.shape[1]
        nelec = int(mf.mol.nelectron)
        nocc = mf.mol.nelectron // 2
        nvir = nmo - nocc

        HF_FULL = float(molecule.hf_energy)
        is_open_shell = int(mf.mol.spin) != 0

        ccsd_block = {"computed": False, "E_ccsd_total": None, "E_ccsd_corr": None,
                      "t1_norm": None, "t2_norm": None, "note": None}
        t1amp = None
        t2amp = None
        E_CCSD_FULL = None

        if RUN_CCSD_REFERENCE:
            try:
                if not is_open_shell:
                    # Closed-shell: original unchanged
                    mycc = cc.CCSD(mf)
                else:
                    # Open-shell: UCCSD instead
                    mycc = cc.UCCSD(mf)
                ecc_corr, t1amp, t2amp = mycc.kernel()
                E_CCSD_FULL = float(mf.e_tot + ecc_corr)
                ccsd_block.update({
                    "computed": True,
                    "E_ccsd_total": float(E_CCSD_FULL),
                    "E_ccsd_corr": float(ecc_corr),
                    "t1_norm": float(np.linalg.norm(np.asarray(t1amp))),
                    "t2_norm": float(np.linalg.norm(np.asarray(t2amp))),
                    "note": "CCSD full-system amplitudes used for theta0 slicing.",
                })
            except Exception as e:
                ccsd_block.update({"computed": False, "note": f"CCSD failed: {repr(e)}"})

    # cudaq.set_target(TARGET)
    cudaq_precision = configure_cudaq_target()
    molecule_results = {
        "molecule_name": mol_name,
        "molecule_name_clean": mol_name_clean,
        "skipped": False,
        "tag": TAG,
        "basis": BASIS,
        "target": TARGET,
        "target_precision_option": TARGET_PRECISION,
        "cudaq_precision": cudaq_precision,
        "run_metadata": run_metadata(),
        "optimizer": OPTIMIZER,
        "seed": int(SEED),
        "input_spec": spec,
        "timing": {"pyscf_run_scf_seconds": float(t1 - t0)},
        "references": {
            "E_hf_full": float(HF_FULL),
            "E_ccsd_full": float(E_CCSD_FULL) if E_CCSD_FULL is not None else None,
        },
        "pyscf_insights": pyscf_info,
        "ccsd": ccsd_block,
        "system_sizes": {"nmo": int(nmo), "nocc": int(nocc), "nvir": int(nvir)},
        "active_space_runs": [],
    }
    if files is not None:
        molecule_results["integrals"] = {"dir": os.path.abspath(integrals_dir), "files_made": []}

    # ── Active-space loop ─────────────────────────────────────────────────
    for space in active_spaces:
        ncore    = int(space["ncore"])
        nele_cas = int(space["nele_cas"])
        norb_cas = int(space["norb_cas"])
        qubit_count = 2 * norb_cas

        # ── FIX 1: deterministic local_rng per (molecule, active_space) ──
        # _stable_hash is hashlib.sha256-based, so the seed is identical
        # across Python sessions, machines, CPU vs GPU. Previously this used
        # Python's built-in hash() which is randomized by PYTHONHASHSEED.
        # rng_global is intentionally NOT used here anymore.
        local_rng = np.random.default_rng(
            SEED + _stable_hash((mol_name, ncore, nele_cas, norb_cas))
        )
        # ──────────────────────────────────────────────────────────────────

        occ = list(range(ncore))
        act = list(range(ncore, ncore + norb_cas))

        if len(act) == 0 or act[-1] >= nmo:
            molecule_results["active_space_runs"].append({
                "space": space, "skipped": True,
                "skip_reason": f"active indices exceed nmo={nmo}",
            })
            continue

        if (2 * ncore + nele_cas) != nelec:
            molecule_results["active_space_runs"].append({
                "space": space, "skipped": True,
                "skip_reason": (
                    f"CASCI sanity fail: 2*ncore+nele_cas="
                    f"{2*ncore+nele_cas} != nelec={nelec}"
                ),
            })
            continue

        data = None
        if files is not None:
            # [2g] CASCI energy, Hamiltonian and CCSD amplitudes from the integral file
            data = files.get(ncore, nele_cas, norb_cas)
            E_CASCI = float(data["e_casci"])
            qop = qubit_hamiltonian(data)
            molecule_results["timing"]["pyscf_run_scf_seconds"] = float(files.seconds)
            molecule_results["integrals"]["files_made"] = list(files.made)
        else:
            casci = mcscf.CASCI(mf, norb_cas, nele_cas)
            casci.ncore = ncore
            casci_out = casci.kernel()
            E_CASCI = float(casci_out[0])

            molecular_ham = molecule.get_molecular_hamiltonian(
                occupied_indices=occ, active_indices=act)
            fermion_ham = get_fermion_operator(molecular_ham)
            qop = jordan_wigner(fermion_ham)
        qubit_ham = make_qubitop_real(qop, tol=JW_IMAG_TOL)
        c0, qubit_ham_nc = split_constant(qubit_ham)
        spin_nc = cudaq.SpinOperator(qubit_ham_nc)

        # Open-shell odd: use HEA parameter count (3 * qubit_count)
        # Open-shell even or closed-shell: use uccsd_num_parameters
        if is_open_shell and (nele_cas % 2 != 0):
            expected = hea_num_parameters(qubit_count)
        else:
            expected = int(cudaq.kernels.uccsd_num_parameters(nele_cas, qubit_count))

        if not is_open_shell:
            # Closed-shell: CCSD-sliced theta0 using the CORRECTED packer
            # (CUDA-Q block order + factor of 2 on doubles).
            labels0 = None
            if data is not None:
                # [2g] the stored active-space amplitudes, same corrected packer
                theta0, labels0, expected_check = build_theta0_and_labels_standard(
                    data["t1_active"], data["t2_active"], nele_cas=nele_cas,
                    norb_cas=norb_cas, scale=THETA_SCALE)
                if expected_check != expected or any(l == ("PAD",) for l in labels0):
                    raise RuntimeError(f"{mol_name} ({nele_cas},{norb_cas}): the packer had to "
                                       f"pad or cut theta0 to {expected} parameters")
                theta0_source = "CCSD-sliced (corrected packer, x2 doubles, from integral file)"
            elif (t1amp is not None) and (t2amp is not None):
                _, _, t1_act, t2_act = slice_ccsd_to_active(
                    t1amp, t2amp, nocc=nocc, nmo=nmo, active_orbs=act)
                theta0, labels0, expected_check = build_theta0_and_labels_standard(
                    t1_act, t2_act, nele_cas=nele_cas, norb_cas=norb_cas,
                    scale=THETA_SCALE)
                if expected_check != expected:
                    theta0 = np.zeros(expected, dtype=float)
                    theta0_source = "zeros (ccsd-pack-mismatch)"
                else:
                    theta0_source = "CCSD-sliced (corrected packer, x2 doubles)"
            else:
                theta0 = np.zeros(expected, dtype=float)
                theta0_source = "zeros (CCSD unavailable/off)"
        else:
            # ── FIX 2: open-shell theta0 — seeded via local_rng ──────────
            # Old code:  np.random.uniform(-0.1, 0.1, expected)
            #   → unseeded, CPU and GPU draw different arrays → different
            #     local minima, meaningless speedup comparison.
            # New code:  local_rng.uniform(-0.1, 0.1, expected)
            #   → deterministic from (SEED, mol_name, ncore, nele_cas, norb_cas)
            #   → CPU and GPU start from IDENTICAL theta0
            #   → range kept at original (-0.1, 0.1) — only seed is fixed
            theta0 = local_rng.uniform(-0.1, 0.1, expected)
            theta0_source = f"seeded_uniform seed={SEED}"
            # ──────────────────────────────────────────────────────────────
       
        # [2e] guard: closed-shell runs must use the corrected CCSD packer at full scale
        if not is_open_shell:
            if not theta0_source.startswith("CCSD-sliced (corrected packer"):
                raise RuntimeError(f"{mol_name} ({nele_cas},{norb_cas}): theta0 source is '{theta0_source}'")
            if THETA_SCALE != 1.0:
                raise RuntimeError(f"THETA_SCALE must be 1.0, got {THETA_SCALE}")

        # [2c] energy of the circuit at theta = 0 (reference) and at theta0 (start), before any optimization
        E_ref_start_nc = energy_expectation(spin_nc, qubit_count, nele_cas,
                                            np.zeros(expected), open_shell=is_open_shell)
        E_theta0_nc    = energy_expectation(spin_nc, qubit_count, nele_cas,
                                            theta0, open_shell=is_open_shell)

        is_heavy = expected > HEAVY_PARAM_THRESHOLD


        local_restarts = N_JITTER_RESTARTS if not is_heavy else HEAVY_RESTARTS
        local_rhobeg   = COBYLA_RHOBEG     if not is_heavy else HEAVY_RHOBEG

        if local_restarts > 0:
            seed_out = best_of_jitters_one_chunk(
                spin_nc, qubit_count, nele_cas,
                theta0=theta0,
                rng=local_rng,          # ── FIX 3a: was rng_global ──────
                n_restarts=local_restarts,
                jitter_scale=JITTER_SCALE,
                chunk_maxiter=VQE_CHUNK_MAXITER,
                method=OPTIMIZER, tol=TOL, rhobeg=local_rhobeg,
                open_shell=is_open_shell
            )
            theta_seed       = seed_out["theta_opt"]
            best_init_index  = int(seed_out["best_init_index"])
        else:
            seed_out        = None
            theta_seed      = theta0.copy()
            best_init_index = -1

        vqe_out = vqe_until_converged(
            spin_nc, qubit_count, nele_cas,
            theta_start=theta_seed,
            rng=local_rng,              # ── FIX 3b: was rng_global ──────
            eps_E=VQE_EPS_E,
            patience=VQE_PATIENCE,
            max_cycles=VQE_MAX_CYCLES,
            chunk_maxiter=VQE_CHUNK_MAXITER,
            jitter_between_cycles=VQE_JITTER_BETWEEN_CYCLES,
            jitter_scale=VQE_JITTER_BETWEEN_SCALE,
            method=OPTIMIZER, tol=TOL, rhobeg=local_rhobeg,
            verbose_cycles=PRINT_EVERY_CYCLE,
            open_shell=is_open_shell
        )
        vqe_out["best_init_index"] = best_init_index

        E_VQE = float(c0 + vqe_out["E_nc_opt"])

        d_vqe_casci    = float(E_VQE - E_CASCI)
        d_vqe_hf_full  = float(E_VQE - HF_FULL)
        d_vqe_ccsd_full = (float(E_VQE - E_CCSD_FULL)
                           if E_CCSD_FULL is not None else None)

        molecule_results["active_space_runs"].append({
            "space": space,
            "skipped": False,
            "sizes": {
                "qubits": int(qubit_count),
                "uccsd_num_parameters": int(expected),
            },
            "active_indices": {
                "occupied_indices": occ,
                "active_indices": act,
            },
            "casci": {"E_casci_total": float(E_CASCI)},
            "hamiltonian": {
                "c0": float(c0),
                "num_qubit_terms_nonconstant": int(len(qubit_ham_nc.terms)),
            },
         "theta0": {
                "source": theta0_source,
                "theta_scale": float(THETA_SCALE),
                "theta0_norm": float(np.linalg.norm(theta0)),
                "E_ref_total": float(c0 + E_ref_start_nc),     # circuit at theta = 0 (should equal HF)
                "E_theta0_total": float(c0 + E_theta0_nc),     # circuit at theta0, before optimization
            },

            "seed_search": seed_out,
            "vqe": {
                "E_nc_opt": float(vqe_out["E_nc_opt"]),
                "E_total": float(E_VQE),
                "theta_opt": np.array(vqe_out["theta_opt"], dtype=float),
                "converged": bool(vqe_out["converged"]),
                "cycles": int(vqe_out["cycles"]),
                "runtime": float(vqe_out["runtime_total"]),
                "simulated_quantum_runtime": float(vqe_out["runtime_quantum_sum"]),
                "optimizer_runtime": float(vqe_out["runtime_optimizer"]),
                "quantum_times": list(vqe_out["quantum_times"]),
                "energy_convergence": list(vqe_out["energy_convergence"]),
                "best_energy_per_cycle": list(vqe_out["best_energy_per_cycle"]),
                "cycle_summaries": list(vqe_out["cycle_summaries"]),
                "success_any": bool(vqe_out["success"]),
                "message": str(vqe_out["message"]),
                "best_init_index": int(vqe_out["best_init_index"]),
                "nit_total": int(vqe_out["nit"]),
                "nfev_total": int(vqe_out["nfev"]),
            },
            "compare": {
                "d_vqe_minus_casci": d_vqe_casci,
                "d_vqe_minus_hf_full": d_vqe_hf_full,
                "d_vqe_minus_ccsd_full": d_vqe_ccsd_full,
            },
        })

    return molecule_results

# -----------------------------
# MAIN: loop all molecules, save PKL per molecule
# -----------------------------
def run_all_molecules(molecules: dict, out_dir: str = "pkl_results"):
    os.makedirs(out_dir, exist_ok=True)

    configure_cudaq_target()

    for mol_name, spec in molecules.items():
        mol_clean = sanitize_name(mol_name)
        file_name = (
            f"{TAG}_{mol_clean}_{sanitize_name(BASIS)}_"
            f"{sanitize_name(TARGET)}_{sanitize_name(OPTIMIZER)}_VQE_results.pkl"
        )
        out_path = os.path.join(out_dir, file_name)

        print(f"[RUN] {mol_name} -> {out_path}", flush=True)

        try:
            mol_res = run_one_molecule(mol_name, spec)
        except Exception as e:
            print(f"[ERROR] {mol_name}: {repr(e)}", flush=True)
            continue

        payload = {mol_name: mol_res}
        save_pkl(payload, out_path)

        print(f"[DONE] {mol_name}", flush=True)

    print(f"[ALL DONE] PKL results saved under: {out_dir}", flush=True)


import argparse

def main():
    global BASIS, TARGET, OPTIMIZER, TARGET_PRECISION, MAX_MEMORY

    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule",  required=True)
    parser.add_argument("--basis",     default=BASIS)
    parser.add_argument("--target",    default=TARGET)
    parser.add_argument(
        "--precision",
        default="fp64",
        choices=["default", "fp32", "fp64"],
        help="CUDA-Q precision option for nvidia target: default, fp32, or fp64"
    )
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--out_dir",   default="pkl_results")
    parser.add_argument("--space_idx", type=int, default=None)
    parser.add_argument("--integrals", default=None,
                        help="[2g] read Hamiltonian, E_HF, E_CASCI and CCSD amplitudes from this "
                             "integrals folder instead of running PySCF; missing files are made there")
    parser.add_argument("--max-memory", type=int, default=MAX_MEMORY,
                        help="PySCF memory limit in MB when a missing integral file is made")
    args = parser.parse_args()

    BASIS     = args.basis
    TARGET    = args.target
    OPTIMIZER = args.optimizer
    TARGET_PRECISION = None if args.precision == "default" else args.precision
    MAX_MEMORY = args.max_memory
    # TARGET_PRECISION = args.precision   # None = default, "fp32" or "fp64" for nvidia

    molecule_name = args.molecule
    out_dir       = args.out_dir
    space_idx     = args.space_idx

    if molecule_name not in molecules:
        raise ValueError(f"Molecule '{molecule_name}' not found!")

    spec = dict(molecules[molecule_name])

    if space_idx is not None:
        spaces = spec.get("valid_active_spaces", [])
        if space_idx < 0 or space_idx >= len(spaces):
            raise ValueError(f"--space_idx {space_idx} out of range")
        spec["valid_active_spaces"] = [spaces[space_idx]]

    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[RUN] {molecule_name} | BASIS={BASIS} | TARGET={TARGET} "
        f"| PRECISION={TARGET_PRECISION or 'default'} | OPT={OPTIMIZER}"
        f"{' | INTEGRALS=' + args.integrals if args.integrals else ''}",
        flush=True,
    )

    mol_res = run_one_molecule(molecule_name, spec, integrals_dir=args.integrals)

    tag       = time.strftime("%d_%b_%Y").upper()
    mol_clean = sanitize_name(molecule_name)
    file_name = (
        f"{tag}_{mol_clean}_{sanitize_name(BASIS)}_"
        f"{sanitize_name(TARGET)}_{sanitize_name(OPTIMIZER)}_VQE_results.pkl"
    )
    out_path = os.path.join(out_dir, file_name)

    with open(out_path, "wb") as f:
        pickle.dump({molecule_name: mol_res}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()