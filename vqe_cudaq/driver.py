"""Per-molecule and batch drivers that orchestrate the full VQE benchmark.

These are the only functions that depend on the run configuration. They snapshot
the relevant :mod:`config` values into locals at entry, so a CLI override taken
just before the call is honoured, and the body reads a stable configuration.
"""

import os
import time
import numpy as np

import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from pyscf import cc, mcscf
import cudaq

from . import config
from .utils import sanitize_name, save_pkl, stable_hash
from .backend import configure_cudaq_target
from .insights import collect_pyscf_insights
from .operators import (
    make_qubitop_real,
    split_constant,
    slice_ccsd_to_active,
    build_theta0_and_labels_standard,
)
from .ansatz import hea_num_parameters
from .vqe import best_of_jitters_one_chunk, vqe_until_converged


def run_one_molecule(mol_name: str, spec: dict):
    """Run every active space of one molecule and return the rich result dict."""
    # ── snapshot run configuration ────────────────────────────────────
    BASIS = config.BASIS
    TAG = config.TAG
    TARGET = config.TARGET
    TARGET_PRECISION = config.TARGET_PRECISION
    OPTIMIZER = config.OPTIMIZER
    SEED = config.SEED
    RUN_CCSD_REFERENCE = config.RUN_CCSD_REFERENCE
    JW_IMAG_TOL = config.JW_IMAG_TOL
    THETA_SCALE = config.THETA_SCALE
    HEAVY_PARAM_THRESHOLD = config.HEAVY_PARAM_THRESHOLD
    N_JITTER_RESTARTS = config.N_JITTER_RESTARTS
    HEAVY_RESTARTS = config.HEAVY_RESTARTS
    COBYLA_RHOBEG = config.COBYLA_RHOBEG
    HEAVY_RHOBEG = config.HEAVY_RHOBEG
    JITTER_SCALE = config.JITTER_SCALE
    VQE_CHUNK_MAXITER = config.VQE_CHUNK_MAXITER
    TOL = config.TOL
    VQE_EPS_E = config.VQE_EPS_E
    VQE_PATIENCE = config.VQE_PATIENCE
    VQE_MAX_CYCLES = config.VQE_MAX_CYCLES
    VQE_JITTER_BETWEEN_CYCLES = config.VQE_JITTER_BETWEEN_CYCLES
    VQE_JITTER_BETWEEN_SCALE = config.VQE_JITTER_BETWEEN_SCALE
    PRINT_EVERY_CYCLE = config.PRINT_EVERY_CYCLE
    # ──────────────────────────────────────────────────────────────────

    mol_name_clean = sanitize_name(mol_name)

    geometry = spec["geometry"]
    charge = int(spec["charge"])
    multiplicity = int(spec["multiplicity"])

    active_spaces = spec["valid_active_spaces"]

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

    # ── Active-space loop ─────────────────────────────────────────────────
    for space in active_spaces:
        ncore    = int(space["ncore"])
        nele_cas = int(space["nele_cas"])
        norb_cas = int(space["norb_cas"])
        qubit_count = 2 * norb_cas

        # ── FIX 1: deterministic local_rng per (molecule, active_space) ──
        # stable_hash is hashlib.sha256-based, so the seed is identical
        # across Python sessions, machines, CPU vs GPU. Previously this used
        # Python's built-in hash() which is randomized by PYTHONHASHSEED.
        # rng_global is intentionally NOT used here anymore.
        local_rng = np.random.default_rng(
            SEED + stable_hash((mol_name, ncore, nele_cas, norb_cas))
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

        if (2 * ncore + nele_cas) != int(mf.mol.nelectron):
            molecule_results["active_space_runs"].append({
                "space": space, "skipped": True,
                "skip_reason": (
                    f"CASCI sanity fail: 2*ncore+nele_cas="
                    f"{2*ncore+nele_cas} != nelec={mf.mol.nelectron}"
                ),
            })
            continue

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
            if (t1amp is not None) and (t2amp is not None):
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


def run_all_molecules(molecules: dict, out_dir: str = "pkl_results"):
    """Run every molecule in ``molecules`` and write one PKL per molecule."""
    TAG = config.TAG
    BASIS = config.BASIS
    TARGET = config.TARGET
    OPTIMIZER = config.OPTIMIZER

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
