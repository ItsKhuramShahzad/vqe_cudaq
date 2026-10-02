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
from pyscf import mcscf
import cudaq

from . import config
from .utils import sanitize_name, save_pkl, stable_hash, run_metadata
from .backend import configure_cudaq_target
from .insights import collect_pyscf_insights
from .operators import (
    make_qubitop_real,
    split_constant,
    slice_ccsd_to_active,
    build_theta0_and_labels_standard,
)
from .ansatz import energy_expectation, final_state_diagnostics
from .hamiltonian import IntegralFiles, SpaceDoesNotFit, qubit_hamiltonian, run_ccsd
from .vqe import best_of_jitters_one_chunk, vqe_until_converged
from .xyz import geometry_in_angstrom


def run_one_molecule(mol_name: str, spec: dict, integrals_dir: str = None):
    """Run every active space of one molecule and return the rich result dict.

    integrals_dir: read the active-space Hamiltonian, E_HF, E_CCSD, E_CASCI and the
    CCSD amplitudes from saved integral files under this folder instead of running
    PySCF. Missing files are computed and saved there. Without it, the geometry
    route (SCF + CCSD + CASCI here) is used.

    Closed-shell molecules only. Every active space starts from the CCSD amplitudes
    (corrected packer); if they cannot be obtained or packed, the run stops with an
    error, it never starts from zeros or random parameters.
    """
    # ── snapshot run configuration ────────────────────────────────────
    BASIS = config.BASIS
    TAG = config.TAG
    TARGET = config.TARGET
    TARGET_PRECISION = config.TARGET_PRECISION
    OPTIMIZER = config.OPTIMIZER
    SEED = config.SEED
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
    MAX_MEMORY = config.MAX_MEMORY
    # ──────────────────────────────────────────────────────────────────

    mol_name_clean = sanitize_name(mol_name)

    charge = int(spec["charge"])
    multiplicity = int(spec["multiplicity"])

    active_spaces = spec["valid_active_spaces"]

    # Closed shell only: the UCCSD start needs restricted CCSD amplitudes.
    if multiplicity != 1:
        raise ValueError(f"{mol_name}: multiplicity {multiplicity}; only closed-shell "
                         f"molecules are supported")

    files = None
    molecule = mf = None
    if integrals_dir is not None:
        # Integral-file mode: no SCF/CCSD here, everything comes from the files.
        from .molecules import molecules as MOLECULES
        files = IntegralFiles(integrals_dir, mol_name, spec, BASIS, max_memory=MAX_MEMORY,
                              all_spaces=MOLECULES.get(mol_name, spec)["valid_active_spaces"])
        first = None
        for s in active_spaces:
            try:
                first = files.get(int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"]))
                break
            except SpaceDoesNotFit:
                continue
        if first is None:
            raise ValueError(f"no active space of {mol_name} fits the {BASIS} basis")

        t1amp = t2amp = None
        nmo = int(first["nmo"])
        nelec = int(first["n_electrons"])
        nocc = nelec // 2
        nvir = nmo - nocc
        HF_FULL = float(first["e_hf"])
        E_CCSD_FULL = float(first["e_ccsd"])
        ccsd_block = {"computed": True, "E_ccsd_total": E_CCSD_FULL,
                      "E_ccsd_corr": float(E_CCSD_FULL - HF_FULL),
                      "t1_norm": None, "t2_norm": None,
                      "note": "read from integral files; amplitudes stored per active space."}
        pyscf_info = {"source": f"integral files: {integrals_dir}"}
        t0 = t1 = 0.0                    # the timing entry is set from files.seconds below
    else:
        # OpenFermion MolecularData interprets coordinates as angstrom.  Normalize
        # explicitly because a few database geometries are recorded in bohr.
        geometry = geometry_in_angstrom(spec)

        moldata = openfermion.MolecularData(geometry, BASIS, multiplicity, charge)
        t0 = time.time()
        molecule = openfermionpyscf.run_pyscf(moldata, run_scf=True, run_fci=False)
        t1 = time.time()

        mf = molecule._pyscf_data["scf"]
        if not mf.converged:
            raise RuntimeError(f"{mol_name}: SCF did not converge")
        pyscf_info = collect_pyscf_insights(mf, molecule)

        nmo = mf.mo_coeff.shape[1]
        nelec = int(mf.mol.nelectron)
        nocc = mf.mol.nelectron // 2
        nvir = nmo - nocc

        HF_FULL = float(molecule.hf_energy)

        # Full-system CCSD: its amplitudes are the VQE starting point, so it must succeed
        # and converge (run_ccsd raises otherwise).
        E_CCSD_FULL, t1amp, t2amp = run_ccsd(mf, max_memory=MAX_MEMORY)
        ccsd_block = {
            "computed": True,
            "E_ccsd_total": float(E_CCSD_FULL),
            "E_ccsd_corr": float(E_CCSD_FULL - HF_FULL),
            "t1_norm": float(np.linalg.norm(np.asarray(t1amp))),
            "t2_norm": float(np.linalg.norm(np.asarray(t2amp))),
            "note": "CCSD full-system amplitudes used for theta0 slicing.",
        }

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
            # CASCI energy, Hamiltonian and CCSD amplitudes from the integral file
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

        expected = int(cudaq.kernels.uccsd_num_parameters(nele_cas, qubit_count))

        # theta0 from the CCSD amplitudes only, with the corrected packer (CUDA-Q block
        # order, x2 on doubles). The packer raises if the amplitudes do not give exactly
        # `expected` parameters; there is no zero or random start.
        if data is not None:
            t1_act, t2_act = data["t1_active"], data["t2_active"]      # stored in the file
            theta0_source = "CCSD-sliced (corrected packer, x2 doubles, from integral file)"
        else:
            _, _, t1_act, t2_act = slice_ccsd_to_active(
                t1amp, t2amp, nocc=nocc, nmo=nmo, active_orbs=act)
            theta0_source = "CCSD-sliced (corrected packer, x2 doubles)"
        theta0, labels0, expected_check = build_theta0_and_labels_standard(
            t1_act, t2_act, nele_cas=nele_cas, norb_cas=norb_cas, scale=THETA_SCALE)

        # Last check: corrected CCSD packer at full scale, right parameter count
        if expected_check != expected or len(theta0) != expected:
            raise RuntimeError(f"{mol_name} ({nele_cas},{norb_cas}): theta0 has "
                               f"{len(theta0)} parameters, the kernel expects {expected}")
        if not theta0_source.startswith("CCSD-sliced (corrected packer"):
            raise RuntimeError(f"{mol_name} ({nele_cas},{norb_cas}): theta0 source is "
                               f"'{theta0_source}'")
        if THETA_SCALE != 1.0:
            raise RuntimeError(f"THETA_SCALE must be 1.0, got {THETA_SCALE}")

        # Energy of the circuit at theta = 0 (reference) and at theta0 (start),
        # before any optimisation or seed search
        E_ref_start_nc = energy_expectation(spin_nc, qubit_count, nele_cas,
                                            np.zeros(expected))
        E_theta0_nc = energy_expectation(spin_nc, qubit_count, nele_cas, theta0)

        is_heavy = expected > HEAVY_PARAM_THRESHOLD
        local_restarts = N_JITTER_RESTARTS if not is_heavy else HEAVY_RESTARTS
        local_rhobeg   = COBYLA_RHOBEG     if not is_heavy else HEAVY_RHOBEG

        t_seed0 = time.perf_counter()   # seed search timed on its own (all candidates)
        if local_restarts > 0:
            seed_out = best_of_jitters_one_chunk(
                spin_nc, qubit_count, nele_cas,
                theta0=theta0,
                rng=local_rng,          # ── FIX 3a: was rng_global ──────
                n_restarts=local_restarts,
                jitter_scale=JITTER_SCALE,
                chunk_maxiter=VQE_CHUNK_MAXITER,
                method=OPTIMIZER, tol=TOL, rhobeg=local_rhobeg,
            )
            theta_seed       = seed_out["theta_opt"]
            best_init_index  = int(seed_out["best_init_index"])
        else:
            seed_out        = None
            theta_seed      = theta0.copy()
            best_init_index = -1
        seed_search_runtime = time.perf_counter() - t_seed0     # 0 when the seed search is skipped

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
        )
        vqe_out["best_init_index"] = best_init_index

        E_VQE = float(c0 + vqe_out["E_nc_opt"])

        # Final state, <S^2> and fidelity with the exact ground state, after the timed
        # VQE and with the same CUDA-Q version. A failure here is recorded, not raised,
        # so it can never cost the VQE result.
        t_fs = time.perf_counter()
        try:
            final_state = final_state_diagnostics(qubit_ham, vqe_out["theta_opt"],
                                                  qubit_count, nele_cas, E_VQE)
        except Exception as e:
            final_state = {"error": repr(e)}
        final_state["seconds"] = float(time.perf_counter() - t_fs)

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
                "E_ref_total": float(c0 + E_ref_start_nc),   # circuit at theta = 0 (should equal HF)
                "E_theta0_total": float(c0 + E_theta0_nc),   # circuit at theta0, before optimisation
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
                "seed_search_runtime": float(seed_search_runtime),
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
            "final_state": final_state,
            "compare": {
                "d_vqe_minus_casci": d_vqe_casci,
                "d_vqe_minus_hf_full": d_vqe_hf_full,
                "d_vqe_minus_ccsd_full": d_vqe_ccsd_full,
            },
        })

    return molecule_results


def run_all_molecules(molecules: dict, out_dir: str = "pkl_results", integrals_dir: str = None):
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
            mol_res = run_one_molecule(mol_name, spec, integrals_dir=integrals_dir)
        except Exception as e:
            print(f"[ERROR] {mol_name}: {repr(e)}", flush=True)
            continue

        payload = {mol_name: mol_res}
        save_pkl(payload, out_path)

        print(f"[DONE] {mol_name}", flush=True)

    print(f"[ALL DONE] PKL results saved under: {out_dir}", flush=True)
