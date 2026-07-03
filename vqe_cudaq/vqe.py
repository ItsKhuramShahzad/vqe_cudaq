"""VQE optimization: single chunk, multi-cycle convergence, and jitter restarts.

All tuning parameters are passed as arguments -- these functions hold no run
configuration state.
"""

import timeit
import numpy as np
from scipy.optimize import minimize
import cudaq

from .ansatz import (
    uccsd_kernel_interleaved,
    hea_kernel_openshell,
    energy_expectation,
)


def optimize_vqe_one_chunk(spin_ham_nc, qubit_count, nele_cas, x0,
                           method="COBYLA", tol=1e-10, maxiter=600, rhobeg=0.2,
                           open_shell=False):
    """Run one bounded optimizer chunk and return the result + timing/trace."""
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


def vqe_until_converged(
    spin_ham_nc, qubit_count, nele_cas, theta_start, rng,
    eps_E=1e-6, patience=3, max_cycles=25,
    chunk_maxiter=600,
    jitter_between_cycles=True, jitter_scale=5e-4,
    method="COBYLA", tol=1e-10, rhobeg=0.2,
    verbose_cycles=False, open_shell=False
):
    """Re-run chunks (optionally jittered) until energy stops improving."""
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


def best_of_jitters_one_chunk(spin_ham_nc, qubit_count, nele_cas, theta0, rng,
                              n_restarts=3, jitter_scale=5e-3,
                              chunk_maxiter=600, method="COBYLA", tol=1e-10, rhobeg=0.2,
                              open_shell=False):
    """Seed search: run several jittered starts and keep the best chunk."""
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
