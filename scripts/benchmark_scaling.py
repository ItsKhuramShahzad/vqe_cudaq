"""Time one VQE energy evaluation for larger active spaces (scaling test).

For one molecule, runs RHF once, then for each active space builds the integrals,
the Jordan-Wigner Hamiltonian and the UCCSD circuit exactly as a VQE run does, and
times single energy evaluations (cudaq.observe). One CSV row per space.

    python scripts/benchmark_scaling.py --molecule Benzene --target nvidia \
        --spaces 6,8 8,8 6,9 8,9 8,10 10,10 --out scaling_gpu.csv

Run from the repository root. The estimate of a full VQE assumes about
40 energy evaluations per UCCSD parameter, as in the 6-14 qubit runs.
"""

import argparse
import csv
import os
import resource
import socket
import sys
import time

import numpy as np
import cudaq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqe_cudaq.hamiltonian import run_scf, compute_active_space, qubit_hamiltonian
from vqe_cudaq.molecules import molecules
from vqe_cudaq.operators import make_qubitop_real, split_constant
from vqe_cudaq.ansatz import energy_expectation, final_state_diagnostics

EVALS_PER_PARAMETER = 40


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2   # kB -> GB (Linux)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--molecule", default="Benzene")
    p.add_argument("--basis", default="cc-pVDZ")
    p.add_argument("--target", default="qpp-cpu", choices=["qpp-cpu", "nvidia"])
    p.add_argument("--spaces", nargs="+", default=["6,8", "8,8", "6,9", "8,9", "8,10", "10,10"],
                   help="active spaces as nele,norb")
    p.add_argument("--evals", type=int, default=20, help="evaluations to time per space")
    p.add_argument("--budget", type=float, default=900.0,
                   help="seconds of timing per space; stops earlier if reached")
    p.add_argument("--skip-above", type=float, default=1800.0,
                   help="skip the remaining spaces once one evaluation takes longer (s)")
    p.add_argument("--diagnostics", action="store_true",
                   help="also time the final-state analysis (needs much memory above 14 qubits)")
    p.add_argument("--max-memory", type=int, default=64000, help="PySCF memory limit in MB")
    p.add_argument("--out", default="scaling.csv")
    args = p.parse_args()

    if args.target == "nvidia":
        cudaq.set_target("nvidia", option="fp64")
    else:
        cudaq.set_target("qpp-cpu")
    precision = str(cudaq.get_target().get_precision())
    if "fp64" not in precision.lower():
        raise SystemExit(f"precision is {precision}, expected fp64")

    spec = molecules[args.molecule]
    nelec = int(spec["Total Electrons"])
    threads = os.environ.get("OMP_NUM_THREADS")
    print(f"{args.molecule} {args.basis} | target {args.target} ({precision}) | "
          f"OMP_NUM_THREADS={threads} | {socket.gethostname()}", flush=True)

    t0 = time.perf_counter()
    mf = run_scf(spec, args.basis, max_memory=args.max_memory)
    print(f"RHF {mf.e_tot:.8f} in {time.perf_counter() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(12345)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["molecule", "basis", "nele", "norb", "qubits", "parameters", "pauli_terms",
                        "target", "omp_threads", "evals_timed", "s_per_eval_median", "s_per_eval_min",
                        "s_per_eval_max", "build_s", "diagnostics_s", "peak_rss_gb",
                        "est_vqe_hours", "e_casci", "hostname", "cpu_model"])
        for item in args.spaces:
            nele, norb = (int(x) for x in item.split(","))
            ncore = (nelec - nele) // 2
            nq = 2 * norb
            tb = time.perf_counter()
            data = compute_active_space(mf, ncore, nele, norb)
            c0, qop_nc = split_constant(make_qubitop_real(qubit_hamiltonian(data)))
            ham = cudaq.SpinOperator(qop_nc)
            npar = int(cudaq.kernels.uccsd_num_parameters(nele, nq))
            build = time.perf_counter() - tb
            theta = rng.normal(0.0, 0.01, npar)        # the cost does not depend on the values

            energy_expectation(ham, nq, nele, theta)     # warm-up (kernel set-up)
            times = []
            t_start = time.perf_counter()
            while len(times) < args.evals and time.perf_counter() - t_start < args.budget:
                te = time.perf_counter()
                e = energy_expectation(ham, nq, nele, theta)
                times.append(time.perf_counter() - te)
            med = float(np.median(times))

            diag = ""
            if args.diagnostics:
                td = time.perf_counter()
                out = final_state_diagnostics(make_qubitop_real(qubit_hamiltonian(data)),
                                              theta, nq, nele, c0 + e)
                diag = f"{time.perf_counter() - td:.1f}"
                print(f"   final-state check: E_check - E = {out['E_check_minus_E_vqe']:.1e}", flush=True)

            est_h = med * EVALS_PER_PARAMETER * npar / 3600
            row = [args.molecule, args.basis, nele, norb, nq, npar, len(qop_nc.terms), args.target,
                   threads, len(times), f"{med:.4f}", f"{min(times):.4f}", f"{max(times):.4f}",
                   f"{build:.1f}", diag, f"{peak_rss_gb():.2f}", f"{est_h:.1f}",
                   f"{data['e_casci']:.8f}", socket.gethostname(), cpu_model()]
            w.writerow(row)
            fh.flush()
            print(f"({nele},{norb}) {nq} qubits, {npar} parameters, {len(qop_nc.terms)} terms: "
                  f"{med:.3f} s/eval ({len(times)} timed) -> full VQE about {est_h:.1f} h; "
                  f"peak memory {peak_rss_gb():.1f} GB", flush=True)
            if med > args.skip_above:
                print(f"one evaluation takes more than {args.skip_above:.0f} s: "
                      f"skipping the larger spaces", flush=True)
                break


if __name__ == "__main__":
    main()
