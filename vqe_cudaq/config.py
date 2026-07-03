"""
Global run settings for the VQE benchmark engine.

These module-level attributes are the single source of truth for the run
configuration. The CLI (``vqe_cudaq.cli``) mutates them in place before a run
(e.g. ``config.BASIS = args.basis``); every other module reads them through
the ``config`` namespace (or snapshots them at function entry), so a CLI
override is seen consistently everywhere.
"""

import time
import numpy as np

# ── What to run ───────────────────────────────────────────────────────
TAG = time.strftime("%d_%b_%Y").upper()   # e.g. "23_FEB_2026"
BASIS = "cc-pVDZ"
TARGET = "qpp-cpu"                          # e.g. "nvidia" or "qpp-cpu"
OPTIMIZER = "COBYLA"
TARGET_PRECISION = None                     # None = default, "fp32"/"fp64" (nvidia)
RUN_CCSD_REFERENCE = True

# ── Determinism ───────────────────────────────────────────────────────
SEED = 12345
rng_global = np.random.default_rng(SEED)

# ── Optimizer settings ────────────────────────────────────────────────
TOL = 1e-10
COBYLA_RHOBEG = 0.2

# Full strength now that the CUDA-Q parameter packing is correct
# (previously 0.3, a band-aid for a broken packer).
THETA_SCALE = 1.0

N_JITTER_RESTARTS = 3
JITTER_SCALE = 5e-3

DIAG_MAX_PARAMS = 0
DIAG_EPS = 1e-3

# ── Multi-cycle VQE convergence ───────────────────────────────────────
VQE_EPS_E = 1e-6
VQE_PATIENCE = 3
VQE_MAX_CYCLES = 25
VQE_CHUNK_MAXITER = 600
VQE_JITTER_BETWEEN_CYCLES = True
VQE_JITTER_BETWEEN_SCALE = 5e-4

# ── Heavy (large-parameter) active spaces ─────────────────────────────
HEAVY_PARAM_THRESHOLD = 150
HEAVY_RESTARTS = 0
HEAVY_RHOBEG = 0.05

# ── Verbosity / tolerances ────────────────────────────────────────────
VERBOSE = False
PRINT_EVERY_CYCLE = False
JW_IMAG_TOL = 1e-6
