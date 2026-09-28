"""Small, dependency-light helpers (logging, filenames, pickling, hashing)."""

import re
import pickle
import hashlib

from . import config


def log(msg: str):
    """Print only when ``config.VERBOSE`` is enabled."""
    if config.VERBOSE:
        print(msg, flush=True)


def sanitize_name(name: str) -> str:
    """Make a filesystem-safe token from a molecule/basis/target name."""
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-\+]", "", name)
    return name


def save_pkl(obj, path: str):
    """Pickle ``obj`` to ``path`` at the highest protocol."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def stable_hash(obj) -> int:
    """Deterministic hash across Python sessions / machines.

    Python's built-in ``hash()`` randomizes string hashing per-process unless
    ``PYTHONHASHSEED`` is fixed before interpreter start, which means
    ``hash((mol_name, ...))`` differs between runs and between CPU/GPU
    machines. ``hashlib.sha256`` is deterministic by construction.
    """
    return int(hashlib.sha256(repr(obj).encode()).hexdigest(), 16) % (2**31)


def run_metadata() -> dict:
    """Software versions, machine, SLURM job and a hash of the package code.

    Stored in every result PKL so CPU and GPU runs can be checked to use the same
    code and environment. ``script_sha256`` hashes all ``vqe_cudaq/*.py`` files.
    """
    import os
    import glob
    import platform
    import socket
    import subprocess
    import numpy as np
    import scipy
    import pyscf
    import openfermion
    import cudaq

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    code = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        with open(path, "rb") as fh:
            code.update(os.path.basename(path).encode())
            code.update(fh.read())

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
        "script_name": "vqe_cudaq",
        "script_sha256": code.hexdigest(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        meta["gpu"] = out.stdout.strip() or None
    except Exception:
        meta["gpu"] = None
    return meta
