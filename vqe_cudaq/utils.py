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
