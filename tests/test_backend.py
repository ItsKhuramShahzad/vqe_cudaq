"""CUDA-Q version tripwire, fp64 enforcement and the metadata stored with every run."""

import pytest

import cudaq
from vqe_cudaq import backend, config
from vqe_cudaq.utils import run_metadata, sanitize_name, stable_hash


def test_cuda_q_version_is_one_the_packing_was_checked_on():
    assert any(v in cudaq.__version__ for v in ("0.11", "0.12", "0.14"))


def test_cpu_target_is_double_precision():
    assert backend.configure_cudaq_target() .lower().endswith("fp64")


def test_a_single_precision_target_stops_the_run(monkeypatch):
    class Fp32Target:
        name = "nvidia"

        def get_precision(self):
            return "SimulationPrecision.fp32"

    monkeypatch.setattr(backend.cudaq, "get_target", lambda: Fp32Target())
    with pytest.raises(RuntimeError, match="fp64"):
        backend.configure_cudaq_target()


def test_default_precision_is_fp64():
    from vqe_cudaq.cli import build_parser
    assert config.TARGET_PRECISION == "fp64"
    assert build_parser().parse_args([]).precision == "fp64"


def test_run_metadata():
    import numpy, pyscf, scipy
    meta = run_metadata()
    for key in ("python", "cudaq", "pyscf", "scipy", "numpy", "openfermion", "hostname",
                "slurm_job_id", "omp_num_threads", "os_cpu_count", "script_sha256", "gpu"):
        assert key in meta, key
    assert meta["cudaq"] == cudaq.__version__
    assert (meta["pyscf"], meta["scipy"], meta["numpy"]) == (
        pyscf.__version__, scipy.__version__, numpy.__version__)
    assert len(meta["script_sha256"]) == 64
    assert run_metadata()["script_sha256"] == meta["script_sha256"]
    # this repository is a git checkout: the commit is recorded
    assert len(meta["git_commit"]) == 40 and isinstance(meta["git_dirty"], bool)


def test_seeds_are_the_same_on_every_machine():
    """The per-space random generator is seeded from sha256, not Python's hash(), so CPU
    and GPU runs draw the same jitters. The value is fixed; a change breaks comparability."""
    assert stable_hash(("Ethylene", 6, 4, 3)) == 1774996446
    assert config.SEED == 12345


def test_sanitize_name():
    assert sanitize_name("NH2-") == "NH2-"
    assert sanitize_name(" cc-pVDZ ") == "cc-pVDZ"
    assert sanitize_name("Water dimer (A)") == "Water_dimer_A"
