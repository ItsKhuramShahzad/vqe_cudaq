"""CUDA-Q backend selection and a version tripwire.

The factor-of-2 UCCSD parameter packing (see :mod:`operators`) relies on the
specific gate decomposition inside ``cudaq.kernels.uccsd``'s
``double_excitation_opt`` (8x ``rz(0.125*theta)``). It was verified identical
across cudaq 0.11.x / 0.12.x / 0.14.x. If you upgrade to a new version, redo
the H2 sanity check before trusting the energies.
"""

import cudaq

from . import config

# ── VERSION TRIPWIRE ──────────────────────────────────────────────────
_CUDAQ_VER = cudaq.__version__
if not any(v in _CUDAQ_VER for v in ("0.11", "0.12", "0.14")):
    raise RuntimeError(
        f"Packing verified against cudaq 0.11/0.12/0.14; got {_CUDAQ_VER}"
    )


def configure_cudaq_target():
    """Set the CUDA-Q target and report/check backend precision.

    - ``qpp-cpu``: default is double precision.
    - ``nvidia``: default is usually fp32 unless ``option="fp64"`` is given.
    """
    target = config.TARGET
    precision_option = config.TARGET_PRECISION

    if target == "nvidia" and precision_option is not None:
        cudaq.set_target(target, option=precision_option)
    else:
        cudaq.set_target(target)

    print("CUDA-Q target:", cudaq.get_target().name, flush=True)

    try:
        precision = str(cudaq.get_target().get_precision())
    except Exception as e:
        precision = f"unknown ({e})"

    print("CUDA-Q precision:", precision, flush=True)
    return precision
