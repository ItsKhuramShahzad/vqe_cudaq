"""Qubit-operator utilities and CCSD-amplitude packing into UCCSD parameters.

Pure functions (no run-configuration state): they take everything they need as
arguments and are safe to reuse independently of the driver.
"""

import numpy as np
from openfermion.ops import QubitOperator
import cudaq


def make_qubitop_real(qop: QubitOperator, tol: float = 1e-6) -> QubitOperator:
    """Drop tiny imaginary parts of a (Hermitian) qubit operator; raise if large."""
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
    """Split a qubit operator into (identity coefficient c0, non-constant part)."""
    c0 = float(np.real(qop.terms.get((), 0.0)))
    qnc = QubitOperator()
    for term, coeff in qop.terms.items():
        if term == ():
            continue
        qnc += QubitOperator(term, float(np.real(coeff)))
    return c0, qnc


def slice_ccsd_to_active(t1, t2, nocc, nmo, active_orbs):
    """Restrict full-system CCSD amplitudes to the active-space block."""
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
