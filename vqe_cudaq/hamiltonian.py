"""Active-space integral files and the qubit Hamiltonian built from them.

The functions below compute the active-space integrals once, save
them to ``.npz`` (``scripts/dump_integrals.py``) and build the qubit Hamiltonian
from the file alone, so a VQE run (``--integrals``) needs no SCF or CCSD. They use
PySCF CASCI (``get_h1eff`` / ``get_h2eff``), so memory is ``norb_cas^4`` instead of
the two full ``nmo^4`` copies that ``openfermionpyscf`` makes. Same Hamiltonian as
``get_molecular_hamiltonian()``, checked to 1e-13.

Each file stores the integrals plus ``e_hf`` and ``e_casci``, so it can be checked
against the molecule it claims to be, and optionally the CCSD amplitudes cut to
the active space (``t1_active``, ``t2_active``), the VQE starting point. Files are
named ``No#_<i>_No(c)_<ncore>_Ne(a)_<nele_cas>_No(a)_<norb_cas>.npz`` under
``<root>/<basis>/<molecule>/``, where ``i`` is the position (from 1) in the
molecule's ``valid_active_spaces``.
"""

import glob
import os
import time

import numpy as np
import pyscf
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.transforms import jordan_wigner, get_fermion_operator
from pyscf import ao2mo, cc, fci, gto, mcscf, scf

from .operators import slice_ccsd_to_active
from .xyz import geometry_in_angstrom


# ══════════════════════════════════════════════════════════════════════
#  Active-space integral files
# ══════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"ncore", "nele_cas", "norb_cas", "n_electrons",
                 "e_core", "h1", "eri", "e_hf", "e_casci"}


class SpaceDoesNotFit(ValueError):
    """The active space does not fit this molecule in this basis. Only this case
    is skipped; any other error stops the run."""


def integral_file_name(index, ncore, nele_cas, norb_cas):
    """File name of one active space, e.g. ``No#_01_No(c)_18_Ne(a)_6_No(a)_4.npz``."""
    return f"No#_{index:02d}_No(c)_{ncore}_Ne(a)_{nele_cas}_No(a)_{norb_cas}.npz"


def run_scf(spec, basis, max_memory=16000):
    """Closed-shell RHF for one molecule entry of :mod:`vqe_cudaq.molecules`."""
    mol = gto.M(atom=geometry_in_angstrom(spec), unit="Angstrom", basis=basis,
                charge=int(spec["charge"]), spin=int(spec["multiplicity"]) - 1,
                max_memory=max_memory, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def run_ccsd(mf, max_memory=16000):
    """CCSD on the whole molecule, all orbitals correlated (as in the driver).
    Returns ``(E_ccsd_total, t1, t2)`` with PySCF's ``t1[i,a]``, ``t2[i,j,a,b]``."""
    mycc = cc.CCSD(mf)
    mycc.max_memory = max_memory
    mycc.verbose = 0
    e_corr, t1, t2 = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    return float(mf.e_tot + e_corr), t1, t2


def compute_active_space(mf, ncore, nele_cas, norb_cas, ccsd=None):
    """Active-space integrals and CASCI energy for one active space.

    ccsd: optional ``(E_ccsd_total, t1, t2)`` from :func:`run_ccsd`. If given, the
    amplitudes are cut to this active space and stored with the integrals."""
    nelec = mf.mol.nelectron
    nmo = mf.mo_coeff.shape[1]
    if 2 * ncore + nele_cas != nelec:
        raise SpaceDoesNotFit(f"2*ncore+nele_cas = {2 * ncore + nele_cas}, "
                              f"but the molecule has {nelec} electrons")
    if ncore + norb_cas > nmo:
        raise SpaceDoesNotFit(f"needs {ncore + norb_cas} orbitals, basis has {nmo}")

    cas = mcscf.CASCI(mf, norb_cas, nele_cas)
    cas.ncore = ncore
    cas.verbose = 0
    h1, e_core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), norb_cas)
    e_casci = float(cas.kernel()[0])

    # the integrals must give back the CASCI energy
    e_check = fci.direct_spin1.kernel(h1, eri, norb_cas, nele_cas)[0] + e_core
    if abs(e_check - e_casci) > 1e-8:
        raise RuntimeError(f"integrals give {e_check}, CASCI gave {e_casci}")

    data = dict(ncore=ncore, nele_cas=nele_cas, norb_cas=norb_cas,
                n_electrons=nelec, nmo=nmo, e_core=float(e_core), h1=h1, eri=eri,
                e_hf=float(mf.e_tot), e_casci=e_casci, pyscf_version=pyscf.__version__)

    if ccsd is not None:
        e_ccsd, t1, t2 = ccsd
        _, _, t1a, t2a = slice_ccsd_to_active(
            t1, t2, nocc=nelec // 2, nmo=nmo,
            active_orbs=list(range(ncore, ncore + norb_cas)))
        data.update(e_ccsd=float(e_ccsd), t1_active=t1a, t2_active=t2a)
    return data


def save_active_space(path, data, **metadata):
    np.savez_compressed(path, **data, **metadata)


def load_active_space(path):
    """Load a file and check it is consistent before returning it."""
    z = np.load(path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(z.files)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    data = {k: z[k] for k in z.files}
    for k in ("ncore", "nele_cas", "norb_cas", "n_electrons"):
        data[k] = int(data[k])
    for k in ("e_core", "e_hf", "e_casci"):
        data[k] = float(data[k])

    if 2 * data["ncore"] + data["nele_cas"] != data["n_electrons"]:
        raise ValueError(f"{path}: 2*ncore+nele_cas does not equal n_electrons")
    n = data["norb_cas"]
    if data["h1"].shape != (n, n) or data["eri"].shape != (n, n, n, n):
        raise ValueError(f"{path}: integral shapes do not match norb_cas={n}")

    if "t1_active" in data:
        no = data["nele_cas"] // 2
        nv = n - no
        if data["t1_active"].shape != (no, nv) or data["t2_active"].shape != (no, no, nv, nv):
            raise ValueError(f"{path}: CCSD amplitude shapes do not match the active space")
        data["e_ccsd"] = float(data["e_ccsd"])
    return data


def qubit_hamiltonian(data):
    """Jordan-Wigner qubit Hamiltonian from the integrals alone."""
    one, two = spinorb_from_spatial(
        data["h1"], np.asarray(data["eri"].transpose(0, 2, 3, 1), order="C"))
    op = InteractionOperator(data["e_core"], one, 0.5 * two)
    return jordan_wigner(get_fermion_operator(op))


def find_integral_file(root, basis, molecule, ncore, nele_cas, norb_cas):
    """Path of the saved file for one active space."""
    folder = os.path.join(root, basis, molecule.replace(" ", "_"))
    pattern = os.path.join(glob.escape(folder),
                           f"No#_*_No(c)_{ncore}_Ne(a)_{nele_cas}_No(a)_{norb_cas}.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no integral file for {molecule} {basis} ncore={ncore} "
                                f"nele_cas={nele_cas} norb_cas={norb_cas} in {folder}")
    if len(matches) > 1:
        raise RuntimeError(f"several integral files for {molecule} {basis} ncore={ncore} "
                           f"nele_cas={nele_cas} norb_cas={norb_cas}: {matches}")
    return matches[0]


def load_integrals(root, basis, molecule, ncore, nele_cas, norb_cas):
    """Load the file for one active space and check it is the one asked for."""
    path = find_integral_file(root, basis, molecule, ncore, nele_cas, norb_cas)
    data = load_active_space(path)
    for key, want in (("molecule", molecule), ("basis", basis)):
        if key in data and str(data[key]) != want:
            raise ValueError(f"{path}: {key}={data[key]} does not match the requested {want}")
    if (data["ncore"], data["nele_cas"], data["norb_cas"]) != (ncore, nele_cas, norb_cas):
        raise ValueError(f"{path}: (ncore, nele_cas, norb_cas) = "
                         f"{(data['ncore'], data['nele_cas'], data['norb_cas'])}, "
                         f"requested {(ncore, nele_cas, norb_cas)}")
    data["path"] = path
    return data


class IntegralFiles:
    """The integral files of one molecule: read them, and make the missing ones.

    A missing file, or one without CCSD amplitudes, is computed, saved and then
    read. SCF and CCSD run at most once per molecule, and only if something is
    missing."""

    def __init__(self, root, molecule, spec, basis, max_memory=16000, all_spaces=None):
        self.root = root
        self.molecule = molecule
        self.spec = spec
        self.basis = basis
        self.max_memory = max_memory
        self.all_spaces = all_spaces if all_spaces is not None else spec["valid_active_spaces"]
        self.mf = self.ccsd = None
        self.seconds = 0.0
        self.made = []

    def get(self, ncore, nele_cas, norb_cas):
        try:
            data = load_integrals(self.root, self.basis, self.molecule, ncore, nele_cas, norb_cas)
            if "t1_active" in data:
                return data
            print(f"[MAKE] {data['path']} has no CCSD amplitudes, computing them", flush=True)
        except FileNotFoundError:
            print(f"[MAKE] {self.molecule} {self.basis} ncore={ncore} nele_cas={nele_cas} "
                  f"norb_cas={norb_cas}: no file, computing it", flush=True)
        self.make(ncore, nele_cas, norb_cas)
        return load_integrals(self.root, self.basis, self.molecule, ncore, nele_cas, norb_cas)

    def make(self, ncore, nele_cas, norb_cas):
        if self.mf is None:
            print(f"[MAKE] running SCF + CCSD once for {self.molecule} in {self.basis} "
                  f"to write the integrals", flush=True)
            t0 = time.time()
            self.mf = run_scf(self.spec, self.basis, max_memory=self.max_memory)
            self.ccsd = run_ccsd(self.mf, max_memory=self.max_memory)
            self.seconds += time.time() - t0

        data = compute_active_space(self.mf, ncore, nele_cas, norb_cas, ccsd=self.ccsd)

        mol_dir = os.path.join(self.root, self.basis, self.molecule.replace(" ", "_"))
        os.makedirs(mol_dir, exist_ok=True)
        try:    # replacing a file without amplitudes: keep its name
            path = find_integral_file(self.root, self.basis, self.molecule,
                                      ncore, nele_cas, norb_cas)
        except FileNotFoundError:
            path = os.path.join(mol_dir, integral_file_name(
                self.number(ncore, nele_cas, norb_cas), ncore, nele_cas, norb_cas))
        # private name first, then rename: a half-written file is never visible
        temp = os.path.join(mol_dir, f".tmp.{os.uname().nodename}.{os.getpid()}.npz")
        save_active_space(temp, data, molecule=self.molecule, basis=self.basis,
                          charge=int(self.spec["charge"]),
                          multiplicity=int(self.spec["multiplicity"]))
        os.replace(temp, path)
        self.made.append(path)
        print(f"[MAKE] saved {path} (SCF + CCSD so far {self.seconds:.1f} s)", flush=True)

    def number(self, ncore, nele_cas, norb_cas):
        """Position of the space in the molecule's full list, from 1 (0 if absent)."""
        for i, s in enumerate(self.all_spaces, start=1):
            if (int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"])) == (ncore, nele_cas, norb_cas):
                return i
        return 0
