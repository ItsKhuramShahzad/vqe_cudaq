"""
Active-space integrals: compute them once, save to .npz, build the Hamiltonian
from the file alone.

compute_active_space() uses PySCF CASCI (get_h1eff, get_h2eff), so memory is
norb_cas^4 instead of the two full nmo^4 copies that
openfermionpyscf.compute_integrals() makes. Same Hamiltonian, checked to 1e-13.

Each file stores the integrals plus e_hf and e_casci, so a file can be checked
against the molecule it claims to be. Optionally it also stores the CCSD
amplitudes cut to the active space (t1_active, t2_active), which are the VQE
starting point, so a VQE run needs nothing but the file.
"""

import glob
import os
import time
import numpy as np
import pyscf
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, cc, fci, gto, mcscf, scf

REQUIRED_KEYS = {"ncore", "nele_cas", "norb_cas", "n_electrons",
                 "e_core", "h1", "eri", "e_hf", "e_casci"}

class SpaceDoesNotFit(ValueError):
    """The active space does not fit this molecule in this basis. Only this case is
    skipped; any other error stops the run."""

def run_scf(entry, basis, max_memory=16000):
    mol = gto.M(atom=entry["geometry"], basis=basis, charge=int(entry["charge"]),
                spin=int(entry["multiplicity"]) - 1, max_memory=max_memory, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def run_ccsd(mf, max_memory=16000):
    """CCSD on the whole molecule, all orbitals correlated (as in the CUDA-Q script).
    Returns (E_ccsd_total, t1, t2) with PySCF's t1[i,a], t2[i,j,a,b]."""
    mycc = cc.CCSD(mf)
    mycc.max_memory = max_memory
    mycc.verbose = 0
    e_corr, t1, t2 = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    return float(mf.e_tot + e_corr), t1, t2


def slice_ccsd_to_active(t1, t2, nocc, active_orbitals):
    """Keep only the amplitudes inside the active space.
    Occupied indices count from 0, virtual indices count from 0 after nocc."""
    occ = [p for p in active_orbitals if p < nocc]
    vir = [p - nocc for p in active_orbitals if p >= nocc]
    if not occ or not vir:
        return (np.zeros((len(occ), len(vir))),
                np.zeros((len(occ),) * 2 + (len(vir),) * 2))
    return (np.asarray(t1)[np.ix_(occ, vir)],
            np.asarray(t2)[np.ix_(occ, occ, vir, vir)])


def compute_active_space(mf, ncore, nele_cas, norb_cas, ccsd=None):
    """Active-space integrals and CASCI energy for one active space.

    ccsd: optional (E_ccsd_total, t1, t2) from run_ccsd(). If given, the
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
        t1a, t2a = slice_ccsd_to_active(t1, t2, nelec // 2,
                                        list(range(ncore, ncore + norb_cas)))
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
    
    """ Path to save file for one active space, e.g
    Integrals/cc-pvdz/Ethylene/space_05_ncore_6_nele_4_norb_3.npz
    """
    
    pattern = os.path.join(root, basis, molecule, f"space_*_ncore_{ncore}_nele_{nele_cas}_norb_{norb_cas}.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No integral file found for {molecule} in {basis} with ncore={ncore}, nele_cas={nele_cas}, norb_cas={norb_cas}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple integral files found for {molecule} in {basis} with ncore={ncore}, nele_cas={nele_cas}, norb_cas={norb_cas}: {matches}")
    return matches[0]

def load_integrals(root, basis, molecule, ncore, nele_cas, norb_cas):
    """Load an integral file for a one active space and check the file is the one asked for."""
    path = find_integral_file(root, basis, molecule, ncore, nele_cas, norb_cas)
    data = load_active_space(path)
    
    for key, want in (("molecule", molecule), ("basis", basis)):
        if key in data and str(data[key]) != want:
            raise ValueError(f"{path}: {key}={data[key]} does not match the requested {want}")
    if (data["ncore"], data["nele_cas"], data["norb_cas"]) != (ncore, nele_cas, norb_cas):
        raise ValueError(f"{path}: (ncore,nele_cas,norb_cas)={data['ncore'],data['nele_cas'],data['norb_cas']} does not match the requested {(ncore, nele_cas, norb_cas)}")
    data["path"] = path
    return data

class IntegralFiles:
    """The integral files of one molecules: read them, and make the missing one..
    A missing file , or one without ccd, is computed, saved adn then read.
    
    scf and ccsd run at most onece per molecules, and only if someting is missing
 
    """
    def __init__(self, root, molecule, spec, basis, max_memory =16000, all_spaces= None):
        self.root = root
        self.molecule = molecule
        self.spec = spec
        self.basis = basis
        self.max_memory = max_memory 
        self.all_spaces = all_spaces  if all_spaces is not None else spec["valid_active_spaces"]
        self.mf = self.ccsd = None
        self.seconds = 0.0
        self.made = []
    
    def get (self, ncore, nele_cas, norb_cas):
        
        try:
            data = load_integrals(self.root, self.basis, self.molecule, ncore, nele_cas, norb_cas)
            if "t1_active" in data:
                return data
            print(f"[MAKE] {data['path']} has no CCSD amplitudes, computing them", flush=True)
        except FileNotFoundError:
            print(f"[MAKE] {self.molecule} {self.basis} ncore={ncore}, nele_cas={nele_cas}, norb_cas={norb_cas} not found, computing it", flush=True)   
            pass
        self.make(ncore, nele_cas, norb_cas)
        
        return load_integrals(self.root, self.basis, self.molecule, ncore, nele_cas, norb_cas)
    
    def make(self, ncore, nele_cas, norb_cas):
        if self.mf is None:
            print(f"[MAKE] running SCF + ccsd once for {self.molecule} in {self.basis}"
                  f"to write the integrals", flush=True)
            
            t0= time.time()
            self.mf = run_scf(self.spec, self.basis, max_memory=self.max_memory)
            self.ccsd= run_ccsd(self.mf, max_memory=self.max_memory)
            self.seconds += time.time() - t0 
            # print(f"[MAKE] SCF + CCSD took {time.time()-t0:.
        
        data = compute_active_space(self.mf, ncore, nele_cas, norb_cas, ccsd=self.ccsd)
        
        mol_dir= os.path.join(self.root, self.basis, self.molecule)
        os.makedirs(mol_dir, exist_ok=True)
        
        try:    # replacing a file without amplitudes: keept its name
            path = find_integral_file(self.root, self.basis, self.molecule, ncore, nele_cas, norb_cas)
            tag= os.path.basename(path)[:-len(".npz")]
        except FileNotFoundError:
            tag= f"space_{self.number(ncore, nele_cas, norb_cas):02d}_ncore_{ncore}_nele_{nele_cas}_norb_{norb_cas}"
            path= os.path.join(mol_dir, tag+".npz")
        # private name first, then rename: a halt written is never visible
        temp = os.path.join(mol_dir, f".{tag}.{os.uname().nodename}.{os.getpid()}.npz")
        save_active_space(temp, data, molecule=self.molecule, basis=self.basis,
                          charge =int(self.spec["charge"]), multiplicity=int(self.spec["multiplicity"]) )
        os.rename(temp, path)
        
        self.made.append(path)
        print(f"[MAKE] saved {path}: total time {self.seconds:.1f}s", flush=True)
        
    def number(self, ncore, nele_cas, norb_cas):
       """ same numbering as the dumpt script: position in the full list, form 1."""
       
       for i, s in enumerate (self.all_spaces, start=1):
           if (int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"])) == (ncore, nele_cas, norb_cas):
               return i
       return 0