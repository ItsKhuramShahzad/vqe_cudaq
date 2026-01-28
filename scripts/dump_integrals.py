import os
import numpy as np

from vqe_cudaq.molecules import molecules
from vqe_cudaq.hamiltonian import molecularHamiltonian


def dump_integrals_all(
    bases=("cc-pVDZ", "sto-3g", "6-31g"),
    outdir="integrals",
):
    """
    Dump one- and two-electron integrals for:
      - all molecules
      - all active spaces
      - multiple basis sets

    Errors in any molecule or active space do NOT stop execution.
    """

    os.makedirs(outdir, exist_ok=True)

    for basis in bases:
        print(f"\n==============================")
        print(f" Basis set: {basis}")
        print(f"==============================")

        basis_dir = os.path.join(outdir, basis)
        os.makedirs(basis_dir, exist_ok=True)

        for mol_name, mol in molecules.items():
            print(f"\n→ Molecule: {mol_name}")

            mol_dir = os.path.join(basis_dir, mol_name.replace(" ", "_"))
            os.makedirs(mol_dir, exist_ok=True)

            try:
                active_spaces = mol.get("valid_active_spaces", [])
            except Exception as e:
                print(f"  [SKIP MOLECULE] Cannot read active spaces → {e}")
                continue

            for i, space in enumerate(active_spaces, start=1):
                try:
                    No_c = space["ncore"]
                    Ne_a= space["nele_cas"]
                    No_a = space["norb_cas"]

                    tag = (
                        f"No#_{i:02d}_"
                        f"No(c)_{No_c}_"
                        f"Ne(a)_{Ne_a}_"
                        f"No(a)_{No_a}"
                    )

                    save_path = os.path.join(mol_dir, f"{tag}.npz")

                    mol_ham, _, _, _, _, _, _, _ = molecularHamiltonian(
                        geometry=mol["geometry"],
                        basis=basis,
                        multiplicity=mol["multiplicity"],
                        charge=mol["charge"],
                        ncore=No_c,
                        nele_cas=Ne_a,
                        norb_cas=No_a,
                        ac=True,
                    )

                    E_core = float(mol_ham.constant)
                    h1 = np.array(mol_ham.one_body_tensor)
                    h2 = np.array(mol_ham.two_body_tensor)

                    np.savez_compressed(
                        save_path,
                        molecule=mol_name,
                        basis=basis,
                        No_c=No_c,
                        Ne_a=Ne_a,
                        No_a=No_a,
                        E_core=E_core,
                        h1=h1,
                        h2=h2,
                    )

                    print(
                        f"  [OK] {tag} | "
                        f"h1{h1.shape} h2{h2.shape}"
                    )

                except Exception as e:
                    print(
                        f"[FAIL] Active space #{i} "
                        f"(ncore={space.get('ncore')}, "
                        f"nele={space.get('nele_cas')}, "
                        f"norb={space.get('norb_cas')}) "
                        f"→ {e}"
                    )
                    continue  # ← IMPORTANT: continue to next active space
        break; 


if __name__ == "__main__":
    dump_integrals_all()
