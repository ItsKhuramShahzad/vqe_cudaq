#!/bin/bash
# Bond-type CASCI for all 12 molecules on the cluster: one array task per molecule.
# Each task does the main run (type combinations up to 8e,10o and frozen-core CCSD), then the same
# spaces without HOMO-1, -2, -3 and without LUMO+1, +2, +3, and writes the Excel summary with the
# selection ratio. A molecule whose main CSV already exists skips the main run (restart).
# Submit from the repository root:
#     sbatch scripts/run_bond_type_casci.sh                 # all 12 molecules
#     sbatch --array=0 scripts/run_bond_type_casci.sh       # short test: Ethylene
# If openpyxl is missing, the CSV files are still complete; rebuild the Excel files later with
#     python scripts/bond_type_casci.py --molecule <name> --summary-only
#SBATCH -A nano
#SBATCH -p ulow
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH -J bond_type_casci
#SBATCH --output=logs/bond_type_casci_%A_%a.log
#SBATCH --error=logs/bond_type_casci_%A_%a.err
#SBATCH --array=0-11

MOLECULES=(
  "Ethylene" "Methanamide" "NH2-" "Benzene" "Naphthalene" "Tetracene"
  "Pentacene" "Adenine" "Thymine" "Guanine" "Cytosine" "Uracil"
)
MOLECULE=${MOLECULES[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p "$TMPDIR" logs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
python -c "import openpyxl" 2>/dev/null || echo "openpyxl not found: only the CSV files will be written"
echo "molecule: $MOLECULE  host: $(hostname)  commit: $(git rev-parse --short HEAD)  start: $(date)"

# PySCF memory a little below the SLURM limit, so the two-electron integrals stay in memory
python scripts/bond_type_casci.py --molecule "$MOLECULE" \
    --max-electrons 8 --max-orbitals 10 \
    --leave-out 1 2 3 --leave-out-virtual 1 2 3 \
    --max-memory 56000

echo "end: $(date)"
