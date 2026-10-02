#!/bin/bash
# Final benchmark, CPU (fe02, one full node per molecule). Submit from the repository root:
#     sbatch scripts/run_cpu_final.sh                    # all 12 molecules
#     SPACE_IDX=7 sbatch --array=3 scripts/run_cpu_final.sh   # short test: Benzene (2,3)
#SBATCH -p ulow
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH -J VQE_CPU_final
#SBATCH -o slurm_cpu_%A_%a.log
#SBATCH --array=0-11

TARGET="qpp-cpu"
OUT_DIR="results/final_2026/cpu"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

source scripts/final_run_common.sh
