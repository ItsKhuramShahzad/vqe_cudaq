#!/bin/bash
# Final benchmark, CPU (fe01/fe02, ulow). One array task per molecule; its 9 active spaces
# run one after another and give one PKL per molecule. Submit from the repository root:
#     sbatch scripts/run_cpu_final.sh                         # all 12 molecules
#     SPACE_IDX=7 sbatch --array=3 scripts/run_cpu_final.sh   # short test: Benzene (2,3)
# Resources: 2 cores (as each MIMIQ space), 8 GB (peak 4.4 GB, Pentacene (6,7)).
#SBATCH -A nano
#SBATCH -p ulow
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH -J VQE_CPU_final
#SBATCH --output=logs/output_cpu_%A_%a.log
#SBATCH --error=logs/error_cpu_%A_%a.log
#SBATCH --array=0-11

TARGET="qpp-cpu"
OUT_DIR="results/final_2026/cpu"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p "$TMPDIR"
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

source scripts/final_run_common.sh
