#!/bin/bash
# Final benchmark, GPU (lyra, H100). Submit from the repository root:
#     sbatch scripts/run_gpu_final.sh                    # all 12 molecules
#     SPACE_IDX=7 sbatch --array=3 scripts/run_gpu_final.sh   # short test: Benzene (2,3)
#SBATCH -p gpuh100
#SBATCH -A phy
#SBATCH --qos=phyh100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB
#SBATCH --time=20-24:00:00
#SBATCH -J VQE_GPU_final
#SBATCH -o slurm_gpu_%A_%a.log
#SBATCH --array=0-11

TARGET="nvidia"
OUT_DIR="results/final_2026/gpu"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

source scripts/final_run_common.sh
