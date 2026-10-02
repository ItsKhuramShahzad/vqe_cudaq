#!/bin/bash
# Final benchmark, GPU (lyra, H100). One array task per molecule; its 9 active spaces
# run one after another and give one PKL per molecule. Submit from the repository root:
#     sbatch scripts/run_gpu_final.sh                         # all 12 molecules
#     SPACE_IDX=7 sbatch --array=3 scripts/run_gpu_final.sh   # short test: Benzene (2,3)
# Resources: 1 GPU, 1 core, 16 GB host memory (peak 4.4 GB, Pentacene (6,7)).
#SBATCH -A phy
#SBATCH --qos=phyh100
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH -J VQE_GPU_final
#SBATCH --output=logs/output_gpu_%A_%a.log
#SBATCH --error=logs/error_gpu_%A_%a.log
#SBATCH --array=0-11

TARGET="nvidia"
OUT_DIR="results/final_2026/gpu"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p "$TMPDIR"
module purge
module load slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

source scripts/final_run_common.sh
