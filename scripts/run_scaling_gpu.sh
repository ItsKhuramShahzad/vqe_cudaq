#!/bin/bash
# Scaling test on the GPU cluster: time per energy evaluation for larger active spaces.
# Submit from the repository root:
#     sbatch scripts/run_scaling_gpu.sh
#SBATCH -A phy
#SBATCH --qos=phyh100
#SBATCH -p gpuh100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -J VQE_scaling_gpu
#SBATCH --output=logs/scaling_gpu_%j.log

export OMP_NUM_THREADS=1
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p "$TMPDIR" results/scaling
module purge
module load slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
echo "node $HOSTNAME | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/benchmark_scaling.py --molecule Benzene --basis cc-pVDZ --target nvidia \
    --spaces 6,7 6,8 8,8 6,9 8,9 8,10 10,10 10,11 10,12 --evals 10 --budget 1200 --skip-above 1800 \
    --out results/scaling/scaling_gpu.csv
