#!/bin/bash
# Scaling test on the CPU cluster: time per energy evaluation for larger active spaces.
# Submit from the repository root, once per core count (the node is reserved exclusively
# so the timings are not disturbed; the threads used follow -c):
#     for c in 1 8 26 52; do sbatch -c $c scripts/run_scaling_cpu.sh; done
#SBATCH -A nano
#SBATCH -p ulow
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH -J VQE_scaling_cpu
#SBATCH --output=logs/scaling_cpu_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p "$TMPDIR" results/scaling
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
echo "node $HOSTNAME | cores $OMP_NUM_THREADS | commit $(git rev-parse --short HEAD)"

python scripts/benchmark_scaling.py --molecule Benzene --basis cc-pVDZ --target qpp-cpu \
    --spaces 6,7 6,8 8,8 6,9 8,9 8,10 --evals 10 --budget 1800 --skip-above 1800 \
    --out results/scaling/scaling_cpu_${OMP_NUM_THREADS}cores.csv
