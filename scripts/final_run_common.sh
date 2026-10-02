# Shared part of run_cpu_final.sh and run_gpu_final.sh (sourced, not run on its own).
# Expects TARGET and OUT_DIR; optional SPACE_IDX runs one active space only (test).

# Same molecules in the same order on both clusters (array index 0-11)
MOLECULES=(
  "Ethylene" "Methanamide" "NH2-" "Benzene" "Naphthalene" "Tetracene"
  "Pentacene" "Adenine" "Thymine" "Guanine" "Cytosine" "Uracil"
)
# Read the Hamiltonians and CCSD amplitudes from integrals/ (empty: start from the geometry)
INTEGRALS="integrals"

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ ! -f vqe_cudaq/cli.py ]; then
  echo "Submit from the vqe_cudaq repository root." >&2; exit 1
fi

# 1) exactly the vqe_final software versions
python - <<'EOF' || exit 1
import sys, cudaq, pyscf, scipy, numpy, openfermion
want = {"python": "3.11.13", "cudaq": "0.11.0", "pyscf": "2.6.2", "scipy": "1.16.0",
        "numpy": "1.26.4", "openfermion": "1.6.1"}
have = {"python": sys.version.split()[0], "cudaq": cudaq.__version__.split()[2],
        "pyscf": pyscf.__version__, "scipy": scipy.__version__,
        "numpy": numpy.__version__, "openfermion": openfermion.__version__}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
print("versions:", have)
if bad:
    sys.exit(f"wrong versions (have, want): {bad}")
EOF

# 2) the package code is exactly the checked-out commit
if ! git diff --quiet HEAD -- vqe_cudaq; then
  echo "vqe_cudaq/ has uncommitted changes; commit or reset them first." >&2; exit 1
fi
echo "commit: $(git rev-parse HEAD)"

MOLECULE=${MOLECULES[$SLURM_ARRAY_TASK_ID]}
EXTRA=()
if [ -n "$INTEGRALS" ]; then EXTRA+=(--integrals "$INTEGRALS"); fi
if [ -n "$SPACE_IDX" ]; then
  EXTRA+=(--space_idx "$SPACE_IDX")
  OUT_DIR="${OUT_DIR/final_2026/final_2026_test}"
fi
mkdir -p "$OUT_DIR"

echo "molecule: $MOLECULE | target: $TARGET | out: $OUT_DIR | job: $SLURM_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "node: $HOSTNAME | OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} | ${EXTRA[*]}"
[ "$TARGET" = "nvidia" ] && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python -m vqe_cudaq.cli \
  --molecule "$MOLECULE" --basis cc-pVDZ --target "$TARGET" --precision fp64 \
  --optimizer COBYLA --out_dir "$OUT_DIR" "${EXTRA[@]}"
