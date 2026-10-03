#!/bin/bash
# Width sweep on the one-hot baseline, split 0 only, parameterised by the
# parameter-to-data ratio. 11 widths from very underparametrised to heavily
# overparametrised, 3 seeds each, 33 tasks.
#
#   sbatch slurm/sweep_width.sbatch
#
# CPU only: at 860 inputs even the widest model here is a small matmul, and
# the baseline trains in seconds on CPU, so this leaves the GPU partition free.
#
#SBATCH --job-name=pmhc-sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-32
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out

set -euo pipefail

# Submitted from inside an interactive allocation, that job's binding leaks in
# and contradicts this one. See docs/03_arm_ladder.md section 9.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
unset SLURM_STEP_GPUS SLURM_GPUS_ON_NODE GPU_DEVICE_ORDINAL

PROJECT="${PROJECT:-/home/tau/lmensi/peptide-HLA-stability}"
PYTHON="${PYTHON:-/home/tau/lmensi/miniconda3/envs/myconda/bin/python}"
SPLITS_DIR="${SPLITS_DIR:-$PROJECT/DATA}"
OUT="${OUT:-$PROJECT/RESULTS/sweep}"

# (hidden, hidden/2); params = 860*h + h + h*(h/2) + h/2 + h/2 + 1
WIDTHS=(2 4 8 16 32 64 128 256 512 1024 2048)
SEEDS=(0 1 2)

i="${SLURM_ARRAY_TASK_ID:-0}"
w="${WIDTHS[$((i / ${#SEEDS[@]}))]}"
s="${SEEDS[$((i % ${#SEEDS[@]}))]}"
half=$(( w / 2 )); [ "$half" -lt 1 ] && half=1
out="$OUT/w${w}_s${s}"

cd "$PROJECT"
mkdir -p "$out" slurm/logs
echo "host $(hostname)  task $i  hidden ($w, $half)  seed $s  -> $out"

"$PYTHON" train_mlp.py \
    --splits-dir "$SPLITS_DIR" \
    --splits 1 \
    --seed "$s" \
    --hidden "$w" "$half" \
    --device cpu \
    --output-dir "$out"

echo "done: $out/metrics.json"
