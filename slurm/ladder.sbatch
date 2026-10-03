#!/bin/bash
# Slurm job array for the arm ladder: one task per (arm, pooling) pair.
#
#   sbatch slurm/ladder.sbatch              # all 7 tasks
#   sbatch --array=1-3 slurm/ladder.sbatch  # just the pca:20 arms
#
# Each task trains all five splits for one arm and writes a complete
# metrics.json, so nothing needs merging afterwards; scripts/compare_arms.py
# reads the output directories directly.
#
#SBATCH --job-name=pmhc-ladder
#SBATCH --partition=tau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out
#SBATCH --requeue

set -euo pipefail

# sbatch exports the submitting environment, so when this is submitted from
# inside an interactive allocation that job's binding and GPU variables leak in
# and contradict this job's own allocation ("Unable to satisfy cpu bind
# request", "No devices were found"). Drop them and let this job's values
# stand. Also no srun below: for a single-task job it adds nothing and is what
# actually enforces the stale binding.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
unset SLURM_STEP_GPUS SLURM_GPUS_ON_NODE GPU_DEVICE_ORDINAL
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

PROJECT="${PROJECT:-/home/tau/lmensi/peptide-HLA-stability}"
PYTHON="${PYTHON:-/home/tau/lmensi/miniconda3/envs/myconda/bin/python}"
# The cache lives on /scratch: /home is 97% full and a session temp directory
# is tmpfs, i.e. RAM. Overridable so a node-local copy can be used instead.
EMBEDDINGS="${EMBEDDINGS:-/scratch/lmensi/peptide-HLA-stability/embeddings}"
SPLITS_DIR="${SPLITS_DIR:-$PROJECT/DATA}"
RESULTS="${RESULTS:-$PROJECT/RESULTS}"

# task -> arm, pooling, output directory. 48G covers the worst case (flatten,
# ~15 GiB peak); the pca and mean tasks need about 5 GiB.
ARMS=(onehot L0      L15     L33     L0      L15     L33)
POOL=(none   pca:20  pca:20  pca:20  flatten flatten flatten)
NAME=(A0_onehot A1_L0_pca20 A2_L15_pca20 A3_L33_pca20 A1_L0_flatten A2_L15_flatten A3_L33_flatten)

i="${SLURM_ARRAY_TASK_ID:-0}"
arm="${ARMS[$i]}"
pooling="${POOL[$i]}"
out="$RESULTS/${NAME[$i]}"

cd "$PROJECT"
mkdir -p "$out" slurm/logs

echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  task $i"
echo "arm $arm  pooling $pooling  ->  $out"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "$arm" = "onehot" ]; then
    # Arm A0 needs no embedding cache and no GPU; its features are built in
    # memory from the split files.
    "$PYTHON" train_mlp.py \
        --splits-dir "$SPLITS_DIR" \
        --output-dir "$out"
else
    "$PYTHON" train_mlp.py \
        --splits-dir "$SPLITS_DIR" \
        --arm "$arm" \
        --pooling "$pooling" \
        --embeddings-dir "$EMBEDDINGS" \
        --output-dir "$out"
fi

echo "done: $out/metrics.json"
