#!/bin/bash
# =========================================================================== #
# STOCK ARM — ray_init-kwargs variant (verl-env-2's original approach: no     #
# `ray start` CLI; fixed 50GB object store, short temp dir, NOSET_CUDA,      #
# cpu/gpu counts via verl's ray_init config). Persist raylet logs always.     #
#   MODE=sanity|timing sbatch examples/gigpo_trainer/slurm_babyai_race_rayinit.sbatch
# =========================================================================== #
#SBATCH --job-name=race-stock-ri
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6re/jdilkes.u6re/frameworks-spike/race-stock-ri-%j.out
#SBATCH --error=/home/u6re/jdilkes.u6re/frameworks-spike/race-stock-ri-%j.err

set -euxo pipefail
MODE="${MODE:-sanity}"

cd "$HOME/frameworks-spike/verl-agent"
source "$HOME/frameworks-spike/venv/bin/activate"

export HF_HOME=/scratch/u6re/jdilkes.u6re/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR=/tmp/race_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$TMPDIR/.cache
export TRITON_CACHE_DIR=$TMPDIR/.cache/triton
mkdir -p "$TRITON_CACHE_DIR"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$HOME/frameworks-spike/wandb"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING

# --- verl-env-2 original Ray recipe ---------------------------------------- #
SHORT_RAY_DIR=/tmp/ray/u6re/${SLURM_JOB_ID}
mkdir -p "$SHORT_RAY_DIR"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_OBJECT_STORE_MEMORY=50000000000
export RAY_DISABLE_IMPORT_WARNING=1 RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0
unset RAY_ADDRESS || true

persist_ray_logs() {
    rc=$?
    dst=$HOME/frameworks-spike/ray_logs/${SLURM_JOB_ID}
    mkdir -p "$dst"
    cp -r "$SHORT_RAY_DIR"/session_latest/logs "$dst"/ 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
        echo "=== RAYLET LOGS (exit $rc) ==="
        tail -60 "$SHORT_RAY_DIR"/session_latest/logs/raylet.err 2>/dev/null || true
        tail -30 "$SHORT_RAY_DIR"/session_latest/logs/gcs_server.err 2>/dev/null || true
    fi
}
trap persist_ray_logs EXIT
# --------------------------------------------------------------------------- #

if [ "$MODE" = "sanity" ]; then
    RUN_SCRIPT=examples/gigpo_trainer/run_babyai_qwen3_1.7b.sh
    STEPS=4; VAL_BEFORE=True
else
    RUN_SCRIPT=examples/gigpo_trainer/run_babyai_qwen3_4b.sh
    STEPS=12; VAL_BEFORE=False
fi

bash "$RUN_SCRIPT" vllm \
    trainer.total_training_steps=$STEPS \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=$VAL_BEFORE \
    trainer.experiment_name="race-stock-ri-${MODE}-j${SLURM_JOB_ID}" \
    ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-32}" \
    "+ray_init._temp_dir=$SHORT_RAY_DIR"

echo "RACE ${MODE} DONE (rayinit variant)"
