#!/bin/bash
# =========================================================================== #
# STOCK ARM — BabyAI GiGPO speed/validity run on verl-agent (lockstep loop)   #
# Isambard-AI: 1 node, 4x GH200 120GB, aarch64. Header copied from the        #
# working smoke job 5991353.                                                  #
#                                                                             #
# Usage:                                                                      #
#   MODE=sanity sbatch examples/gigpo_trainer/slurm_babyai_race.sbatch        #
#   MODE=timing sbatch examples/gigpo_trainer/slurm_babyai_race.sbatch        #
# =========================================================================== #
#SBATCH --job-name=race-stock
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# no --mem: a cgroup cap makes Ray's /dev/shm object store (3.5T shm here)
# blow the job's memory limit — the proven DICE jobs also set no --mem.
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6re/jdilkes.u6re/frameworks-spike/race-stock-%j.out
#SBATCH --error=/home/u6re/jdilkes.u6re/frameworks-spike/race-stock-%j.err

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

# Pre-started Ray head + raylet log capture (see port-arm sbatch for rationale:
# internal ray.init() raylet dies on these nodes; DICE jobs pre-start and connect).
export RAY_TMPDIR="$TMPDIR/ray"
mkdir -p "$RAY_TMPDIR"
SHM_AVAIL_KB=$(df --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
if [ -n "$SHM_AVAIL_KB" ] && [ "$SHM_AVAIL_KB" -gt 1048576 ]; then
    export RAY_OBJECT_STORE_MEMORY=$(( SHM_AVAIL_KB * 1024 * 80 / 100 ))
else
    export RAY_OBJECT_STORE_MEMORY=$((1 * 1024 * 1024 * 1024))
fi
export RAY_memory_monitor_refresh_ms=0
dump_raylet() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "=== RAYLET LOGS (exit $rc) ==="
        tail -60 "$RAY_TMPDIR"/session_latest/logs/raylet.err 2>/dev/null
        tail -30 "$RAY_TMPDIR"/session_latest/logs/raylet.out 2>/dev/null
        tail -30 "$RAY_TMPDIR"/session_latest/logs/gcs_server.err 2>/dev/null
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap dump_raylet EXIT
ray start --head --port=6379 --num-gpus=4 --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
    --object-store-memory="$RAY_OBJECT_STORE_MEMORY" --temp-dir="$RAY_TMPDIR" \
    --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:6379"

# ---- MATCHED CONTRACT (must mirror the port-arm sbatch) ------------------- #
# episodes/optimizer step: 128 (train_data_size=16 x group_size=8)
# env: BabyAI-GoToRedBallGrey-v0, 20 env-step cap, history window 20
# thinking OFF both arms; lr 1e-6; no KL; gamma 0.95; gigpo w=1.0 mean_norm
# Per-turn cap here: data.max_response_length (one turn = one sample).
# NOTE this arm trains per-STEP samples (up to 128 x 20 = 2560/optim step)
# vs the port arm's per-EPISODE samples (128) — a structural difference of
# the frameworks, deliberately not "matched away".
# --------------------------------------------------------------------------- #

if [ "$MODE" = "sanity" ]; then
    RUN_SCRIPT=examples/gigpo_trainer/run_babyai_qwen3_1.7b.sh
    STEPS=4
    VAL_BEFORE=True
else
    RUN_SCRIPT=examples/gigpo_trainer/run_babyai_qwen3_4b.sh
    STEPS=12
    VAL_BEFORE=False
fi

# First positional arg is the ENGINE ($1 in the run script) — overrides follow.
bash "$RUN_SCRIPT" vllm \
    trainer.total_training_steps=$STEPS \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=$VAL_BEFORE \
    trainer.experiment_name="race-stock-${MODE}-j${SLURM_JOB_ID}"

echo "RACE ${MODE} DONE"
# Post-run (login node):
#   python $HOME/frameworks-spike/verl-latest/recipe/gigpo/validity_check.py \
#       $HOME/frameworks-spike/race-stock-<job>.out [--compare <port .out>]
