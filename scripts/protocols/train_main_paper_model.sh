#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-gaussian25_h64_t64_sigma5}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-0.0001}"
MIN_LR="${MIN_LR:-0.00001}"
LR_SCHEDULE="${LR_SCHEDULE:-linear}"
FORCE_RETRAIN="${FORCE_RETRAIN:-false}"

dry_run() {
  [[ "${DRY_RUN:-false}" == "true" ]]
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ! dry_run; then
    "$@"
  fi
}

run_cmd "${PYTHON}" run_train.py \
  dataset=gaussian25 \
  train_dataset_type=gaussian \
  model_type=mlp \
  hidden_dim=64 \
  time_dim=64 \
  num_blocks=3 \
  batch_size="${BATCH_SIZE}" \
  lr="${LR}" \
  optimizer.lr="${LR}" \
  train.max_steps="${MAX_STEPS}" \
  train.min_lr="${MIN_LR}" \
  train.lr_schedule="${LR_SCHEDULE}" \
  train.log_every_steps=100 \
  train.sample_every_steps=10000 \
  train.checkpoint_every_steps=10000 \
  train.sample_num_samples=2000 \
  train.ema_decay=0.0 \
  mlp.conditioning_type=timestep \
  mlp.conditioning_strategy=input_add \
  mlp.input_encoding=raw \
  continue_training_path=null \
  device="${DEVICE}" \
  use_wandb=false \
  log.use_wandb=false \
  log.run_name="${RUN_NAME}" \
  force_retrain="${FORCE_RETRAIN}"
