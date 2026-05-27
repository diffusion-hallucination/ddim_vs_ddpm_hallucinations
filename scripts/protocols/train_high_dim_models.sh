#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DIMS="${DIMS:-2 4 8 32 64}"
LR="${LR:-0.0001}"
FOURIER_FEATURES="${FOURIER_FEATURES:-256}"
FOURIER_SCALE="${FOURIER_SCALE:-0.5}"
FOURIER_SEED="${FOURIER_SEED:-42}"
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

run_name_for_dim() {
  local dim="$1"
  case "${dim}" in
    2|4|8)
      echo "gaussian_5d_dim_${dim}_mlp_lambda_fourier_w512_c512_b8_linear_steps_100000_norm2sqrtd"
      ;;
    32|64)
      echo "gaussian_5d_dim_${dim}_mlp_lambda_fourier_w768_c512_b8_linear_steps_200000_norm2sqrtd"
      ;;
    *)
      echo "Unsupported high-dimensional setting: ${dim}" >&2
      return 2
      ;;
  esac
}

arch_for_dim() {
  local dim="$1"
  case "${dim}" in
    2|4|8)
      echo "512 512 8 100000 8192"
      ;;
    32|64)
      echo "768 512 8 200000 4096"
      ;;
    *)
      echo "Unsupported high-dimensional setting: ${dim}" >&2
      return 2
      ;;
  esac
}

for dim in ${DIMS}; do
  read -r hidden_dim time_dim num_blocks max_steps batch_size < <(arch_for_dim "${dim}")
  run_name="$(run_name_for_dim "${dim}")"
  run_cmd "${PYTHON}" run_train.py \
    dataset=gaussian5d \
    "dataset.num_dims=${dim}" \
    "dataset.save_folder=./results" \
    "dataset.normalization_mode=fixed_2sqrtd" \
    "dataset.resample_each_epoch=true" \
    train_dataset_type=gaussian \
    model_type=mlp \
    "hidden_dim=${hidden_dim}" \
    "time_dim=${time_dim}" \
    "num_blocks=${num_blocks}" \
    "batch_size=${batch_size}" \
    "lr=${LR}" \
    "optimizer.lr=${LR}" \
    "train.max_steps=${max_steps}" \
    train.log_every_steps=100 \
    train.sample_every_steps=10000 \
    train.checkpoint_every_steps=10000 \
    train.sample_num_samples=2000 \
    train.lambda_bins=20 \
    mlp.conditioning_type=lambda_lookup \
    mlp.conditioning_strategy=per_block \
    mlp.input_encoding=fourier \
    "mlp.fourier_features=${FOURIER_FEATURES}" \
    "mlp.fourier_scale=${FOURIER_SCALE}" \
    "mlp.fourier_seed=${FOURIER_SEED}" \
    continue_training_path=null \
    device="${DEVICE}" \
    use_wandb=false \
    log.use_wandb=false \
    "log.run_name=${run_name}" \
    "force_retrain=${FORCE_RETRAIN}"
done
