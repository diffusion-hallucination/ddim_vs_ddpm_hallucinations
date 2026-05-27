#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
IMAGE_RUN_NAME="${IMAGE_RUN_NAME:-triangle_only_image_model}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
IMG_SIZE="${IMG_SIZE:-64}"
CKPT_NAME="${CKPT_NAME:-final_model.pth}"
DDIM_STEPS_LIST="${DDIM_STEPS_LIST:-50}"
DDIM_SEEDS="${DDIM_SEEDS:-0}"
DDPM_SEEDS="${DDPM_SEEDS:-0}"

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

for seed in ${DDPM_SEEDS}; do
  run_cmd "${PYTHON}" run_eval.py \
    exp_name=exp_d_image_diffusion \
    "run_name=${IMAGE_RUN_NAME}" \
    "results_dir=${RESULTS_DIR}" \
    "device=${DEVICE}" \
    "num_samples=${NUM_SAMPLES}" \
    "img_size=${IMG_SIZE}" \
    "ckpt_name=${CKPT_NAME}" \
    sampling_mode=ddpm \
    "image_seed=${seed}"
done

for seed in ${DDIM_SEEDS}; do
  for steps in ${DDIM_STEPS_LIST}; do
    run_cmd "${PYTHON}" run_eval.py \
      exp_name=exp_d_image_diffusion \
      "run_name=${IMAGE_RUN_NAME}" \
      "results_dir=${RESULTS_DIR}" \
      "device=${DEVICE}" \
      "num_samples=${NUM_SAMPLES}" \
      "img_size=${IMG_SIZE}" \
      "ckpt_name=${CKPT_NAME}" \
      sampling_mode=ddim \
      skip_type=quad \
      "ddim_timesteps=${steps}" \
      "image_seed=${seed}"
  done
done
