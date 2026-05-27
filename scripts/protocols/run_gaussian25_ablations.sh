#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-gaussian25_ablations}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-gaussian25_ablation_h64}"

BASE_TIMESTEPS="${BASE_TIMESTEPS:-1000}"
BASE_SIGMA="${BASE_SIGMA:-0.02}"
BASE_ELL="${BASE_ELL:-2.0}"
T_VALUES="${T_VALUES-500 2000 3000}"
SIGMA_VALUES="${SIGMA_VALUES-0.01 0.05 0.10}"
ELL_VALUES="${ELL_VALUES-1.0 1.5 3.0}"

TRAIN="${TRAIN:-true}"
EVAL="${EVAL:-true}"
FORCE_RETRAIN="${FORCE_RETRAIN:-false}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-0.0001}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-100000}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
DDIM_TIMESTEPS="${DDIM_TIMESTEPS:-50}"
KAPPA_TARGET="${KAPPA_TARGET:-7.0}"
SHELL_OFFSET="${SHELL_OFFSET:-4.0}"
EXP_E_TAU_VALS="${EXP_E_TAU_VALS:-[5,9,15,30,40]}"
EXP_E_OVERLAY_TAU_TARGETS="${EXP_E_OVERLAY_TAU_TARGETS:-[9]}"

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

slug_number() {
  printf '%s' "$1" | sed -e 's/-/m/g' -e 's/\./p/g'
}

maybe_train_variant() {
  local run_name="$1"
  local timesteps="$2"
  local sigma="$3"
  local ell="$4"

  [[ "${TRAIN}" == "true" ]] || return 0

  run_cmd "${PYTHON}" run_train.py \
    dataset=gaussian25 \
    train_dataset_type=gaussian \
    model_type=mlp \
    hidden_dim=64 \
    time_dim=64 \
    num_blocks=3 \
    "timesteps=${timesteps}" \
    "dataset.global_sigma=${sigma}" \
    "dataset.stdev=${sigma}" \
    "dataset.mode_spacing=${ell}" \
    "dataset.num_samples=${TRAIN_SAMPLES}" \
    "dataset.num_test_samples=${TRAIN_SAMPLES}" \
    "batch_size=${BATCH_SIZE}" \
    "lr=${LR}" \
    "optimizer.lr=${LR}" \
    "train.max_steps=${MAX_STEPS}" \
    train.log_every_steps=100 \
    train.sample_every_steps=10000 \
    train.checkpoint_every_steps=10000 \
    train.sample_num_samples=2000 \
    train.ema_decay=0.0 \
    mlp.conditioning_type=timestep \
    mlp.conditioning_strategy=input_add \
    mlp.input_encoding=raw \
    "device=${DEVICE}" \
    use_wandb=false \
    log.use_wandb=false \
    "log.run_name=${run_name}" \
    "force_retrain=${FORCE_RETRAIN}"
}

eval_variant() {
  local run_name="$1"
  local timesteps="$2"

  [[ "${EVAL}" == "true" ]] || return 0

  local artifact_run_name="${ARTIFACT_ROOT}/${run_name}"
  local common_eval_args=(
    "run_name=${run_name}"
    "artifact_run_name=${artifact_run_name}"
    "results_dir=${RESULTS_DIR}"
    "device=${DEVICE}"
    "timesteps=${timesteps}"
    "ddim_timesteps=${DDIM_TIMESTEPS}"
    "skip_type=quad"
    "num_samples=${NUM_SAMPLES}"
    "invalid_sigma_multiple=5"
    "hall_radius_sigma_multiple=5"
    "sample_classification_mode=sigma_geometry"
    "sample_classification_scale_by_sqrt_varpi=false"
    "sample_classification_shell_offset=${SHELL_OFFSET}"
    "tau2_variance_scale_mode=multiply_by_varpi"
    "expected_hidden_dim=64"
    "expected_time_dim=64"
    "expected_dataset_kind=gaussian_mixture_2d"
    "gaussian_protocol_tag=fixed"
  )

  for sampler in ddim ddpm; do
    run_cmd "${PYTHON}" run_eval.py \
      "${common_eval_args[@]}" \
      exp_name=exp_a_two_mode_assumption \
      "sampling_mode=${sampler}" \
      "kappa_target=${KAPPA_TARGET}"
    run_cmd "${PYTHON}" run_eval.py \
      "${common_eval_args[@]}" \
      exp_name=exp_b_convergence_to_nearby_line \
      "sampling_mode=${sampler}" \
      "kappa_target=${KAPPA_TARGET}"
  done

  run_cmd "${PYTHON}" run_eval.py \
    "${common_eval_args[@]}" \
    exp_name=exp_c_jacobian_at_midpoint \
    sampling_mode=ddim \
    "kappa_target=${KAPPA_TARGET}"

  run_cmd "${PYTHON}" run_eval.py \
    "${common_eval_args[@]}" \
    exp_name=exp_e_ddim_ddpm_hall_rate \
    sampling_mode=ddim \
    "exp_e_tau_vals=${EXP_E_TAU_VALS}" \
    "exp_e_overlay_tau_targets=${EXP_E_OVERLAY_TAU_TARGETS}"
}

run_variant() {
  local variant_id="$1"
  local timesteps="$2"
  local sigma="$3"
  local ell="$4"
  local run_name="${RUN_NAME_PREFIX}_${variant_id}"

  echo
  echo "== ${variant_id}: T=${timesteps}, sigma=${sigma}, ell=${ell} =="
  maybe_train_variant "${run_name}" "${timesteps}" "${sigma}" "${ell}"
  eval_variant "${run_name}" "${timesteps}"
}

for t in ${T_VALUES}; do
  run_variant "t$(slug_number "${t}")" "${t}" "${BASE_SIGMA}" "${BASE_ELL}"
done

for sigma in ${SIGMA_VALUES}; do
  run_variant "sigma$(slug_number "${sigma}")" "${BASE_TIMESTEPS}" "${sigma}" "${BASE_ELL}"
done

for ell in ${ELL_VALUES}; do
  run_variant "ell$(slug_number "${ell}")" "${BASE_TIMESTEPS}" "${BASE_SIGMA}" "${ell}"
done
