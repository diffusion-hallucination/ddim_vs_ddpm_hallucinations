#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-gaussian25_h64_t64_sigma5}"
ARTIFACT_RUN_NAME="${ARTIFACT_RUN_NAME:-${RUN_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
VIS_NUM_SAMPLES="${VIS_NUM_SAMPLES:-${NUM_SAMPLES}}"
TIMESTEPS="${TIMESTEPS:-1000}"
DDIM_TIMESTEPS="${DDIM_TIMESTEPS:-50}"
DDIM_STEP_GRID="${DDIM_STEP_GRID:-50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000}"
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

COMMON_EVAL_ARGS=(
  "run_name=${RUN_NAME}"
  "artifact_run_name=${ARTIFACT_RUN_NAME}"
  "results_dir=${RESULTS_DIR}"
  "device=${DEVICE}"
  "timesteps=${TIMESTEPS}"
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

eval_cmd() {
  local exp_name="$1"
  local sampling_mode="$2"
  shift 2
  run_cmd "${PYTHON}" run_eval.py \
    "${COMMON_EVAL_ARGS[@]}" \
    "exp_name=${exp_name}" \
    "sampling_mode=${sampling_mode}" \
    "$@"
}

for sampler in ddim ddpm; do
  eval_cmd exp_a_two_mode_assumption "${sampler}"
  eval_cmd exp_b_convergence_to_nearby_line "${sampler}"
done

eval_cmd exp_e_ddim_ddpm_hall_rate ddim \
  "exp_e_tau_vals=${EXP_E_TAU_VALS}" \
  "exp_e_overlay_tau_targets=${EXP_E_OVERLAY_TAU_TARGETS}" \
  "exp_e_include_mixed_z=true"

eval_cmd visualize_samples ddpm "num_samples=${VIS_NUM_SAMPLES}"
for steps in ${DDIM_STEP_GRID}; do
  eval_cmd visualize_samples ddim \
    "num_samples=${VIS_NUM_SAMPLES}" \
    "ddim_timesteps=${steps}"
done

VIS_GRID="[${DDIM_STEP_GRID// /,}]"
run_cmd "${PYTHON}" run_visualization.py \
  "run_name=${ARTIFACT_RUN_NAME}" \
  "results_dir=${RESULTS_DIR}" \
  "mode=ddim_interpolation_rate_sweep" \
  "expected_ddim_steps_grid=${VIS_GRID}" \
  "plot_min_ddim_steps=200"
