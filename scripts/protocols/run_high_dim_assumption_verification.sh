#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DIMS="${DIMS:-2 4 8 32 64}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-dimensional_tau2_varpi}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
TIMESTEPS="${TIMESTEPS:-1000}"
DDIM_TIMESTEPS="${DDIM_TIMESTEPS:-50}"
SKIP_TYPE="${SKIP_TYPE:-quad}"
SHELL_OFFSET="${SHELL_OFFSET:-4.0}"
CHECKPOINT_POLL_SECONDS="${CHECKPOINT_POLL_SECONDS:-60}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

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

should_skip() {
  local artifact_path="$1"
  [[ "${SKIP_EXISTING}" == "true" && -f "${artifact_path}" ]]
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

solver_tag_for_sampler() {
  local sampler="$1"
  case "${sampler}" in
    ddim)
      echo "ddim_${SKIP_TYPE}_${DDIM_TIMESTEPS}"
      ;;
    ddpm)
      echo "ddpm_${TIMESTEPS}"
      ;;
    *)
      echo "Unsupported sampler: ${sampler}" >&2
      return 2
      ;;
  esac
}

dims_contains() {
  local needle="$1"
  local dim
  for dim in ${DIMS}; do
    [[ "${dim}" == "${needle}" ]] && return 0
  done
  return 1
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

wait_for_checkpoint() {
  local run_name="$1"
  local checkpoint_path="results/${run_name}/checkpoints/final_model.pth"
  if dry_run; then
    echo "[dry-run] would wait for ${checkpoint_path}"
    return
  fi
  until [[ -f "${checkpoint_path}" ]]; do
    echo "Waiting for checkpoint: ${checkpoint_path}"
    sleep "${CHECKPOINT_POLL_SECONDS}"
  done
}

for dim in ${DIMS}; do
  read -r hidden_dim time_dim num_blocks max_steps batch_size < <(arch_for_dim "${dim}")
  run_name="$(run_name_for_dim "${dim}")"
  wait_for_checkpoint "${run_name}"
  for sampler in ddim ddpm; do
    solver_tag="$(solver_tag_for_sampler "${sampler}")"
    exp_a_json="${RESULTS_DIR}/${ARTIFACT_ROOT}/dim_${dim}/two_mode_assumption/exp_a_tau_vs_kappa_${solver_tag}.json"
    exp_a_u_json="${RESULTS_DIR}/${ARTIFACT_ROOT}/dim_${dim}/two_mode_assumption/exp_a_u_vs_kappa_${solver_tag}.json"
    exp_b_json="${RESULTS_DIR}/${ARTIFACT_ROOT}/dim_${dim}/exp_convergence/exp_b_convergence_curve_${solver_tag}.json"

    common_args=(
      "run_name=${run_name}"
      "artifact_run_name=${ARTIFACT_ROOT}/dim_${dim}"
      "results_dir=${RESULTS_DIR}"
      "device=${DEVICE}"
      "timesteps=${TIMESTEPS}"
      "ddim_timesteps=${DDIM_TIMESTEPS}"
      "skip_type=${SKIP_TYPE}"
      "num_samples=${NUM_SAMPLES}"
      "invalid_sigma_multiple=5"
      "hall_radius_sigma_multiple=5"
      "sample_classification_mode=sigma_geometry"
      "sample_classification_scale_by_sqrt_varpi=false"
      "sample_classification_shell_offset=${SHELL_OFFSET}"
      "tau2_variance_scale_mode=multiply_by_varpi"
      "expected_hidden_dim=${hidden_dim}"
      "expected_time_dim=${time_dim}"
      "gaussian_protocol_tag=fixed"
      "sampling_mode=${sampler}"
    )

    if should_skip "${exp_a_json}" && should_skip "${exp_a_u_json}"; then
      echo "Skipping existing Exp A artifacts: ${exp_a_json}, ${exp_a_u_json}"
    else
      run_cmd "${PYTHON}" run_eval.py \
        "${common_args[@]}" \
        exp_name=exp_a_two_mode_assumption
    fi

    if should_skip "${exp_b_json}"; then
      echo "Skipping existing Exp B convergence artifact: ${exp_b_json}"
    else
      run_cmd "${PYTHON}" run_eval.py \
        "${common_args[@]}" \
        exp_name=exp_b_convergence_to_nearby_line
    fi
  done
done

if dims_contains 64; then
  wait_for_checkpoint "$(run_name_for_dim 64)"
fi

VIS_DIMS="[${DIMS// /,}]"
run_cmd "${PYTHON}" run_visualization.py \
  "mode=high_dim_assumption_summary" \
  "results_dir=${RESULTS_DIR}" \
  "high_dim_artifact_root=${ARTIFACT_ROOT}" \
  "high_dim_dims=${VIS_DIMS}"

run_cmd "${PYTHON}" run_visualization.py \
  "mode=high_dim_convergence_summary" \
  "results_dir=${RESULTS_DIR}" \
  "high_dim_artifact_root=${ARTIFACT_ROOT}" \
  "high_dim_dims=${VIS_DIMS}"
