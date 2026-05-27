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
TIMESTEPS="${TIMESTEPS:-1000}"
DDIM_TIMESTEPS="${DDIM_TIMESTEPS:-50}"
KAPPA_TARGET="${KAPPA_TARGET:-7.0}"
SHELL_OFFSET="${SHELL_OFFSET:-4.0}"
TAU3_TARGET="${TAU3_TARGET:-9}"
TAU3_INTERNAL_RAW_DDPM_STEPS="${TAU3_INTERNAL_RAW_DDPM_STEPS:-21}"
TAU3_THETA_FRACTION="${TAU3_THETA_FRACTION:-0.15}"
TAU3_CLOSED_FORM_NUM_A_PER_SIDE="${TAU3_CLOSED_FORM_NUM_A_PER_SIDE:-50}"
TAU3_EMPIRICAL_NUM_A_PER_SIDE="${TAU3_EMPIRICAL_NUM_A_PER_SIDE:-50}"
TAU3_EMPIRICAL_NUM_ROLLOUTS="${TAU3_EMPIRICAL_NUM_ROLLOUTS:-10000}"
DDIM_ETA_VALUES="${DDIM_ETA_VALUES:-[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}"
EXP_K_TAU_VALS="${EXP_K_TAU_VALS:-[9]}"
EXP_E_EXACT_TAU_VALS="${EXP_E_EXACT_TAU_VALS:-[3]}"
EXP_E_EXACT_TAU_INTERNAL_VALS_DDPM="${EXP_E_EXACT_TAU_INTERNAL_VALS_DDPM:-[3]}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-results/${RUN_NAME}/checkpoints/final_model.pth}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-logs/protocols}"

dry_run() {
  [[ "${DRY_RUN:-false}" == "true" ]]
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if ! dry_run; then
    "$@"
  fi
}

if dry_run; then
  echo "[dry-run] would wait for ${CHECKPOINT_PATH}"
else
  until [[ -f "${CHECKPOINT_PATH}" ]]; do
    echo "Waiting for checkpoint: ${CHECKPOINT_PATH}"
    sleep "${POLL_SECONDS}"
  done
fi

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
  "tau3_target=${TAU3_TARGET}"
  "tau3_internal_raw_ddpm_steps=${TAU3_INTERNAL_RAW_DDPM_STEPS}"
  "tau3_theta_fraction=${TAU3_THETA_FRACTION}"
  "tau3_closed_form_num_a_per_side=${TAU3_CLOSED_FORM_NUM_A_PER_SIDE}"
  "tau3_empirical_num_a_per_side=${TAU3_EMPIRICAL_NUM_A_PER_SIDE}"
  "tau3_empirical_num_rollouts=${TAU3_EMPIRICAL_NUM_ROLLOUTS}"
)

launch_bg() {
  local name="$1"
  shift
  if dry_run; then
    run_cmd "$@"
    return
  fi
  mkdir -p "${LOG_DIR}"
  local log_path="${LOG_DIR}/${name}.log"
  echo "Launching ${name}; log: ${log_path}"
  (
    print_cmd "$@"
    "$@"
  ) >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
}

PIDS=()
NAMES=()

cmd_i=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_i_midpoint_entry_verification"
  "sampling_mode=ddim"
  "tau3_target=11"
  "tau3_theta_fraction=0.35"
)
launch_bg exp_i_midpoint_entry "${cmd_i[@]}"

cmd_c=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_c_jacobian_at_midpoint"
  "sampling_mode=ddim"
)
launch_bg exp_c_jacobian "${cmd_c[@]}"

cmd_j=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_j_ddim_eta_hallucination_rate"
  "sampling_mode=ddim"
  "ddim_eta_values=${DDIM_ETA_VALUES}"
)
launch_bg exp_j_eta_rate "${cmd_j[@]}"

cmd_k=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_k_ddim_eta_hall_radius"
  "sampling_mode=ddim"
  "ddim_eta_values=${DDIM_ETA_VALUES}"
  "exp_e_tau_vals=${EXP_K_TAU_VALS}"
  "exp_e_overlay_tau_targets=${EXP_K_TAU_VALS}"
)
launch_bg exp_k_eta_radius "${cmd_k[@]}"

cmd_e_exact=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_e_ddim_ddpm_hall_rate"
  "sampling_mode=ddim"
  "use_exact_score=true"
  "exp_e_tau_vals=${EXP_E_EXACT_TAU_VALS}"
  "exp_e_overlay_tau_targets=${EXP_E_EXACT_TAU_VALS}"
  "exp_e_tau_internal_vals_ddpm=${EXP_E_EXACT_TAU_INTERNAL_VALS_DDPM}"
  "exp_e_include_mixed_z=false"
)
launch_bg exp_e_exactscore_radius "${cmd_e_exact[@]}"

cmd_l=(
  "${PYTHON}" run_eval.py
  "${COMMON_EVAL_ARGS[@]}"
  "exp_name=exp_l_ddpm_tau3_empirical_a_threshold"
  "sampling_mode=ddpm"
)
if dry_run; then
  run_cmd "${cmd_l[@]}"
else
  mkdir -p "${LOG_DIR}"
  echo "Launching exp_l_tau3; log: ${LOG_DIR}/exp_l_tau3.log"
  (
    print_cmd "${cmd_l[@]}"
    "${cmd_l[@]}"
  ) >"${LOG_DIR}/exp_l_tau3.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("exp_l_tau3")
fi

status=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "${NAMES[$idx]} failed; see ${LOG_DIR}/${NAMES[$idx]}.log" >&2
    status=1
  fi
done
exit "${status}"
