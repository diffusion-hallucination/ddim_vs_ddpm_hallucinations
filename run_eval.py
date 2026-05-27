import glob
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd())

from common.diffusion import Diffusion
from common.gaussian_exact_score import ExactGaussianMixtureEpsModel, NoisyExactGaussianMixtureEpsModel
from experiments.exp_a_two_mode import (
    compute_tau_streaming,
    plot_tau_1_tau_2_vs_kappa,
    sample_two_mode_gaussian,
    compute_interpolated_distances,
)
from experiments.exp_b_convergence_to_line import convergence_to_nearby_line_streaming
from experiments.exp_c_compute_jacobian import (
    sample_two_mode_gaussian_with_eig_values,
    visualize_eig_values_grouped_mean_std,
)
from experiments.exp_d_image_diffusion import (
    load_image_checkpoint,
    sample_image_diffusion,
    save_image_grid,
    compute_hallucination_rate
)
from experiments.exp_e_hall_with_radius import (
    compute_hallucination_with_radius,
    plot_hallucination_results,
    plot_hallucination_overlay_single_tau,
)
from experiments.exp_i_midpoint_entry_verification import (
    run_midpoint_entry_verification,
    run_midpoint_probability_table,
)
from experiments.exp_j_ddim_eta_hallucination_rate import run_ddim_eta_hallucination_rate_sweep
from experiments.exp_k_ddim_eta_hall_radius import run_ddim_eta_hall_radius_sweep
from experiments.exp_l_tau3_bisector_closed_form import run_tau3_empirical_a_threshold_companion
from common.artifact_io import write_csv, write_json
from common.experiment_artifacts import (
    save_exp_a_solver_artifacts,
    save_exp_a_u_solver_artifacts,
    save_exp_b_solver_artifacts,
    save_exp_c_solver_artifacts,
    save_exp_e_solver_artifacts,
)
from common.reverse_solvers import solver_display_name, solver_tag_from_cfg
from common.gaussian_mixture_2d import (
    build_mixture_spec_from_dataset_cfg,
    classify_samples_by_geometry,
    geometry_radius_scale,
    load_mixture_spec_for_run,
    mode_radii_from_sigmas,
    pair_radii_from_sigmas,
    sample_classification_config,
)
from common.score_training import maybe_wrap_model_for_eps
import copy
from common.utils import load_checkpoint
from experiments.plt_utils import visualize_gaussian_with_modes

def set_seed(seed):
    # seed python, numpy, and torch before evaluation.
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_mixture_context_for_run(cfg, eval_cfg, dataset_name=None):
    # load the Gaussian mixture context
    def with_arrays(spec: dict, source: str) -> dict:
        return {
            **spec,
            "normalized_means": np.asarray(spec["normalized_means"], dtype=np.float64),
            "normalized_sigmas": np.asarray(spec["normalized_sigmas"], dtype=np.float64),
            "raw_means": np.asarray(spec["raw_means"], dtype=np.float64),
            "raw_sigmas": np.asarray(spec["raw_sigmas"], dtype=np.float64),
            "mode_weights": np.asarray(spec["mode_weights"], dtype=np.float64),
            "source": source,
        }

    if dataset_name is None:
        dataset_name = cfg.dataset.name
    run_dir = os.path.join("results", eval_cfg.run_name)
    spec = load_mixture_spec_for_run(run_dir)
    if spec is not None:
        return with_arrays(spec, "mixture_spec")
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is not None and str(dataset_cfg.get("kind", "")).lower() == "gaussian_mixture_2d":
        return with_arrays(build_mixture_spec_from_dataset_cfg(dataset_cfg), "dataset_cfg")
    log_cfg = getattr(cfg, "log", None)
    run_name = "" if log_cfg is None else getattr(log_cfg, "run_name", "")
    raise FileNotFoundError(
        f"No mixture_spec.json found for run {run_name!r}, "
        f"and dataset {dataset_name!r} is not a supported current Gaussian mixture config."
    )


def load_eval_mixture_arrays(cfg, eval_cfg, dataset_name=None):
    # load the mixture arrays
    ctx = load_mixture_context_for_run(cfg, eval_cfg, dataset_name)
    modes = np.asarray(ctx["normalized_means"], dtype=np.float64)
    mode_sigmas = np.asarray(ctx.get("normalized_sigmas"), dtype=np.float64)
    mode_weights = np.asarray(ctx.get("mode_weights"), dtype=np.float64)
    std_dev = float(ctx.get("global_sigma_normalized", float(cfg.dataset.stdev)))
    return ctx, modes, mode_sigmas, mode_weights, std_dev


def optional_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def bool_cfg(value, default: bool = False) -> bool:
    if value in (None, "", "None", "null"):
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def artifact_run_name(eval_cfg) -> str:
    return optional_text(getattr(eval_cfg, "artifact_run_name", None)) or str(eval_cfg.run_name)


def artifact_subdir(base_name: str, eval_cfg) -> str:
    suffix = "_exactscore" if bool(getattr(eval_cfg, "use_exact_score", False)) else ""
    return f"{base_name}{suffix}"


def float_list_cfg(eval_cfg, field_name: str):
    # parse a float list override from the eval config.
    value = getattr(eval_cfg, field_name, None)
    if optional_text(value) is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner == "":
            return []
        return [float(v.strip()) for v in inner.split(",") if v.strip()]
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def ddim_tau_labels_to_raw_steps(diffusion, tau_vals, *, ddim_steps: int, skip_type: str) -> list[int]:
    # Convert paper-facing DDIM-step labels to the raw DDPM timestep cutoff used by the buffers.
    seq = diffusion.make_ddim_seq(str(skip_type), int(ddim_steps))
    if not seq:
        raise ValueError("DDIM sequence is empty.")
    raw_steps = []
    for tau in tau_vals:
        tau_int = int(tau)
        if tau_int < 1:
            raise ValueError(f"DDIM tau labels must be >= 1, got {tau_int}.")
        idx = min(tau_int - 1, len(seq) - 1)
        raw_steps.append(max(int(seq[idx]), 1))
    return raw_steps


def tau2_variance_scale(eval_cfg, num_dims: int) -> tuple[str, float]:
    mode = str(getattr(eval_cfg, "tau2_variance_scale_mode", "auto")).strip().lower()
    if mode not in {"auto", "none", "multiply_by_varpi"}:
        raise ValueError(
            "tau2_variance_scale_mode must be one of {'auto', 'none', 'multiply_by_varpi'}"
        )
    if mode == "auto":
        protocol_tag = optional_text(getattr(eval_cfg, "gaussian_protocol_tag", None))
        mode = "multiply_by_varpi" if protocol_tag == "fixed" else "none"
    factor = float(num_dims) if mode == "multiply_by_varpi" else 1.0
    return mode, factor


def sample_classification_cfg(eval_cfg, mode_sigmas, std_dev: float, num_dims: int) -> dict:
    # build the geometry classification config for sampled points.
    sigmas = (
        np.full((1,), float(std_dev), dtype=np.float64)
        if mode_sigmas is None
        else np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    )

    classification_mode = str(getattr(eval_cfg, "sample_classification_mode", "sigma_geometry")).strip().lower()
    if classification_mode != "sigma_geometry":
        raise ValueError("sample_classification_mode must be 'sigma_geometry'")

    raw_scale = getattr(eval_cfg, "sample_classification_scale_by_sqrt_varpi", False)
    if isinstance(raw_scale, bool):
        requested_scale = raw_scale
    else:
        scale_text = str(raw_scale).strip().lower()
        if scale_text in {"1", "true", "yes", "y", "on"}:
            requested_scale = True
        elif scale_text in {"0", "false", "no", "n", "off", "", "none", "null"}:
            requested_scale = False
        else:
            raise ValueError(
                "sample_classification_scale_by_sqrt_varpi must be parseable as a boolean"
            )

    shell_offset_value = getattr(eval_cfg, "sample_classification_shell_offset", 4.0)
    shell_offset = None if optional_text(shell_offset_value) is None else float(shell_offset_value)
    shell_dim_threshold = int(getattr(eval_cfg, "sample_classification_shell_dimension_threshold", 10))

    # Once the shell-aware rule sigma (sqrt(varpi) + C) activates, do not apply a second sqrt(varpi) multiplier.
    if shell_offset is not None and int(num_dims) > shell_dim_threshold:
        effective_scale = False
    else:
        scale = geometry_radius_scale(int(num_dims), scale_by_sqrt_varpi=bool(requested_scale))
        effective_scale = float(scale) != 1.0

    return sample_classification_config(
        sigmas,
        classification_mode=classification_mode,
        invalid_sigma_multiple=float(getattr(eval_cfg, "invalid_sigma_multiple", 5.0)),
        num_dims=int(num_dims),
        scale_by_sqrt_varpi=effective_scale,
        shell_offset=shell_offset,
        shell_dimension_threshold=shell_dim_threshold,
    )


def validate_gaussian_eval_provenance(eval_cfg, cfg, ckpt_path):
    # enforce that the loaded gaussian checkpoint matches the intended source.
    dataset = getattr(cfg, "dataset", None)
    dataset_name = ""
    if dataset is not None:
        dataset_name = getattr(dataset, "name", None)
        if dataset_name is None and hasattr(dataset, "get"):
            dataset_name = dataset.get("name", "")
    if not str(dataset_name or "").strip().lower().startswith("gaussian"):
        return

    def expected_int(field_name: str):
        value = getattr(eval_cfg, field_name, None)
        if optional_text(value) is None:
            return None
        return int(value)

    expected_hidden_dim = expected_int("expected_hidden_dim")
    if expected_hidden_dim is not None:
        actual_hidden_dim = int(getattr(cfg, "hidden_dim"))
        if actual_hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"Gaussian eval provenance mismatch for run_name={eval_cfg.run_name!r}: "
                f"expected hidden_dim={expected_hidden_dim}, got hidden_dim={actual_hidden_dim} "
                f"from checkpoint/config {ckpt_path!r}."
            )

    expected_time_dim = expected_int("expected_time_dim")
    if expected_time_dim is not None:
        actual_time_dim = int(getattr(cfg, "time_dim"))
        if actual_time_dim != expected_time_dim:
            raise ValueError(
                f"Gaussian eval provenance mismatch for run_name={eval_cfg.run_name!r}: "
                f"expected time_dim={expected_time_dim}, got time_dim={actual_time_dim} "
                f"from checkpoint/config {ckpt_path!r}."
            )

    expected_dataset_kind = optional_text(getattr(eval_cfg, "expected_dataset_kind", None))
    if expected_dataset_kind is not None:
        actual_dataset_kind = ""
        if dataset is not None:
            actual_dataset_kind = getattr(dataset, "kind", None)
            if actual_dataset_kind is None and hasattr(dataset, "get"):
                actual_dataset_kind = dataset.get("kind", "")
        actual_dataset_kind = str(actual_dataset_kind or "").strip()
        if actual_dataset_kind != expected_dataset_kind:
            raise ValueError(
                f"Gaussian eval provenance mismatch for run_name={eval_cfg.run_name!r}: "
                f"expected dataset.kind={expected_dataset_kind!r}, got {actual_dataset_kind!r} "
                f"from checkpoint/config {ckpt_path!r}."
            )

    protocol_tag = optional_text(getattr(eval_cfg, "gaussian_protocol_tag", None))
    if protocol_tag is not None:
        print(f"[gaussian_protocol] {protocol_tag}")


def build_exact_eval_cfg(eval_cfg):
    # build a lightweight cfg for exact-score eval without a checkpoint.
    if not (hasattr(eval_cfg, "dataset") and getattr(eval_cfg, "dataset", None) is not None):
        raise ValueError(
            "Exact-score eval without a checkpoint requires a dataset config. "
            "Pass +dataset=<name> and the relevant dataset overrides."
        )
    cfg_dict = {
        'dataset': OmegaConf.to_container(eval_cfg.dataset, resolve=True),
        'device': getattr(eval_cfg, 'device', 'cpu'),
        'timesteps': int(getattr(eval_cfg, 'timesteps', 1000)),
        'beta_start': float(getattr(eval_cfg, 'beta_start', 1e-4)),
        'beta_end': float(getattr(eval_cfg, 'beta_end', 2e-2)),
        'model_type': str(getattr(eval_cfg, 'model_type', 'mlp')),
        'hidden_dim': int(getattr(eval_cfg, 'hidden_dim', 128)),
        'time_dim': int(getattr(eval_cfg, 'time_dim', 128)),
        'num_blocks': int(getattr(eval_cfg, 'num_blocks', 3)),
    }
    return OmegaConf.create(cfg_dict)


def build_exact_eval_model(diffusion, cfg, eval_cfg):
    # construct the exact-score model for the mixture.
    mixture_ctx = load_mixture_context_for_run(cfg, eval_cfg, cfg.dataset.name)
    score_noise_std = float(getattr(eval_cfg, "exact_score_noise_std", 0.0))
    exact_cls = NoisyExactGaussianMixtureEpsModel if score_noise_std > 0.0 else ExactGaussianMixtureEpsModel
    exact_kwargs = {}
    if score_noise_std > 0.0:
        exact_kwargs = {
            "score_noise_std": score_noise_std,
            "score_noise_seed": int(getattr(eval_cfg, "exact_score_noise_seed", 42)),
        }
    exact_model = exact_cls(
        diffusion=diffusion,
        gaussian_modes=np.asarray(mixture_ctx['normalized_means'], dtype=np.float64),
        std_dev=float(mixture_ctx.get('global_sigma_normalized', float(cfg.dataset.stdev))),
        mode_weights=np.asarray(mixture_ctx.get('mode_weights'), dtype=np.float64),
        mode_sigmas=np.asarray(mixture_ctx.get('normalized_sigmas'), dtype=np.float64),
        **exact_kwargs,
    )
    exact_model = exact_model.to(device=eval_cfg.device)
    exact_model.eval()
    return exact_model


def existing_eval_checkpoint_path(eval_cfg):
    # search for an existing checkpoint path for this eval run.
    if bool(getattr(eval_cfg, 'use_exact_score', False)):
        geometry_source = str(getattr(eval_cfg, "exact_score_geometry_source", "checkpoint_or_dataset")).strip().lower()
        if geometry_source not in {"checkpoint_or_dataset", "dataset_only"}:
            raise ValueError(
                "exact_score_geometry_source must be one of "
                "{'checkpoint_or_dataset', 'dataset_only'}"
            )
        if geometry_source == "dataset_only":
            return None

    run_root = os.path.join('results', eval_cfg.run_name)
    ckpt_name = getattr(eval_cfg, 'ckpt_name', 'final_model.pth')
    candidates = [
        os.path.join(run_root, 'checkpoints', ckpt_name),
        os.path.join(run_root, ckpt_name),
    ]

    explicit = getattr(eval_cfg, 'train_model_load_path', '')
    if explicit:
        if os.path.isabs(explicit):
            candidates.insert(0, explicit)
        else:
            candidates = [
                os.path.join('results', explicit),
                os.path.join('results', explicit, ckpt_name),
                os.path.join('results', explicit, 'checkpoints', ckpt_name),
            ] + candidates

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def prepare_eval_model(model, diffusion, cfg, eval_cfg):
    # wrap the checkpoint model or replace it with the exact-score model.
    if not bool(getattr(eval_cfg, "use_exact_score", False)):
        wrapped = maybe_wrap_model_for_eps(model, diffusion, cfg)
        wrapped = wrapped.to(device=eval_cfg.device)
        wrapped.eval()
        return wrapped
    if isinstance(model, ExactGaussianMixtureEpsModel):
        model = model.to(device=eval_cfg.device)
        model.eval()
        return model
    return build_exact_eval_model(diffusion, cfg, eval_cfg)


def resolve_eval_checkpoint_path(eval_cfg):
    # build the expected checkpoint path even when nothing exists yet.
    existing = existing_eval_checkpoint_path(eval_cfg)
    if existing is not None:
        return existing

    run_root = os.path.join('results', eval_cfg.run_name)
    ckpt_name = getattr(eval_cfg, 'ckpt_name', 'final_model.pth')
    explicit = getattr(eval_cfg, 'train_model_load_path', '')
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join('results', explicit, ckpt_name)
    return os.path.join(run_root, 'checkpoints', ckpt_name)


def initialize_primitives(eval_cfg):
    # load the checkpoint, diffusion object, and output folder for an eval run.
    ckpt_path = existing_eval_checkpoint_path(eval_cfg)

    if ckpt_path is not None:
        ckpt_dict = load_checkpoint(ckpt_path, device=eval_cfg.device)
        cfg = OmegaConf.create(ckpt_dict['args'])
        diffusion = ckpt_dict["diffusion"]
    elif bool(getattr(eval_cfg, 'use_exact_score', False)):
        diffusion = Diffusion(args=eval_cfg, device=eval_cfg.device)
        cfg = build_exact_eval_cfg(eval_cfg)
        ckpt_dict = {
            'model': build_exact_eval_model(diffusion, cfg, eval_cfg),
            'args': OmegaConf.to_container(cfg, resolve=True),
        }
    else:
        missing = resolve_eval_checkpoint_path(eval_cfg)
        raise FileNotFoundError(
            f"No checkpoint found for run_name={eval_cfg.run_name!r}. Expected something like {missing!r}."
        )

    validate_gaussian_eval_provenance(eval_cfg, cfg, ckpt_path)

    artifact_name = artifact_run_name(eval_cfg)
    if eval_cfg.remote_exec: 
        fig_save_folder = os.path.join('../', artifact_name)
    else: 
        fig_save_folder = os.path.join(eval_cfg.results_dir, artifact_name)

    os.makedirs(fig_save_folder, exist_ok=True)    
    return ckpt_dict, diffusion, cfg, fig_save_folder


def initialize_primitives_for_source_run(
    eval_cfg,
    *,
    run_name: str,
    expected_hidden_dim=None,
    expected_time_dim=None,
    expected_dataset_kind=None,
):
    # reinitialize eval primitives for an alternate source checkpoint.
    alt_cfg = copy.deepcopy(eval_cfg)
    alt_cfg.run_name = str(run_name)
    if expected_hidden_dim is not None:
        alt_cfg.expected_hidden_dim = int(expected_hidden_dim)
    if expected_time_dim is not None:
        alt_cfg.expected_time_dim = int(expected_time_dim)
    if expected_dataset_kind is not None:
        alt_cfg.expected_dataset_kind = str(expected_dataset_kind)
    return initialize_primitives(alt_cfg)

def remove_stale_pngs(folder: str, pattern: str) -> None:
    # remove stale pngs before regenerating a figure set.
    for p in glob.glob(os.path.join(folder, pattern)):
        try:
            os.remove(p)
        except OSError:
            pass

def save_valid_only_gaussian_samples(
    final_samples,
    gaussian_modes,
    dataset_name: str,
    fig_save_folder: str,
    fig_name: str,
    invalid_sigma_multiple: float,
    std_dev: float,
    mode_sigmas=None,
):
    # save gaussian samples with invalid points marked through the geometry rule.
    modes = np.asarray(gaussian_modes, dtype=np.float64)
    mode_sigmas_arr = np.full((modes.shape[0],), float(std_dev), dtype=np.float64) if mode_sigmas is None else np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    sigma_multiple = float(invalid_sigma_multiple)
    mode_radii = mode_radii_from_sigmas(mode_sigmas_arr, sigma_multiple=sigma_multiple)
    pair_radii = pair_radii_from_sigmas(mode_sigmas_arr, sigma_multiple=sigma_multiple)
    geom = classify_samples_by_geometry(
        final_samples,
        modes,
        threshold=None,
        mode_radii=mode_radii,
        pair_radii=pair_radii,
    )
    visualize_gaussian_with_modes(
        final_samples,
        invalid_mask=np.asarray(geom["invalid_mask"], dtype=bool),
        true_modes=modes,
        dataset_name=dataset_name,
        fig_save_folder=fig_save_folder,
        fig_name=fig_name,
    )
    return geom


def run_exp_j_ddim_eta_hallucination_rate(eval_cfg):
    # run exp j by sweeping \eta and measuring final hallucination rates.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    eta_folder = os.path.join(fig_save_folder, artifact_subdir("hallucinations_eta", eval_cfg))
    os.makedirs(eta_folder, exist_ok=True)

    num_dims = int(cfg["dataset"].get("num_dims", 2))
    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    classification_cfg = sample_classification_cfg(eval_cfg, mode_sigmas, std_dev, num_dims)

    eta_values = float_list_cfg(eval_cfg, "ddim_eta_values")
    if eta_values is None:
        eta_values = [float(v) for v in np.linspace(0.0, 1.0, 11)]

    ddpm_baseline_run_name = optional_text(getattr(eval_cfg, "ddpm_baseline_run_name", None))
    ddpm_baseline_expected_hidden_dim = optional_text(getattr(eval_cfg, "ddpm_baseline_expected_hidden_dim", None))
    ddpm_baseline_expected_time_dim = optional_text(getattr(eval_cfg, "ddpm_baseline_expected_time_dim", None))
    ddpm_baseline_expected_dataset_kind = optional_text(getattr(eval_cfg, "ddpm_baseline_expected_dataset_kind", None))

    ddpm_baseline_model = model_for_eval
    ddpm_baseline_diffusion = diffusion
    ddpm_baseline_source_run = str(eval_cfg.run_name)

    if ddpm_baseline_run_name is not None and ddpm_baseline_run_name != str(eval_cfg.run_name):
        ddpm_ckpt_dict, ddpm_diffusion, ddpm_cfg, _ = initialize_primitives_for_source_run(
            eval_cfg,
            run_name=ddpm_baseline_run_name,
            expected_hidden_dim=(None if ddpm_baseline_expected_hidden_dim is None else int(ddpm_baseline_expected_hidden_dim)),
            expected_time_dim=(None if ddpm_baseline_expected_time_dim is None else int(ddpm_baseline_expected_time_dim)),
            expected_dataset_kind=ddpm_baseline_expected_dataset_kind,
        )
        ddpm_baseline_model = prepare_eval_model(ddpm_ckpt_dict["model"], ddpm_diffusion, ddpm_cfg, eval_cfg)
        ddpm_baseline_diffusion = ddpm_diffusion
        ddpm_baseline_source_run = str(ddpm_baseline_run_name)

    run_ddim_eta_hallucination_rate_sweep(
        dataset_name=cfg.dataset.name,
        artifact_run_name=artifact_run_name(eval_cfg),
        protocol_tag=optional_text(getattr(eval_cfg, "gaussian_protocol_tag", None)),
        source_run_name=str(eval_cfg.run_name),
        diffusion=diffusion,
        model=model_for_eval,
        gaussian_modes=gaussian_modes,
        classification_cfg=classification_cfg,
        num_samples=int(getattr(eval_cfg, "num_samples", 100000)),
        timesteps=int(eval_cfg.timesteps),
        ddim_steps=int(eval_cfg.ddim_timesteps),
        skip_type=str(eval_cfg.skip_type),
        eta_values=eta_values,
        save_folder=eta_folder,
        device=str(eval_cfg.device),
        num_dims=num_dims,
        ddpm_baseline_model=ddpm_baseline_model,
        ddpm_baseline_diffusion=ddpm_baseline_diffusion,
        ddpm_baseline_run_name=ddpm_baseline_source_run,
        chunk_size=int(getattr(eval_cfg, "chunk_size", 25000)),
    )


def run_exp_k_ddim_eta_hall_radius(eval_cfg):
    # run exp k by sweeping \eta over the exp e radius experiment.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    if str(getattr(eval_cfg, "sampling_mode", "ddim")).lower() != "ddim":
        raise ValueError("exp_k_ddim_eta_hall_radius only supports sampling_mode='ddim'.")

    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    dataset_name = cfg["dataset"]["name"]

    eta_values = float_list_cfg(eval_cfg, "ddim_eta_values")
    if eta_values is None:
        eta_values = [float(v) for v in np.linspace(0.0, 1.0, 11)]

    tau_targets_cfg = getattr(eval_cfg, "exp_e_overlay_tau_targets", None)
    tau_targets = [int(t) for t in (tau_targets_cfg if tau_targets_cfg is not None else [9])]
    tau_targets = sorted(dict.fromkeys(tau_targets))
    if not tau_targets:
        raise ValueError("exp_k_ddim_eta_hall_radius requires at least one tau target.")
    if any(int(t) < 1 for t in tau_targets):
        raise ValueError(f"Experiment K requires tau >= 1, got overlay targets {tau_targets}")

    tau_override = getattr(eval_cfg, "exp_e_tau_vals", None)
    tau_vals = [int(t) for t in (tau_override if tau_override is not None else tau_targets)]
    tau_vals = sorted(dict.fromkeys(tau_vals + tau_targets))
    tau_internal_vals_cfg = getattr(eval_cfg, "exp_e_tau_internal_vals_ddim", None)
    if tau_internal_vals_cfg is None:
        tau_internal_vals = list(tau_vals)
    else:
        tau_internal_vals = [int(t) for t in tau_internal_vals_cfg]
        if len(tau_internal_vals) != len(tau_vals):
            raise ValueError("Experiment K internal tau list must match exp_e_tau_vals.")

    r_percentage = list(getattr(eval_cfg, "r_percentage", [5, 15, 25, 35, 45, 50]))
    num_samples = int(getattr(eval_cfg, "num_samples", 10000))
    ddim_steps = int(getattr(eval_cfg, "ddim_timesteps", 50))
    skip_type = str(getattr(eval_cfg, "skip_type", "quad"))
    protocol_label = optional_text(getattr(eval_cfg, "gaussian_protocol_tag", None)) or "unspecified_protocol"
    hall_folder = os.path.join(fig_save_folder, artifact_subdir("hallucinations_radius_eta", eval_cfg))
    os.makedirs(hall_folder, exist_ok=True)

    result = run_ddim_eta_hall_radius_sweep(
        diffusion=diffusion,
        dataset_name=dataset_name,
        gaussian_modes=gaussian_modes,
        mode_sigmas=mode_sigmas,
        std_dev=std_dev,
        model=model_for_eval,
        save_folder=hall_folder,
        protocol_label=protocol_label,
        run_name=str(eval_cfg.run_name),
        eta_values=eta_values,
        tau_vals=tau_vals,
        tau_internal_vals=tau_internal_vals,
        tau_targets=tau_targets,
        r_percentage=r_percentage,
        num_samples=num_samples,
        ddim_steps=ddim_steps,
        skip_type=skip_type,
        hall_radius_sigma_multiple=float(getattr(eval_cfg, "hall_radius_sigma_multiple", 5.0)),
    )
    print("Saved DDIM-eta hall-radius sweep under:", os.path.abspath(hall_folder))
    print("Cumulative eta plot:", os.path.abspath(result["cumulative_pdf_path"]))

def run_exp_a_two_mode_assumption(eval_cfg):
    # run exp a and save the tau-versus-\kappa artifacts.

    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    two_mode_folder = os.path.join(fig_save_folder, artifact_subdir("two_mode_assumption", eval_cfg))
    os.makedirs(two_mode_folder, exist_ok=True)
    
    cfg = OmegaConf.create(ckpt_dict["args"])
    num_dims = int(cfg["dataset"].get("num_dims", 2))

    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    tau2_mode, tau2_factor = tau2_variance_scale(eval_cfg, num_dims)

    sampling_mode = eval_cfg.sampling_mode.lower()
    file_stem = f"taus_vs_kappa_{solver_tag_from_cfg(eval_cfg, sampling_mode)}"

    pdf_path = os.path.join(two_mode_folder, f"{file_stem}.pdf")
    remove_stale_pngs(two_mode_folder, f"{file_stem}*.png")

    tau_kappa_dict = plot_tau_1_tau_2_vs_kappa(
        model=model_for_eval,
        diffusion=diffusion,
        dataset_name=cfg.dataset.name, 
        num_samples=eval_cfg.num_samples,
        num_dims=num_dims,
        sampling_mode=eval_cfg.sampling_mode,     
        timesteps=eval_cfg.timesteps,   
        ddim_steps=eval_cfg.ddim_timesteps,    
        skip_type=eval_cfg.skip_type,    
        ddim_eta=0.0,       
        kappa_min=1.0,
        kappa_max=20.0,
        num_kappa_values=500,
        std_dev=std_dev,
        device=eval_cfg.device,
        chunk_size=8192, # B 
        save_path=pdf_path,
        gaussian_modes=gaussian_modes,
        tau2_variance_scale_factor=tau2_factor,
        tau2_variance_scale_mode=tau2_mode,
    )

    save_valid_only_gaussian_samples(
        tau_kappa_dict["final_sample"],
        gaussian_modes=gaussian_modes,
        dataset_name=cfg.dataset.name,
        fig_save_folder=two_mode_folder,
        fig_name=f"exp_a_samples_valid_only_{solver_tag_from_cfg(eval_cfg, sampling_mode)}",
        invalid_sigma_multiple=float(getattr(eval_cfg, "invalid_sigma_multiple", 5.0)),
        std_dev=std_dev,
        mode_sigmas=mode_sigmas,
    )

    save_exp_a_solver_artifacts(
        root=two_mode_folder,
        solver_tag=solver_tag_from_cfg(eval_cfg, sampling_mode),
        sampling_mode=sampling_mode,
        use_exact_score=bool(getattr(eval_cfg, "use_exact_score", False)),
        tau_kappa_dict=tau_kappa_dict,
        kappa_target=float(getattr(eval_cfg, "kappa_target", 5.0)),
    )
    save_exp_a_u_solver_artifacts(
        root=two_mode_folder,
        solver_tag=solver_tag_from_cfg(eval_cfg, sampling_mode),
        sampling_mode=sampling_mode,
        use_exact_score=bool(getattr(eval_cfg, "use_exact_score", False)),
        tau_kappa_dict=tau_kappa_dict,
        kappa_target=float(getattr(eval_cfg, "kappa_target", 5.0)),
    )
    
    
def run_exp_b_convergence_to_nearby_line(eval_cfg):
    # run exp b and save the convergence-to-line artifacts.
    
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    two_mode_folder = os.path.join(fig_save_folder, artifact_subdir("exp_convergence", eval_cfg))
    os.makedirs(two_mode_folder, exist_ok=True)

    print("CWD:", os.getcwd())
    print("Saving under:", os.path.abspath(fig_save_folder))
    
    cfg = OmegaConf.create(ckpt_dict["args"])

    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    num_dims = int(cfg["dataset"].get("num_dims", 2))
    tau2_mode, tau2_factor = tau2_variance_scale(eval_cfg, num_dims)

    sampling_mode = eval_cfg.sampling_mode.lower()
    file_stem = f"exp_convergence_{solver_tag_from_cfg(eval_cfg, sampling_mode)}"

    pdf_path = os.path.join(two_mode_folder, f"{file_stem}.pdf")
    remove_stale_pngs(two_mode_folder, f"{file_stem}*.png")
    remove_stale_pngs(two_mode_folder, f"legend_{file_stem}*.png")

    tau_kappa_dict = compute_tau_streaming(
        model=model_for_eval,
        diffusion=diffusion,
        gaussian_modes=gaussian_modes,
        num_samples=eval_cfg.num_samples,
        num_dims=num_dims,
        sampling_mode=eval_cfg.sampling_mode,
        timesteps=eval_cfg.timesteps,
        ddim_steps=eval_cfg.ddim_timesteps,
        skip_type=eval_cfg.skip_type,
        ddim_eta=0.0,
        kappa_min=1.0,
        kappa_max=20.0,
        num_kappa_values=500,
        std_dev=std_dev,
        device=eval_cfg.device,
        chunk_size=8192,
        tau2_variance_scale_factor=tau2_factor,
        tau2_variance_scale_mode=tau2_mode,
    )

    convergence_dict = convergence_to_nearby_line_streaming(
        model=model_for_eval,
        diffusion=diffusion,
        dataset_name=cfg.dataset.name,
        num_samples=eval_cfg.num_samples,
        num_dims=num_dims,
        sampling_mode=eval_cfg.sampling_mode,
        timesteps=eval_cfg.timesteps,
        ddim_steps=eval_cfg.ddim_timesteps,
        skip_type=eval_cfg.skip_type,
        ddim_eta=0.0,
        std_dev=std_dev,
        device=eval_cfg.device,
        chunk_size=8192,
        tau_kappa_dict=tau_kappa_dict,     
        kappa_target=eval_cfg.kappa_target,
        fig_save_path=pdf_path,
        gaussian_modes=gaussian_modes,
        mode_sigmas=mode_sigmas,
        invalid_sigma_multiple=float(getattr(eval_cfg, "invalid_sigma_multiple", 5.0)),
    )

    print("Saving convergence fig to:", os.path.abspath(pdf_path))

    save_exp_b_solver_artifacts(
        root=two_mode_folder,
        solver_tag=solver_tag_from_cfg(eval_cfg, sampling_mode),
        sampling_mode=sampling_mode,
        use_exact_score=bool(getattr(eval_cfg, "use_exact_score", False)),
        convergence_dict=convergence_dict,
        tau_kappa_dict=tau_kappa_dict,
        kappa_target=float(eval_cfg.kappa_target),
    )

'''
This experiment visualizes the gaussian distribution that is generated by 
sampling from the reverse process.
'''
def run_exp_visualize_gaussian(eval_cfg): 
    # visualize a final sample cloud and write its sample-count file.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    num_dims = int(cfg["dataset"].get("num_dims", 2))
    classification_cfg = sample_classification_cfg(eval_cfg, mode_sigmas, std_dev, num_dims)
    sample_save_folder = os.path.join(fig_save_folder, artifact_subdir("samples", eval_cfg))
    os.makedirs(sample_save_folder, exist_ok=True)
    
    result_dict = sample_two_mode_gaussian(
        model_for_eval, 
        diffusion, 
        timesteps=eval_cfg.timesteps, 
        dataset_name=cfg.dataset.name, 
        num_samples=eval_cfg.num_samples,
        init_samples = None,
        device=eval_cfg.device,
        vis_trajectory=True, 
        sampling_mode=eval_cfg.sampling_mode, 
        skip_type=eval_cfg.skip_type,
        ddim_steps=eval_cfg.ddim_timesteps,
        gaussian_modes=gaussian_modes,
        num_dims=num_dims,
    )

    hall_result_dict = compute_interpolated_distances(
        result_dict["final_sample"], 
        true_modes = gaussian_modes, 
        threshold=classification_cfg["threshold"],
        save_folder = sample_save_folder, 
        sampling_method = eval_cfg.sampling_mode, 
        ddim_timesteps = eval_cfg.ddim_timesteps,
        mode_radii=classification_cfg["mode_radii"],
        pair_radii=classification_cfg["pair_radii"],
    )

    if eval_cfg.sampling_mode == "ddim": 
        fig_name = "samples_ddim_" + str(eval_cfg.ddim_timesteps)
    else: 
        fig_name = "samples_ddpm_" + str(eval_cfg.timesteps)

    visualize_gaussian_with_modes(
        result_dict["final_sample"], 
        true_modes = gaussian_modes, 
        dataset_name = cfg.dataset.name,
        fig_save_folder = sample_save_folder, 
        fig_name = fig_name, 
        invalid_mask = hall_result_dict["invalid_mask"]
    )


def run_exp_c_jacobian_at_midpoint(eval_cfg): 
    # run exp c and save the midpoint jacobian diagnostics.
    if str(eval_cfg.sampling_mode).lower() != "ddim":
        raise ValueError("Experiment C is DDIM-only in the camera-ready pipeline.")
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    num_dims = int(cfg["dataset"].get("num_dims", 2))
    tau2_mode, tau2_factor = tau2_variance_scale(eval_cfg, num_dims)
    eig_value_save_path = os.path.join(fig_save_folder, artifact_subdir("eigvalues", eval_cfg))

    os.makedirs(eig_value_save_path, exist_ok=True)
    remove_stale_pngs(eig_value_save_path, "*.png")
    
    result_dict = sample_two_mode_gaussian_with_eig_values(
        model_for_eval,
        diffusion,
        timesteps=eval_cfg.timesteps,
        dataset_name=cfg.dataset.name,
        num_samples=eval_cfg.num_samples,
        init_samples=None,
        device=eval_cfg.device,
        vis_trajectory=True,
        skip_type=eval_cfg.skip_type,
        ddim_steps=eval_cfg.ddim_timesteps,
        sampling_mode=eval_cfg.sampling_mode,
        save_folder=eig_value_save_path,
        gaussian_modes=gaussian_modes,
    )

    tau_kappa_dict = compute_tau_streaming(
        model=model_for_eval,
        diffusion=diffusion,
        gaussian_modes=gaussian_modes,
        num_samples=eval_cfg.num_samples,
        num_dims=num_dims,
        sampling_mode=eval_cfg.sampling_mode,
        timesteps=eval_cfg.timesteps,
        ddim_steps=eval_cfg.ddim_timesteps,
        skip_type=eval_cfg.skip_type,
        ddim_eta=0.0,
        kappa_min=1.0,
        kappa_max=20.0,
        num_kappa_values=500,
        std_dev=std_dev,
        device=eval_cfg.device,
        chunk_size=8192,
        tau2_variance_scale_factor=tau2_factor,
        tau2_variance_scale_mode=tau2_mode,
    )

    visualize_eig_values_grouped_mean_std(
        all_eigvals=result_dict["all_eigvals"],
        mode_midpoints=result_dict["midpoints"],
        save_path=os.path.join(eig_value_save_path, f"{solver_tag_from_cfg(eval_cfg)}.pdf"),
        tol=1e-12,
        plot_individual=False,
        tau_kappa_dict=tau_kappa_dict,
        kappa_target=eval_cfg.kappa_target, 
    )

    save_exp_c_solver_artifacts(
        root=eig_value_save_path,
        solver_tag=solver_tag_from_cfg(eval_cfg),
        sampling_mode=eval_cfg.sampling_mode,
        use_exact_score=bool(getattr(eval_cfg, "use_exact_score", False)),
        result_dict=result_dict,
    )

''' 
We run an evaluation on an image dataset with the configuration. 
This can now execute image sampling also. 
'''
def run_exp_d_image_diffusion(cfg):
    # run the image-domain evaluation entrypoint.
    # Load the image checkpoint model and its training config
    ckpt_path = os.path.join("results", cfg.run_name, "checkpoints", cfg.ckpt_name)
    ckpt_dict = load_image_checkpoint(ckpt_path, device=cfg.device)
    train_cfg = OmegaConf.create(ckpt_dict["args"])

    diffusion = Diffusion(args=train_cfg, device=cfg.device)

    # Remote Execution # 
    if cfg.remote_exec:
        fig_save_folder = os.path.join("../", cfg.run_name)
    else:
        fig_save_folder = os.path.join(cfg.results_dir, cfg.run_name)
    os.makedirs(fig_save_folder, exist_ok=True)

    result_dict = sample_image_diffusion(
        model=ckpt_dict["model"],
        diffusion=diffusion,
        num_samples=cfg.num_samples,
        img_size=cfg.img_size,
        channels=train_cfg.dataset.channels,
        device=cfg.device,
        sampling_mode=cfg.sampling_mode,
        skip_type=cfg.skip_type,
        ddim_steps=cfg.ddim_timesteps,
        vis_trajectory=bool(getattr(cfg, "vis_trajectory", False)),
        seed = cfg.image_seed
    )

    if cfg.sampling_mode == "ddim":
        sample_tag = f"seed_{cfg.image_seed}_steps_{cfg.ddim_timesteps}"
    else:
        sample_tag = f"seed_{cfg.image_seed}"

    sample_save_path = os.path.join(
        fig_save_folder, f"image_samples_{cfg.sampling_mode}_{sample_tag}.png"
    )
    save_image_grid(result_dict["final_sample"], sample_save_path)
    hallucination_dict = compute_hallucination_rate(result_dict["final_sample"], cfg.run_name, 'triangle_only', "datasets")
    ckpt_stem = cfg.ckpt_name.replace(".pth", "")
    if cfg.sampling_mode == "ddpm": 
        hallucination_save_path = os.path.join(
            fig_save_folder, f"hallucination_ddpm_{ckpt_stem}_seed_{cfg.image_seed}.npy"
        )
    else: 
        hallucination_save_path = os.path.join(
            fig_save_folder, f"hallucination_ddim_{ckpt_stem}_seed_{cfg.image_seed}_steps_{cfg.ddim_timesteps}.npy"
        )
    np.save(hallucination_save_path, hallucination_dict)
    
def run_exp_e_hall_radius_multi_separate(eval_cfg):
    # run exp e for each solver variant and save separate radius plots.
    # The saved tau label can come from a DDIM-side source, while the executed raw reverse-step
    # horizon may differ for the DDPM reference via exp_e_tau_internal_vals_ddpm.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)

    hall_folder = os.path.join(fig_save_folder, artifact_subdir("hallucinations_radius", eval_cfg))
    os.makedirs(hall_folder, exist_ok=True)
    remove_stale_pngs(hall_folder, "*_hall_with_radius.png")
    for stale_path in glob.glob(os.path.join(hall_folder, "exp_e_samples_valid_only_*")):
        try:
            os.remove(stale_path)
        except OSError:
            pass

    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    dataset_name = cfg["dataset"]["name"]

    tau_override = getattr(eval_cfg, "exp_e_tau_vals", None)
    tau_vals = [int(t) for t in (tau_override if tau_override is not None else [5, 9, 15, 30, 40])]
    tau_vals = sorted(dict.fromkeys(tau_vals))
    if not tau_vals:
        raise ValueError("exp_e_tau_vals must contain at least one tau value.")
    if any(int(t) < 1 for t in tau_vals):
        raise ValueError(f"Experiment E requires tau >= 1, got {tau_vals}")
    tau_internal_vals_ddim_cfg = getattr(eval_cfg, "exp_e_tau_internal_vals_ddim", None)
    tau_internal_vals_ddpm_cfg = getattr(eval_cfg, "exp_e_tau_internal_vals_ddpm", None)
    ddpm_internal_tau_vals = ddim_tau_labels_to_raw_steps(
        diffusion,
        tau_vals,
        ddim_steps=int(getattr(eval_cfg, "ddim_timesteps", 50)),
        skip_type=str(getattr(eval_cfg, "skip_type", "quad")),
    )

    def internal_tau_vals_for_mode(mode: str) -> list[int]:
        vals_cfg = tau_internal_vals_ddpm_cfg if str(mode).lower() == "ddpm" else tau_internal_vals_ddim_cfg
        if vals_cfg is None:
            return list(ddpm_internal_tau_vals) if str(mode).lower() == "ddpm" else list(tau_vals)
        vals = [int(t) for t in vals_cfg]
        if len(vals) != len(tau_vals):
            raise ValueError(
                f"Internal Experiment E tau list for mode={mode!r} must match exp_e_tau_vals in length."
            )
        if any(int(t) < 1 for t in vals):
            raise ValueError(f"Experiment E internal tau list must have tau >= 1, got {vals}")
        return vals

    # keep your defaults unless overridden in config
    r_percentage = list(getattr(eval_cfg, "r_percentage", [5, 15, 25, 35, 45, 50]))
    num_samples = int(getattr(eval_cfg, "num_samples", 10000))

    ddim_steps = int(getattr(eval_cfg, "ddim_timesteps", 50))
    skip_type = getattr(eval_cfg, "skip_type", "quad")
    ddim_eta = float(getattr(eval_cfg, "ddim_eta", 0.0))

    base_sampling_mode = eval_cfg.sampling_mode.lower()
    if base_sampling_mode not in {"ddim", "ddpm"}:
        raise ValueError("Experiment E supports only sampling_mode='ddim' or 'ddpm'.")

    base_tag = solver_tag_from_cfg(eval_cfg, base_sampling_mode)

    specs = [(base_tag, base_sampling_mode, {})]
    if base_sampling_mode != "ddpm" and bool_cfg(getattr(eval_cfg, "exp_e_include_mixed_z", True), default=True):
        specs.extend(
            [
                (f"{base_tag}_ddpm_mix_z2", base_sampling_mode, {"z_ddpm": 2}),
                (f"{base_tag}_ddpm_mix_z5", base_sampling_mode, {"z_ddpm": 5}),
                (f"{base_tag}_ddpm_mix_z8", base_sampling_mode, {"z_ddpm": 8}),
            ]
        )
    if base_sampling_mode != "ddpm":
        specs.append(("ddpm", "ddpm", {}))

    for tag, mode, extra in specs:
        # run the base solver and, for DDIM, the DDPM reference.
        tau_internal_vals = internal_tau_vals_for_mode(mode)
        hall_dict = compute_hallucination_with_radius(
            dataset_name=dataset_name,
            diffusion=diffusion,
            model=model_for_eval,
            sampling_mode=mode,
            num_samples=num_samples,
            std_dev=std_dev,
            r_percentage=r_percentage,
            tau_vals=tau_vals,
            tau_internal_vals=tau_internal_vals,
            ddim_steps=ddim_steps,
            skip_type=skip_type,
            ddim_eta=ddim_eta,
            hall_radius_sigma_multiple=float(getattr(eval_cfg, "hall_radius_sigma_multiple", 5.0)),
            gaussian_modes=gaussian_modes,
            mode_sigmas=mode_sigmas,
            **extra,
        )

        # This saves: <hall_folder>/<tag>_hall_with_radius.pdf
        plot_max_tau = 15 if tag == "ddpm" else None
        plot_hallucination_results(hall_dict, hall_folder, sampling_mode=tag, band="sem", max_tau=plot_max_tau)
        np.save(os.path.join(hall_folder, f"{tag}_hall_with_radius.npy"), hall_dict, allow_pickle=True) # get rid of this if we scale dims, replace w/ npy artifact if needded
        save_exp_e_solver_artifacts(
            root=hall_folder,
            solver_tag=tag,
            sampling_mode=mode,
            use_exact_score=bool(getattr(eval_cfg, "use_exact_score", False)),
            hall_dict=hall_dict,
        )

    print("Saved separate radius plots under:", os.path.abspath(hall_folder))

def run_exp_i_midpoint_entry_verification(eval_cfg):
    # run exp i and save the midpoint-entry summary tables.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)
    exp_folder = os.path.join(fig_save_folder, artifact_subdir("midpoint_entry_verification", eval_cfg))
    os.makedirs(exp_folder, exist_ok=True)
    for stale_path in glob.glob(os.path.join(exp_folder, "*")):
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    theta_frac = getattr(eval_cfg, "exp_i_theta_fraction", None)
    if theta_frac is None:
        theta_frac = getattr(eval_cfg, "tau3_theta_fraction", None)
    if theta_frac is None:
        theta_frac = 0.35
    tau3_ddim_time = int(getattr(eval_cfg, "exp_i_tau3_target", 11))
    tau3_internal_raw_ddpm_steps = getattr(eval_cfg, "tau3_internal_raw_ddpm_steps", None)
    tau3_internal_raw_ddpm_steps = (
        tau3_ddim_time if tau3_internal_raw_ddpm_steps is None else int(tau3_internal_raw_ddpm_steps)
    )

    _, gaussian_modes, mode_sigmas, _, _ = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    summary = run_midpoint_entry_verification(
        model=model_for_eval,
        diffusion=diffusion,
        dataset_name=cfg.dataset.name,
        gaussian_modes=gaussian_modes,
        mode_sigmas=mode_sigmas,
        timesteps=int(eval_cfg.timesteps),
        ddim_steps=int(eval_cfg.ddim_timesteps),
        skip_type=str(eval_cfg.skip_type),
        num_samples=int(eval_cfg.num_samples),
        device=eval_cfg.device,
        save_folder=exp_folder,
        theta_frac=float(theta_frac),
        tau_target=tau3_ddim_time,
        internal_tau_steps=tau3_ddim_time,
        invalid_sigma_multiple=float(getattr(eval_cfg, "invalid_sigma_multiple", 5.0)),
        num_dims=int(cfg["dataset"].get("num_dims", 2)),
    )
    print("Saved midpoint-entry verification under:", os.path.abspath(exp_folder))
    print("Inside-tau3 fraction of interpolated:", summary["inside_tau3_fraction_of_interpolated"])

    table_rows = []
    for sampling_mode in ("ddim", "ddpm"):
        internal_tau_steps = tau3_ddim_time if sampling_mode == "ddim" else tau3_internal_raw_ddpm_steps
        table_summary = run_midpoint_probability_table(
            model=model_for_eval,
            diffusion=diffusion,
            dataset_name=cfg.dataset.name,
            gaussian_modes=gaussian_modes,
            mode_sigmas=mode_sigmas,
            timesteps=int(eval_cfg.timesteps),
            ddim_steps=int(eval_cfg.ddim_timesteps),
            skip_type=str(eval_cfg.skip_type),
            sampling_mode=sampling_mode,
            num_samples=int(eval_cfg.num_samples),
            device=eval_cfg.device,
            theta_frac=float(theta_frac),
            tau_target=tau3_ddim_time,
            internal_tau_steps=internal_tau_steps,
            invalid_sigma_multiple=float(getattr(eval_cfg, "invalid_sigma_multiple", 5.0)),
            num_dims=int(cfg["dataset"].get("num_dims", 2)),
            sampling_seed=int(getattr(eval_cfg, "sampling_seed", 0)),
        )
        table_rows.append(
            {
                "sampler": sampling_mode,
                "sampler_label": "DDIM" if sampling_mode == "ddim" else "DDPM",
                "solver_tag": solver_tag_from_cfg(eval_cfg, sampling_mode=sampling_mode),
                **table_summary,
            }
        )

    stem = (
        f"exp_i_midpoint_probability_table_theta_{str(float(theta_frac)).replace('.', 'p')}"
        f"_tau3ddim_{int(tau3_ddim_time)}_ddpmraw_{int(tau3_internal_raw_ddpm_steps)}"
        f"{'_exactscore' if bool(getattr(eval_cfg, 'use_exact_score', False)) else ''}"
    )
    fieldnames = [
        "sampler",
        "sampler_label",
        "solver_tag",
        "p_h",
        "p_h_given_m",
        "p_m",
        "p_h_given_m_times_p_m",
        "p_h_given_not_m_times_not_m",
        "p_h_given_not_m",
        "p_not_m",
        "num_samples",
        "valid_count",
        "interpolated_count",
        "true_mode_count",
        "invalid_count",
        "m_count",
        "h_and_m_count",
        "h_and_not_m_count",
        "first_hit_mean_raw_t_given_m",
        "first_hit_median_raw_t_given_m",
        "tau3_ddim_time",
        "tau3_internal_raw_steps",
    ]
    csv_path = os.path.join(exp_folder, f"{stem}.csv")
    json_path = os.path.join(exp_folder, f"{stem}.json")
    md_path = os.path.join(exp_folder, f"{stem}.md")
    write_csv(csv_path, fieldnames, table_rows)
    write_json(
        json_path,
        {
            "run_name": str(eval_cfg.run_name),
            "artifact_run_name": artifact_run_name(eval_cfg),
            "theta_fraction_of_ell_t": float(theta_frac),
            "tau3_ddim_time": int(tau3_ddim_time),
            "tau3_internal_raw_ddpm_steps": int(tau3_internal_raw_ddpm_steps),
            "probability_space": "valid_final_samples_only",
            "rows": table_rows,
        },
        sort_keys=False,
    )

    def pct(value):
        try:
            v = float(value)
        except Exception:
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return f"{100.0 * v:.4f}%"

    md_lines = [
        "# Experiment I Midpoint Probability Table",
        "",
        "| sampler | P(H) | P(H | M) | P(M) | P(H | M)P(M) | P(H | M^c)P(M^c) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    print("\nExperiment I midpoint probability table")
    print("sampler | P(H) | P(H | M) | P(M) | P(H | M)P(M) | P(H | M^c)P(M^c)")
    for row in table_rows:
        values = [
            row["sampler_label"],
            pct(row.get("p_h")),
            pct(row.get("p_h_given_m")),
            pct(row.get("p_m")),
            pct(row.get("p_h_given_m_times_p_m")),
            pct(row.get("p_h_given_not_m_times_not_m")),
        ]
        print(" | ".join(values))
        md_lines.append(f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Saved midpoint probability table:", os.path.abspath(csv_path))


def run_exp_l_ddpm_tau3_empirical_a_threshold(eval_cfg):
    # run the empirical exp l a-threshold companion.
    learned_cfg = copy.deepcopy(eval_cfg)
    learned_cfg.use_exact_score = False
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(learned_cfg)
    exp_folder = os.path.join(fig_save_folder, "tau3_bisector_closed_form_verification")
    os.makedirs(exp_folder, exist_ok=True)

    learned_model = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, learned_cfg)
    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, learned_cfg, cfg.dataset.name)
    paper_tau3_ddim = int(getattr(learned_cfg, "tau3_target", 9))
    internal_tau3_raw_ddpm_steps = getattr(learned_cfg, "tau3_internal_raw_ddpm_steps", None)
    internal_tau3_raw_ddpm_steps = (
        paper_tau3_ddim if internal_tau3_raw_ddpm_steps is None else int(internal_tau3_raw_ddpm_steps)
    )
    outputs = run_tau3_empirical_a_threshold_companion(
        diffusion=diffusion,
        gaussian_modes=gaussian_modes,
        mode_sigmas=mode_sigmas,
        std_dev=float(std_dev),
        learned_model=learned_model,
        save_folder=exp_folder,
        protocol_tag=optional_text(getattr(learned_cfg, "gaussian_protocol_tag", None)) or "",
        tau_target=int(internal_tau3_raw_ddpm_steps),
        theta_fraction=getattr(learned_cfg, "tau3_theta_fraction", None),
        num_a=int(getattr(learned_cfg, "tau3_empirical_num_a_per_side", 50)),
        num_a_per_side_theorem=int(getattr(learned_cfg, "tau3_closed_form_num_a_per_side", getattr(learned_cfg, "tau3_empirical_num_a_per_side", 50))),
        num_rollouts_per_start=int(getattr(learned_cfg, "tau3_empirical_num_rollouts", 10_000)),
        invalid_sigma_multiple=float(getattr(learned_cfg, "invalid_sigma_multiple", 5.0)),
        device=str(learned_cfg.device),
        chunk_size=int(getattr(learned_cfg, "chunk_size", 4096)),
        sampling_seed=int(getattr(learned_cfg, "tau3_empirical_seed", 42)),
        paper_tau3_ddim=paper_tau3_ddim,
        internal_tau3_raw_ddpm_steps=internal_tau3_raw_ddpm_steps,
    )

    summary = outputs.get("summary", {})
    print("Saved tau3 empirical a-threshold companion under:", os.path.abspath(exp_folder))
    print(
        "[empirical_a_threshold] "
        f"a_star_positive={summary.get('a_star_positive')} "
        f"a_star_negative={summary.get('a_star_negative')} "
        f"a_threshold_scalar={summary.get('a_threshold_scalar')}"
    )
    if "fixed_midpoint_exact_full_mixture_truncated" in outputs:
        exact_folder = os.path.join(
            exp_folder,
            "empirical_a_threshold",
            "fixed_midpoint_exact_full_mixture_truncated",
        )
        print("Saved K plot:", os.path.abspath(os.path.join(exact_folder, "K_worst_time_series.pdf")))
        print("Saved lambda_rep plot:", os.path.abspath(os.path.join(exact_folder, "lambda_rep_worst_time_series.pdf")))
    else:
        print(
            "K/lambda_rep plots were not generated because no empirical a-threshold was found; "
            "see empirical_a_threshold/summary.json and success_vs_a.csv."
        )


def run_exp_e_hall_radius_overlay_single_tau(eval_cfg):
    # overlay several exp e solver curves at a few fixed tau values.
    ckpt_dict, diffusion, cfg, fig_save_folder = initialize_primitives(eval_cfg)
    model_for_eval = prepare_eval_model(ckpt_dict["model"], diffusion, cfg, eval_cfg)

    hall_folder = os.path.join(fig_save_folder, artifact_subdir("hallucinations_radius", eval_cfg))
    os.makedirs(hall_folder, exist_ok=True)

    _, gaussian_modes, mode_sigmas, _, std_dev = load_eval_mixture_arrays(cfg, eval_cfg, cfg.dataset.name)
    dataset_name = cfg["dataset"]["name"]

    tau_targets_cfg = getattr(eval_cfg, "exp_e_overlay_tau_targets", None)
    tau_targets = [int(t) for t in (tau_targets_cfg if tau_targets_cfg is not None else [9])]
    tau_targets = sorted(dict.fromkeys(tau_targets))
    if not tau_targets:
        raise ValueError("exp_e_overlay_tau_targets must contain at least one tau value.")
    if any(int(t) < 1 for t in tau_targets):
        raise ValueError(f"Experiment E requires tau >= 1, got overlay targets {tau_targets}")

    tau_override = getattr(eval_cfg, "exp_e_tau_vals", None)
    tau_vals = [int(t) for t in (tau_override if tau_override is not None else tau_targets)]
    tau_vals = sorted(dict.fromkeys(tau_vals + tau_targets))
    tau_internal_vals_ddim_cfg = getattr(eval_cfg, "exp_e_tau_internal_vals_ddim", None)
    tau_internal_vals_ddpm_cfg = getattr(eval_cfg, "exp_e_tau_internal_vals_ddpm", None)
    ddpm_internal_tau_vals = ddim_tau_labels_to_raw_steps(
        diffusion,
        tau_vals,
        ddim_steps=int(getattr(eval_cfg, "ddim_timesteps", 50)),
        skip_type=str(getattr(eval_cfg, "skip_type", "quad")),
    )

    def internal_tau_vals_for_mode(mode: str) -> list[int]:
        vals_cfg = tau_internal_vals_ddpm_cfg if str(mode).lower() == "ddpm" else tau_internal_vals_ddim_cfg
        if vals_cfg is None:
            return list(ddpm_internal_tau_vals) if str(mode).lower() == "ddpm" else list(tau_vals)
        vals = [int(t) for t in vals_cfg]
        if len(vals) != len(tau_vals):
            raise ValueError(
                f"Internal Experiment E tau list for mode={mode!r} must match exp_e_tau_vals in length."
            )
        if any(int(t) < 1 for t in vals):
            raise ValueError(f"Experiment E internal tau list must have tau >= 1, got {vals}")
        return vals

    # keep your defaults unless overridden
    r_percentage = list(getattr(eval_cfg, "r_percentage", [5, 15, 25, 35, 45, 50]))
    num_samples = int(getattr(eval_cfg, "num_samples", 10000))

    ddim_steps = int(getattr(eval_cfg, "ddim_timesteps", 50))
    skip_type = getattr(eval_cfg, "skip_type", "quad")
    ddim_eta = float(getattr(eval_cfg, "ddim_eta", 0.0))

    base_sampling_mode = eval_cfg.sampling_mode.lower()
    if base_sampling_mode not in {"ddim", "ddpm"}:
        raise ValueError("Experiment E supports only sampling_mode='ddim' or 'ddpm'.")

    results_by_label = {}
    base_label = solver_display_name(base_sampling_mode)

    results_by_label[base_label] = compute_hallucination_with_radius(
        dataset_name=dataset_name,
        diffusion=diffusion,
        model=model_for_eval,
        sampling_mode=base_sampling_mode,
        num_samples=num_samples,
        std_dev=std_dev,
        r_percentage=r_percentage,
        tau_vals=tau_vals,
        tau_internal_vals=internal_tau_vals_for_mode(base_sampling_mode),
        ddim_steps=ddim_steps,
        skip_type=skip_type,
        ddim_eta=ddim_eta,
        hall_radius_sigma_multiple=float(getattr(eval_cfg, "hall_radius_sigma_multiple", 5.0)),
        gaussian_modes=gaussian_modes,
        mode_sigmas=mode_sigmas,
    )

    if base_sampling_mode != "ddpm" and bool_cfg(getattr(eval_cfg, "exp_e_include_mixed_z", True), default=True):
        for z_ddpm in (2, 5, 8):
            results_by_label[f"{base_label} + {z_ddpm} DDPM steps"] = compute_hallucination_with_radius(
                dataset_name=dataset_name,
                diffusion=diffusion,
                model=model_for_eval,
                sampling_mode=base_sampling_mode,
                z_ddpm=z_ddpm,
                num_samples=num_samples,
                std_dev=std_dev,
                r_percentage=r_percentage,
                tau_vals=tau_vals,
                tau_internal_vals=internal_tau_vals_for_mode(base_sampling_mode),
                ddim_steps=ddim_steps,
                skip_type=skip_type,
                ddim_eta=ddim_eta,
                hall_radius_sigma_multiple=float(getattr(eval_cfg, "hall_radius_sigma_multiple", 5.0)),
                gaussian_modes=gaussian_modes,
                mode_sigmas=mode_sigmas,
            )

    if base_sampling_mode != "ddpm":
        results_by_label["DDPM"] = compute_hallucination_with_radius(
            dataset_name=dataset_name,
            diffusion=diffusion,
            model=model_for_eval,
            sampling_mode="ddpm",
            num_samples=num_samples,
            std_dev=std_dev,
            r_percentage=r_percentage,
            tau_vals=tau_vals,
            tau_internal_vals=internal_tau_vals_for_mode("ddpm"),
            hall_radius_sigma_multiple=float(getattr(eval_cfg, "hall_radius_sigma_multiple", 5.0)),
            gaussian_modes=gaussian_modes,
            mode_sigmas=mode_sigmas,
        )

    for tau_target in tau_targets:
        save_path = os.path.join(
            hall_folder,
            f"overlay_tau{tau_target}_{dataset_name}_{solver_tag_from_cfg(eval_cfg, base_sampling_mode)}.pdf"
        )

        plot_hallucination_overlay_single_tau(
            results_by_label,
            save_path=save_path,
            tau_target=int(tau_target),
            band="sem",
            alpha=0.22
        )

        print("Saved overlay:", os.path.abspath(save_path))


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="eval_config",
)
def run_eval(cfg: DictConfig):
    # dispatch the evaluation experiment.
    seed = 42
    set_seed(seed)
    
    if cfg.exp_name == "exp_a_two_mode_assumption": 
        run_exp_a_two_mode_assumption(cfg)

    elif cfg.exp_name == "exp_b_convergence_to_nearby_line": 
        run_exp_b_convergence_to_nearby_line(cfg)

    elif cfg.exp_name == "exp_c_jacobian_at_midpoint":
        run_exp_c_jacobian_at_midpoint(cfg)

    elif cfg.exp_name == "visualize_samples": 
        run_exp_visualize_gaussian(cfg)

    elif cfg.exp_name == "exp_i_midpoint_entry_verification":
        run_exp_i_midpoint_entry_verification(cfg)

    elif cfg.exp_name == "exp_l_ddpm_tau3_empirical_a_threshold":
        run_exp_l_ddpm_tau3_empirical_a_threshold(cfg)

    elif cfg.exp_name == "exp_e_ddim_ddpm_hall_rate": 
        run_exp_e_hall_radius_multi_separate(cfg)
        run_exp_e_hall_radius_overlay_single_tau(cfg)

    elif cfg.exp_name == "exp_k_ddim_eta_hall_radius":
        run_exp_k_ddim_eta_hall_radius(cfg)

    elif cfg.exp_name == "exp_j_ddim_eta_hallucination_rate":
        run_exp_j_ddim_eta_hallucination_rate(cfg)

    elif cfg.exp_name == "exp_d_image_diffusion":
        run_exp_d_image_diffusion(cfg)

    else: 
        raise ValueError (f"{cfg.exp_name} is not recognized") 

   
if __name__ == "__main__": 
    run_eval()
