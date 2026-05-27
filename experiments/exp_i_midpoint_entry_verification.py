import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.artifact_io import write_csv
from common.gaussian_mixture_2d import classify_samples_by_geometry, mode_radii_from_sigmas, pair_radii_from_sigmas
from experiments.exp_a_two_mode import sample_two_mode_gaussian


LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12


def plot_summary(
    save_path: str,
    first_hit_raw_t: np.ndarray,
    ever_entered_mask: np.ndarray,
    inside_tau3_mask: np.ndarray,
    tau_target: int,
    internal_tau_steps: int,
    theta_frac: float,
    solver_label: str,
    interpolated_count: int,
) -> None:
    # Plot a diagnostic for the midpoint event M used by the Exp. I table.
    # tau_target is the paper-facing DDIM tau_3 label; internal_tau_steps is the raw-time cutoff executed by the sampler.
    valid_hits = first_hit_raw_t[np.isfinite(first_hit_raw_t)].astype(np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    ax = axes[0]
    if valid_hits.size:
        bins = np.arange(0.5, valid_hits.max() + 1.5, 1.0)
        ax.hist(valid_hits, bins=bins, color="tab:blue", alpha=0.85, edgecolor="white")
    ax.axvline(float(internal_tau_steps), color="tab:red", linestyle="--", linewidth=2)
    ax.set_xlabel("First-hit raw DDPM time $t$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Interpolated trajectory count", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    labels = [
        "Ever entered",
        rf"Inside sometime with $t \leq {internal_tau_steps}$",
        rf"Never entered",
    ]
    ever_frac = float(np.mean(ever_entered_mask)) if ever_entered_mask.size else float("nan")
    inside_tau3_frac = float(np.mean(inside_tau3_mask)) if inside_tau3_mask.size else float("nan")
    never_frac = 1.0 - ever_frac if np.isfinite(ever_frac) else float("nan")
    vals = [
        ever_frac * 100.0 if np.isfinite(ever_frac) else np.nan,
        inside_tau3_frac * 100.0 if np.isfinite(inside_tau3_frac) else np.nan,
        never_frac * 100.0 if np.isfinite(never_frac) else np.nan,
    ]
    ax.bar(labels, vals, color=["tab:green", "tab:orange", "tab:gray"], alpha=0.85)
    ax.set_ylabel("Interpolated trajectories (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_no_interpolations(
    save_path: str,
    *,
    tau_target: int,
    internal_tau_steps: int,
    theta_frac: float,
) -> None:
    # write a simple placeholder figure when the selected run produces no interpolated final samples.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "no interpolated final samples were observed",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.text(
        0.5,
        0.40,
        rf"$\theta={theta_frac}$, $\tau_3={tau_target}$, internal $t={internal_tau_steps}$",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_midpoint_ratio_trajectory(
    *,
    trajectories: list[np.ndarray],
    raw_ts: np.ndarray,
    diffusion,
    mode_i: np.ndarray,
    mode_j: np.ndarray,
) -> np.ndarray:
    # compute $\|x_t - m_t\| / \ell_t$ at each stored raw-time node for the assigned closest pair.
    # experiment i defines the midpoint event by thresholding this ratio at $\vartheta$.
    ratios = []
    base_pair_length = np.linalg.norm(mode_j - mode_i, axis=1)
    for x_t, t_eff in zip(trajectories, np.asarray(raw_ts, dtype=np.int64).tolist()):
        alpha_bar_t = float(diffusion.alphas_prod[int(t_eff)].detach().cpu().item())
        sqrt_alpha_bar_t = float(np.sqrt(alpha_bar_t))
        midpoint_t = 0.5 * sqrt_alpha_bar_t * (mode_i + mode_j)
        ell_t = sqrt_alpha_bar_t * base_pair_length
        ell_t = np.maximum(ell_t, 1e-12)
        dist_t = np.linalg.norm(np.asarray(x_t, dtype=np.float64) - midpoint_t, axis=1)
        ratios.append((dist_t / ell_t).astype(np.float64, copy=False))
    if not ratios:
        return np.zeros((0, mode_i.shape[0]), dtype=np.float64)
    return np.stack(ratios, axis=0)


def raw_time_midpoint_event_from_ratios(
    *,
    raw_ts: np.ndarray,
    ratio_by_state: np.ndarray,
    internal_tau_steps: int,
    theta_frac: float,
) -> np.ndarray:
    # mark trajectories that enter the midpoint ball at some stored node with raw-time $t$ below the sampler cutoff.
    keep = np.asarray(raw_ts, dtype=np.int64) <= int(internal_tau_steps)
    if not np.any(keep):
        return np.zeros((ratio_by_state.shape[1],), dtype=bool)
    return np.any(ratio_by_state[keep] <= float(theta_frac), axis=0)


def compute_first_hit_raw_t(
    *,
    raw_ts: np.ndarray,
    ratio_by_state: np.ndarray,
    theta_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    # record the first stored raw-time node at which the trajectory enters the midpoint ball.
    hits = ratio_by_state <= float(theta_frac)
    ever_entered_mask = np.any(hits, axis=0)
    first_hit_raw_times = np.full((ratio_by_state.shape[1],), np.nan, dtype=np.float64)
    if np.any(ever_entered_mask):
        first_hit_idx = np.argmax(hits, axis=0)
        first_hit_raw_times[ever_entered_mask] = np.asarray(raw_ts, dtype=np.int64)[
            first_hit_idx[ever_entered_mask]
        ]
    return ever_entered_mask, first_hit_raw_times


def run_midpoint_entry_verification(
    model,
    diffusion,
    dataset_name: str,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    *,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
    num_samples: int,
    device: str,
    save_folder: str,
    theta_frac: float,
    tau_target: int,
    internal_tau_steps: int | None = None,
    invalid_sigma_multiple: float,
    num_dims: int = 2,
):
    # run experiment i and summarize when interpolated trajectories first enter the midpoint ball.
    # tau_target is the DDIM-side paper label; internal_tau_steps is the raw-time cutoff used by this sampler.
    os.makedirs(save_folder, exist_ok=True)
    internal_tau_steps = int(tau_target) if internal_tau_steps is None else int(internal_tau_steps)

    result = sample_two_mode_gaussian(
        model,
        diffusion,
        timesteps=int(timesteps),
        dataset_name=dataset_name,
        num_samples=int(num_samples),
        init_samples=None,
        device=device,
        vis_trajectory=True,
        sampling_mode="ddim",
        skip_type=str(skip_type),
        ddim_steps=int(ddim_steps),
        gaussian_modes=np.asarray(gaussian_modes, dtype=np.float64),
        num_dims=int(num_dims),
    )

    final_samples = np.asarray(result["final_sample"], dtype=np.float64)
    modes = np.asarray(gaussian_modes, dtype=np.float64)
    sigmas = np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    mode_radii = mode_radii_from_sigmas(sigmas, sigma_multiple=float(invalid_sigma_multiple))
    pair_radii = pair_radii_from_sigmas(sigmas, sigma_multiple=float(invalid_sigma_multiple))
    geom = classify_samples_by_geometry(
        final_samples,
        modes,
        threshold=None,
        mode_radii=mode_radii,
        pair_radii=pair_radii,
    )

    interp_idx = np.where(np.asarray(geom["interpolated_mask"], dtype=bool))[0]
    stem = (
        f"ddim_theta_{str(theta_frac).replace('.', 'p')}"
        f"_tau3ddim_{int(tau_target)}_internal_{int(internal_tau_steps)}"
    )
    if interp_idx.size == 0:
        # Save a summary even when this run produces no interpolated endpoints.
        summary = {
            "sampling_mode": "ddim",
            "ddim_steps": int(ddim_steps),
            "num_samples": int(num_samples),
            "interpolated_count": 0,
            "theta_fraction_of_ell": float(theta_frac),
            "tau3_target": int(tau_target),
            "tau3_ddim_time": int(tau_target),
            "tau3_internal_raw_steps": int(internal_tau_steps),
            "region_definition": "time_scaled_midpoint_ball_with_radius_theta_times_time_scaled_pair_distance",
            "radius_percentage_basis": "full_pair_distance",
            "radius_space": "ell_t",
            "pair_assignment": "final_interpolated_nearest_pair",
            "time_basis": "raw_ddpm_time",
            "event_definition": "inside_region_at_some_sampled_solver_node_with_raw_t_lte_tau3",
            "ever_entered_count": 0,
            "ever_entered_fraction_of_interpolated": None,
            "inside_tau3_count": 0,
            "inside_tau3_fraction_of_interpolated": None,
            "first_hit_mean_raw_t": None,
            "first_hit_median_raw_t": None,
            "first_hit_quantiles_raw_t": None,
            "status": "no_interpolated_final_samples",
        }
        with open(os.path.join(save_folder, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_csv(
            os.path.join(save_folder, f"{stem}_rows.csv"),
            [
                "sample_index",
                "segment_i",
                "segment_j",
                "first_hit_raw_t",
                "ever_entered_region",
                "inside_region_with_raw_t_lte_target",
            ],
            [],
        )
        plot_no_interpolations(
            save_path=os.path.join(save_folder, f"{stem}.pdf"),
            tau_target=int(tau_target),
            internal_tau_steps=int(internal_tau_steps),
            theta_frac=float(theta_frac),
        )
        return summary
    nearest_pair_idx = np.asarray(geom["nearest_pair_idx"], dtype=np.int64)[interp_idx]
    trajectories = [np.asarray(t.cpu().numpy(), dtype=np.float64) for t in result["trajectories"][:-1]]
    seq = np.asarray(result["seq"], dtype=np.int64)

    mode_i = modes[nearest_pair_idx[:, 0]]
    mode_j = modes[nearest_pair_idx[:, 1]]
    ratio_by_state = compute_midpoint_ratio_trajectory(
        trajectories=[x_t[interp_idx] for x_t in trajectories],
        raw_ts=seq,
        diffusion=diffusion,
        mode_i=mode_i,
        mode_j=mode_j,
    )
    ever_entered_mask, first_hit_raw_times = compute_first_hit_raw_t(
        raw_ts=seq,
        ratio_by_state=ratio_by_state,
        theta_frac=float(theta_frac),
    )
    inside_tau3_mask = raw_time_midpoint_event_from_ratios(
        raw_ts=seq,
        ratio_by_state=ratio_by_state,
        internal_tau_steps=int(internal_tau_steps),
        theta_frac=float(theta_frac),
    )
    rows = []
    for local_idx, sample_idx in enumerate(interp_idx):
        i, j = sorted((int(nearest_pair_idx[local_idx, 0]), int(nearest_pair_idx[local_idx, 1])))
        rows.append(
            {
                "sample_index": int(sample_idx),
                "segment_i": int(i),
                "segment_j": int(j),
                "first_hit_raw_t": (
                    "" if not np.isfinite(first_hit_raw_times[local_idx]) else int(first_hit_raw_times[local_idx])
                ),
                "ever_entered_region": bool(ever_entered_mask[local_idx]),
                "inside_region_with_raw_t_lte_target": bool(inside_tau3_mask[local_idx]),
            }
        )

    summary = {
        "sampling_mode": "ddim",
        "ddim_steps": int(ddim_steps),
        "num_samples": int(num_samples),
        "interpolated_count": int(interp_idx.size),
        "theta_fraction_of_ell": float(theta_frac),
        "tau3_target": int(tau_target),
        "tau3_ddim_time": int(tau_target),
        "tau3_internal_raw_steps": int(internal_tau_steps),
        "region_definition": "time_scaled_midpoint_ball_with_radius_theta_times_time_scaled_pair_distance",
        "radius_percentage_basis": "full_pair_distance",
        "radius_space": "ell_t",
        "pair_assignment": "final_interpolated_nearest_pair",
        "time_basis": "raw_ddpm_time",
        "event_definition": "inside_region_at_some_sampled_solver_node_with_raw_t_lte_tau3",
        "ever_entered_count": int(np.sum(ever_entered_mask)),
        "ever_entered_fraction_of_interpolated": float(np.mean(ever_entered_mask)),
        "inside_tau3_count": int(np.sum(inside_tau3_mask)),
        "inside_tau3_fraction_of_interpolated": float(np.mean(inside_tau3_mask)),
        "first_hit_mean_raw_t": (
            float(np.nanmean(first_hit_raw_times)) if np.isfinite(first_hit_raw_times).any() else None
        ),
        "first_hit_median_raw_t": (
            float(np.nanmedian(first_hit_raw_times)) if np.isfinite(first_hit_raw_times).any() else None
        ),
        "first_hit_quantiles_raw_t": (
            np.nanquantile(first_hit_raw_times, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
            if np.isfinite(first_hit_raw_times).any()
            else None
        ),
    }

    with open(os.path.join(save_folder, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(
        os.path.join(save_folder, f"{stem}_rows.csv"),
        [
            "sample_index",
            "segment_i",
            "segment_j",
            "first_hit_raw_t",
            "ever_entered_region",
            "inside_region_with_raw_t_lte_target",
        ],
        rows,
    )
    plot_summary(
        save_path=os.path.join(save_folder, f"{stem}.pdf"),
        first_hit_raw_t=first_hit_raw_times,
        ever_entered_mask=ever_entered_mask,
        inside_tau3_mask=inside_tau3_mask,
        tau_target=int(tau_target),
        internal_tau_steps=int(internal_tau_steps),
        theta_frac=float(theta_frac),
        solver_label="DDIM",
        interpolated_count=int(interp_idx.size),
    )

    return summary


def run_midpoint_probability_table(
    model,
    diffusion,
    dataset_name: str,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    *,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
    sampling_mode: str,
    num_samples: int,
    device: str,
    theta_frac: float,
    tau_target: int,
    internal_tau_steps: int | None = None,
    invalid_sigma_multiple: float,
    num_dims: int = 2,
    sampling_seed: int = 0,
):
    # run the exp. i midpoint table on all trajectories using sampler-specific raw-time cutoffs.
    # all probabilities are measured on valid final samples only, so invalid samples are outside the population.
    np.random.seed(int(sampling_seed))
    torch.manual_seed(int(sampling_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))
    internal_tau_steps = int(tau_target) if internal_tau_steps is None else int(internal_tau_steps)
    result = sample_two_mode_gaussian(
        model,
        diffusion,
        timesteps=int(timesteps),
        dataset_name=dataset_name,
        num_samples=int(num_samples),
        init_samples=None,
        device=device,
        vis_trajectory=True,
        sampling_mode=str(sampling_mode),
        skip_type=str(skip_type),
        ddim_steps=int(ddim_steps),
        gaussian_modes=np.asarray(gaussian_modes, dtype=np.float64),
        num_dims=int(num_dims),
    )

    final_samples = np.asarray(result["final_sample"], dtype=np.float64)
    modes = np.asarray(gaussian_modes, dtype=np.float64)
    sigmas = np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    mode_radii = mode_radii_from_sigmas(sigmas, sigma_multiple=float(invalid_sigma_multiple))
    pair_radii = pair_radii_from_sigmas(sigmas, sigma_multiple=float(invalid_sigma_multiple))
    geom = classify_samples_by_geometry(
        final_samples,
        modes,
        threshold=None,
        mode_radii=mode_radii,
        pair_radii=pair_radii,
    )

    interpolated_mask = np.asarray(geom["interpolated_mask"], dtype=bool)
    invalid_mask = np.asarray(geom["invalid_mask"], dtype=bool)
    true_mode_mask = np.asarray(geom["true_mode_mask"], dtype=bool)
    nearest_pair_idx = np.asarray(geom["nearest_pair_idx"], dtype=np.int64)
    trajectories = [np.asarray(t.cpu().numpy(), dtype=np.float64) for t in result["trajectories"][:-1]]
    seq = np.asarray(result["seq"], dtype=np.int64)

    # use the closest pair assigned by the final sample so the midpoint event is defined on every trajectory.
    mode_i = modes[nearest_pair_idx[:, 0]]
    mode_j = modes[nearest_pair_idx[:, 1]]
    ratio_by_state = compute_midpoint_ratio_trajectory(
        trajectories=trajectories,
        raw_ts=seq,
        diffusion=diffusion,
        mode_i=mode_i,
        mode_j=mode_j,
    )
    midpoint_window_mask = raw_time_midpoint_event_from_ratios(
        raw_ts=seq,
        ratio_by_state=ratio_by_state,
        internal_tau_steps=int(internal_tau_steps),
        theta_frac=float(theta_frac),
    )
    _, first_hit_raw_times = compute_first_hit_raw_t(
        raw_ts=seq,
        ratio_by_state=ratio_by_state,
        theta_frac=float(theta_frac),
    )

    valid_mask = ~invalid_mask
    h_mask = interpolated_mask & valid_mask
    m_mask = midpoint_window_mask & valid_mask
    not_m_mask = valid_mask & (~m_mask)
    h_and_m_mask = h_mask & m_mask
    h_and_not_m_mask = h_mask & not_m_mask
    num_valid = int(np.sum(valid_mask))

    p_h = float(np.sum(h_mask) / num_valid) if num_valid > 0 else float("nan")
    p_m = float(np.sum(m_mask) / num_valid) if num_valid > 0 else float("nan")
    p_h_and_m = float(np.sum(h_and_m_mask) / num_valid) if num_valid > 0 else float("nan")
    p_not_m = float(np.sum(not_m_mask) / num_valid) if num_valid > 0 else float("nan")
    p_h_and_not_m = float(np.sum(h_and_not_m_mask) / num_valid) if num_valid > 0 else float("nan")
    p_h_given_m = float(np.sum(h_and_m_mask) / np.sum(m_mask)) if np.sum(m_mask) > 0 else float("nan")
    p_h_given_not_m = (
        float(np.sum(h_and_not_m_mask) / np.sum(not_m_mask)) if np.sum(not_m_mask) > 0 else float("nan")
    )

    return {
        "sampling_mode": str(sampling_mode).lower(),
        "num_samples": int(final_samples.shape[0]),
        "theta_fraction_of_ell_t": float(theta_frac),
        "tau3_target": int(tau_target),
        "tau3_ddim_time": int(tau_target),
        "tau3_internal_raw_steps": int(internal_tau_steps),
        "time_basis": "raw_ddpm_time",
        "probability_space": "valid_samples_only",
        "h_event_definition": "final_sample_is_interpolated_within_valid_samples",
        "m_event_definition": "trajectory_enters_midpoint_ball_of_radius_theta_times_ell_t_at_some_sampled_solver_node_with_raw_t_lte_internal_tau_within_valid_samples",
        "pair_assignment": "final_sample_nearest_pair_for_all_trajectories",
        "radius_space": "ell_t",
        "kept_raw_t_nodes": [int(t) for t in np.asarray(seq, dtype=np.int64).tolist()],
        "p_h_given_m": float(p_h_given_m),
        "p_m": float(p_m),
        "p_h_given_not_m": float(p_h_given_not_m),
        "p_h_given_m_times_p_m": float(p_h_and_m),
        "p_h_given_not_m_times_not_m": float(p_h_and_not_m),
        "p_h": float(p_h),
        "p_not_m": float(p_not_m),
        "valid_count": int(np.sum(valid_mask)),
        "interpolated_count": int(np.sum(h_mask)),
        "true_mode_count": int(np.sum(true_mode_mask & valid_mask)),
        "invalid_count": int(np.sum(invalid_mask)),
        "sampling_seed": int(sampling_seed),
        "m_count": int(np.sum(m_mask)),
        "h_and_m_count": int(np.sum(h_and_m_mask)),
        "h_and_not_m_count": int(np.sum(h_and_not_m_mask)),
        "first_hit_mean_raw_t_given_m": (
            float(np.nanmean(first_hit_raw_times[m_mask])) if np.any(m_mask) else None
        ),
        "first_hit_median_raw_t_given_m": (
            float(np.nanmedian(first_hit_raw_times[m_mask])) if np.any(m_mask) else None
        ),
    }
