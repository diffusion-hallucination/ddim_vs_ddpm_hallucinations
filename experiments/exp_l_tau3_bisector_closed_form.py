import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.artifact_io import save_dual_figure as save_figure, to_jsonable, write_csv
from common.gaussian_exact_score import oracle_eps_and_score_at_t
from common.gaussian_pairs import build_pair_geometry_table
from common.gaussian_mixture_2d import (
    classify_samples_by_geometry,
    geometry_radius_scale,
    mode_radii_from_sigmas,
    pair_radii_from_sigmas,
)
from common.reverse_solvers import call_model, ddpm_step_from_eps_with_generator


LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
SEGMENT_HALF_FRACTION = 0.5
MAX_INTERPOLATION_RATE_FOR_ASTAR = 1.0e-3
FIXED_A_ENDPOINT_EXTENSION = 25.0 * math.exp(-7.0)
DEFAULT_THEOREM_A_INTERVALS_PER_SIDE = 50


def a_midpoint_fraction_from_protocol(a_midpoint_radius, protocol_tag: str) -> float:
    # resolve the midpoint neighborhood fraction used to define the physical coordinate window.
    if a_midpoint_radius is not None:
        return float(a_midpoint_radius)
    return 0.15


def reference_pair_length(pair_geometry: dict) -> float:
    # resolve the common closest-pair length used by the paper's physical coordinate $a$.
    # exp. l assumes one nearest-neighbor spacing $\ell$ so that the same base pair segment is shared by every closest pair.
    ell = np.asarray(pair_geometry["ell"], dtype=np.float64).reshape(-1)
    if ell.size == 0:
        raise ValueError("exp. l requires at least one closest pair.")
    ell_ref = float(pair_geometry["closest_pair_distance"])
    if not np.allclose(ell, ell_ref, rtol=1e-6, atol=1e-10):
        raise ValueError(
            "exp. l physical-coordinate outputs require a common closest-pair length across the selected pair set."
        )
    return ell_ref


def physical_a_window(*, pair_geometry: dict, a_midpoint_fraction: float) -> tuple[float, float, float]:
    # convert the nondimensional midpoint fraction into the paper's physical coordinate $a$.
    ell_ref = reference_pair_length(pair_geometry)
    return float(a_midpoint_fraction * ell_ref), float(SEGMENT_HALF_FRACTION * ell_ref), float(ell_ref)


def a_endpoint_extension_from_protocol(a_endpoint_extension, protocol_tag: str) -> float:
    # resolve the fixed Exp. L extension beyond the pure pair segment.
    # in the fixed protocol, the theorem-facing ambient window extends from $\ell/2$ to $(\ell+\varepsilon)/2$
    # with $\varepsilon = 25 e^{-7}$.
    if a_endpoint_extension is not None:
        return float(a_endpoint_extension)
    protocol = str(protocol_tag).strip().lower()
    if protocol == "fixed":
        return float(FIXED_A_ENDPOINT_EXTENSION)
    return 0.0


def interval_midpoints(lower: float, upper: float, num_intervals: int) -> np.ndarray:
    # build interval midpoints so the theorem checks follow the paper's "50 intervals per side" wording.
    if int(num_intervals) < 1:
        raise ValueError("num_intervals must be at least 1.")
    lower = float(lower)
    upper = float(upper)
    if not upper > lower:
        raise ValueError(f"upper must exceed lower, got [{lower}, {upper}].")
    edges = np.linspace(lower, upper, int(num_intervals) + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def interval_check_points(lower: float, upper: float, num_intervals: int) -> np.ndarray:
    # Use interval midpoints when there is a genuine interval, and the endpoint itself for a degenerate threshold.
    lower = float(lower)
    upper = float(upper)
    if upper < lower - 1e-12:
        raise ValueError(f"upper must be at least lower, got [{lower}, {upper}].")
    if upper <= lower + 1e-12:
        return np.asarray([lower], dtype=np.float64)
    return interval_midpoints(lower, upper, int(num_intervals))


def theorem_a_grid(
    *,
    a_midpoint_radius: float,
    lambda_upper_bound_abs: float,
    num_intervals_per_side: int,
) -> np.ndarray:
    # Build one shared theorem-facing a-grid whose masks realize the paper checks:
    #   - 50 intervals per side on vartheta <= |a| <= 2 vartheta for K(t)
    #   - 50 intervals per side on vartheta <= |a| <= a_* for lambda_rep.
    # We also keep a=0 for the separate centerline diagnostic and slice plots.
    theta = float(a_midpoint_radius)
    positive_k = interval_midpoints(theta, float(2.0 * theta), int(num_intervals_per_side))
    positive_lambda = interval_check_points(theta, float(lambda_upper_bound_abs), int(num_intervals_per_side))
    vals = np.concatenate(
        [
            -positive_lambda[::-1],
            -positive_k[::-1],
            np.asarray([0.0], dtype=np.float64),
            positive_k,
            positive_lambda,
        ]
    )
    return np.unique(np.asarray(vals, dtype=np.float64))


def build_pair_geometry(modes: np.ndarray) -> dict:
    # build the closest-pair midpoint and axis data shared by the exp. l analyses.
    # this table is indexed by closest pair and is reused for both the exact and learned branches.
    geometry = build_pair_geometry_table(modes)
    return {
        "pair_i": np.asarray(geometry["pair_i"], dtype=np.int64),
        "pair_j": np.asarray(geometry["pair_j"], dtype=np.int64),
        "mu_i": np.asarray(geometry["mu_i"], dtype=np.float64),
        "mu_j": np.asarray(geometry["mu_j"], dtype=np.float64),
        "ell": np.asarray(geometry["ell"], dtype=np.float64),
        "u": np.asarray(geometry["u"], dtype=np.float64),
        "midpoint_fixed": np.asarray(geometry["midpoint"], dtype=np.float64),
        "closest_pair_distance": float(geometry["closest_pair_distance"]),
    }


def fixed_midpoint_a_grid(
    *,
    a_grid: np.ndarray,
    pair_geometry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Map the paper coordinate A_t=a to ambient points x_t = m + a u on the closest-pair bisector line.
    num_pairs = int(np.asarray(pair_geometry["ell"], dtype=np.float64).shape[0])
    midpoint_fixed = np.asarray(pair_geometry["midpoint_fixed"], dtype=np.float64)
    u = np.asarray(pair_geometry["u"], dtype=np.float64)
    a_vals = np.asarray(a_grid, dtype=np.float64).reshape(-1)
    a_physical = np.broadcast_to(a_vals[None, None, :], (num_pairs, 1, int(a_vals.size))).copy()
    z_physical = np.zeros_like(a_physical)
    x_grid = midpoint_fixed[:, None, None, :] + a_physical[:, :, :, None] * u[:, None, None, :]
    return a_physical, z_physical, x_grid


@torch.no_grad()
def full_exact_mixture_score_grid(
    *,
    diffusion,
    x_grid: np.ndarray,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    std_dev: float,
    t: int,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    # evaluate the exact full-mixture Gaussian score on the same fixed-midpoint grid.
    # unlike the theorem-facing two-mode branch, this keeps all 25 Gaussian components and computes
    # the exact score numerically from their posterior responsibilities.
    num_pairs = int(x_grid.shape[0])
    num_dims = int(x_grid.shape[-1])
    grid_shape = tuple(int(v) for v in x_grid.shape[1:-1])
    x_flat = torch.as_tensor(
        x_grid.reshape(num_pairs * int(np.prod(grid_shape, dtype=np.int64)), num_dims),
        device=device,
        dtype=torch.float32,
    )
    score_chunks = []
    for start in range(0, int(x_flat.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(x_flat.shape[0]))
        xb = x_flat[start:end]
        _, score_true = oracle_eps_and_score_at_t(
            diffusion=diffusion,
            x_t=xb,
            gaussian_modes=np.asarray(gaussian_modes, dtype=np.float64),
            std_dev=float(std_dev),
            t_eff=int(t),
            mode_sigmas=np.asarray(mode_sigmas, dtype=np.float64),
        )
        score_chunks.append(score_true.detach().cpu())
    score_flat = torch.cat(score_chunks, dim=0).numpy().astype(np.float64, copy=False)
    return score_flat.reshape((num_pairs,) + grid_shape + (num_dims,))


def fixed_midpoint_full_exact_mixture_drift(
    *,
    a_grid: np.ndarray,
    pair_geometry: dict,
    diffusion,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    std_dev: float,
    t: int,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # compute the stored reverse-process drift b(a,t) using the exact full 25-mode Gaussian score.
    # b_code stores the theorem drift itself, not a negated code-clock proxy.
    a_physical, z_physical, x_grid = fixed_midpoint_a_grid(
        a_grid=a_grid,
        pair_geometry=pair_geometry,
    )
    score_grid = full_exact_mixture_score_grid(
        diffusion=diffusion,
        x_grid=x_grid,
        gaussian_modes=np.asarray(gaussian_modes, dtype=np.float64),
        mode_sigmas=np.asarray(mode_sigmas, dtype=np.float64),
        std_dev=float(std_dev),
        t=int(t),
        device=str(device),
        chunk_size=int(chunk_size),
    )
    u = np.asarray(pair_geometry["u"], dtype=np.float64)
    beta_t = float(diffusion.betas[int(t)].detach().cpu().item())
    b_code_vector = (-0.5 * beta_t) * x_grid - float(beta_t) * score_grid
    b_code = np.sum(b_code_vector * u[:, None, None, :], axis=-1)
    return b_code, a_physical, z_physical


def summarize_region_metrics(
    *,
    b_code_grid: np.ndarray,
    a_physical: np.ndarray,
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    a_midpoint_radius: float,
    lambda_upper_bound_abs: float,
) -> dict:
    # Summarize the proposition checks using the stored reverse-process drift b(a,z,t).
    # Proposition 5.1 asks for:
    #   |b(a,t)| <= K(t) |a| on theta <= |a| <= 2 theta,
    #   b(a,t) <= -lambda_rep a on theta <= a <= a_*,
    #   b(a,t) >= -lambda_rep a on -a_* <= a <= -theta.
    # Dividing by a shows that both outer-region inequalities are checked by the single condition
    #   b(a,z,t) / a <= -lambda_rep.
    # The centerline check keeps a singleton z-axis for array-shape compatibility.
    nonzero_mask = np.abs(a_grid) > 1e-12
    k_lower_bound_abs = float(a_midpoint_radius)
    k_upper_bound_abs = float(2.0 * a_midpoint_radius)
    k_region_mask = (
        (np.abs(a_grid) >= float(k_lower_bound_abs) - 1e-12)
        & (np.abs(a_grid) <= float(k_upper_bound_abs) + 1e-12)
        & nonzero_mask
    )
    pos_mask = (
        (a_grid >= float(a_midpoint_radius) - 1e-12)
        & (a_grid <= float(lambda_upper_bound_abs) + 1e-12)
    )
    neg_mask = (
        (a_grid <= -float(a_midpoint_radius) + 1e-12)
        & (a_grid >= -float(lambda_upper_bound_abs) - 1e-12)
    )
    center_idx = int(np.argmin(np.abs(a_grid)))
    if not np.any(nonzero_mask) or not np.any(k_region_mask) or not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("The a-grid must include zero and cover both verification regions.")

    ratio_grid = np.full_like(b_code_grid, np.nan, dtype=np.float64)
    ratio_grid[..., nonzero_mask] = b_code_grid[..., nonzero_mask] / a_physical[..., nonzero_mask]

    pair_k = np.full((b_code_grid.shape[0],), np.nan, dtype=np.float64)
    pair_lambda_plus = np.full((b_code_grid.shape[0],), np.nan, dtype=np.float64)
    pair_lambda_minus = np.full((b_code_grid.shape[0],), np.nan, dtype=np.float64)

    if np.any(k_region_mask):
        pair_k = np.nanmax(np.abs(ratio_grid[:, :, k_region_mask]), axis=(1, 2))
    if np.any(pos_mask):
        pair_lambda_plus = -np.nanmax(ratio_grid[:, :, pos_mask], axis=(1, 2))
    if np.any(neg_mask):
        pair_lambda_minus = -np.nanmax(ratio_grid[:, :, neg_mask], axis=(1, 2))
    pair_lambda_rep = np.fmin(pair_lambda_plus, pair_lambda_minus)
    pair_bmid_abs = np.nanmax(np.abs(b_code_grid[:, :, center_idx]), axis=1)

    finite_lambda = np.where(np.isfinite(pair_lambda_rep))[0]
    if finite_lambda.size > 0:
        worst_pair_idx = int(finite_lambda[int(np.argmin(pair_lambda_rep[finite_lambda]))])
        pair_ratio_region = np.where(
            np.broadcast_to((pos_mask | neg_mask)[None, :], ratio_grid[worst_pair_idx].shape),
            ratio_grid[worst_pair_idx],
            -np.inf,
        )
        worst_flat = int(np.nanargmax(pair_ratio_region))
        worst_z_idx = int(np.unravel_index(worst_flat, pair_ratio_region.shape)[0])
    else:
        finite_k = np.where(np.isfinite(pair_k))[0]
        worst_pair_idx = int(finite_k[0]) if finite_k.size > 0 else 0
        pair_ratio_region = np.where(
            np.broadcast_to(k_region_mask[None, :], ratio_grid[worst_pair_idx].shape),
            np.abs(ratio_grid[worst_pair_idx]),
            -np.inf,
        )
        worst_flat = int(np.nanargmax(pair_ratio_region))
        worst_z_idx = int(np.unravel_index(worst_flat, pair_ratio_region.shape)[0])

    return {
        "ratio_grid": ratio_grid,
        "pair_k": pair_k,
        "pair_lambda_plus": pair_lambda_plus,
        "pair_lambda_minus": pair_lambda_minus,
        "pair_lambda_rep": pair_lambda_rep,
        "pair_bmid_abs": pair_bmid_abs,
        "center_idx": center_idx,
        "worst_pair_idx": worst_pair_idx,
        "worst_z_idx": worst_z_idx,
        "k_region_lower_abs": float(k_lower_bound_abs),
        "k_region_upper_abs": float(k_upper_bound_abs),
        "lambda_region_upper_abs": float(lambda_upper_bound_abs),
        "z_grid": np.asarray(z_grid, dtype=np.float64),
        "K_avg": float(np.nanmean(pair_k)) if np.any(np.isfinite(pair_k)) else float("nan"),
        "K_worst": float(np.nanmax(pair_k)) if np.any(np.isfinite(pair_k)) else float("nan"),
        "lambda_plus_avg": float(np.nanmean(pair_lambda_plus)) if np.any(np.isfinite(pair_lambda_plus)) else float("nan"),
        "lambda_plus_worst": float(np.nanmin(pair_lambda_plus)) if np.any(np.isfinite(pair_lambda_plus)) else float("nan"),
        "lambda_minus_avg": float(np.nanmean(pair_lambda_minus)) if np.any(np.isfinite(pair_lambda_minus)) else float("nan"),
        "lambda_minus_worst": float(np.nanmin(pair_lambda_minus)) if np.any(np.isfinite(pair_lambda_minus)) else float("nan"),
        "lambda_rep_avg": float(np.nanmean(pair_lambda_rep)) if np.any(np.isfinite(pair_lambda_rep)) else float("nan"),
        "lambda_rep_worst": float(np.nanmin(pair_lambda_rep)) if np.any(np.isfinite(pair_lambda_rep)) else float("nan"),
        "bmid_abs_avg": float(np.nanmean(pair_bmid_abs)) if np.any(np.isfinite(pair_bmid_abs)) else float("nan"),
        "bmid_abs_worst": float(np.nanmax(pair_bmid_abs)) if np.any(np.isfinite(pair_bmid_abs)) else float("nan"),
    }


def plot_worst_case_time_series(
    k_save_stem: str,
    lambda_save_stem: str,
    t_vals: np.ndarray,
    per_t_rows: list[dict],
    dpi: int,
    simplify_lambda_labels: bool = False,
) -> None:
    # plot the worst-case k and \lambda_{rep} summaries over time.
    k_worst = np.asarray([row["K_worst"] for row in per_t_rows], dtype=np.float64)
    lambda_rep_worst = np.asarray([row["lambda_rep_worst_t"] for row in per_t_rows], dtype=np.float64)
    lambda_rep_min = float(np.nanmin(lambda_rep_worst))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(t_vals, k_worst, linewidth=2.2, color="tab:blue", label=r"$K(t)$")
    ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$K(t)$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=TICK_FONTSIZE)
    ax.invert_xaxis()
    fig.tight_layout()
    save_figure(fig, k_save_stem, dpi=dpi)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    lambda_curve_label = r"$\lambda_{\mathrm{rep}}$" if bool(simplify_lambda_labels) else r"$\lambda_{\mathrm{rep}}(t)$"
    # The horizontal reference line is the single worst-case repulsion constant used in the paper-facing
    # summary, so label it directly as lambda_rep rather than as min_t lambda_rep(t).
    lambda_min_label = rf"$\lambda_{{\mathrm{{rep}}}} = {lambda_rep_min:.3f}$"
    lambda_ylabel = r"$\lambda_{\mathrm{rep}}$" if bool(simplify_lambda_labels) else r"$\lambda_{\mathrm{rep}}(t)$"
    ax.plot(t_vals, lambda_rep_worst, linewidth=2.2, color="tab:green", label=lambda_curve_label)
    ax.axhline(
        lambda_rep_min,
        linewidth=1.4,
        linestyle=":",
        color="black",
        alpha=0.85,
        label=lambda_min_label,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(lambda_ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=TICK_FONTSIZE)
    ax.invert_xaxis()
    fig.tight_layout()
    save_figure(fig, lambda_save_stem, dpi=dpi)


def coordinate_folder_name(drift_source: str, *, truncated: bool = False) -> str:
    # choose the artifact folder name for one fixed-midpoint branch.
    base = {
        "exact_full_mixture": "fixed_midpoint_exact_full_mixture",
    }[str(drift_source)]
    if truncated:
        return f"{base}_truncated"
    return base


def run_fixed_midpoint_coordinate_system(
    *,
    drift_source: str,
    diffusion,
    pair_geometry: dict,
    a_midpoint_radius: float,
    tau_target: int,
    num_a_per_side_theorem: int,
    save_folder: str,
    dpi: int,
    device: str = "cpu",
    model_chunk_size: int = 8192,
    a_upper_bound_abs: float | None = None,
    folder_name: str | None = None,
    full_gaussian_modes: np.ndarray | None = None,
    full_mode_sigmas: np.ndarray | None = None,
    full_std_dev: float | None = None,
    a_endpoint_extension: float = 0.0,
    paper_tau3_ddim: int | None = None,
    internal_tau3_raw_ddpm_steps: int | None = None,
) -> dict:
    # run one fixed-midpoint exp. l branch and write its summaries and figures.
    # This is the shared driver for the full-mixture exact-score Exp. L check.
    # The paper coordinate is the physical A_t=a on the closest-pair bisector line.
    if int(num_a_per_side_theorem) < 1:
        raise ValueError("tau3_closed_form_num_a_per_side must be at least 1.")
    ell_ref = reference_pair_length(pair_geometry)
    lambda_upper_bound_abs = (
        float(SEGMENT_HALF_FRACTION * ell_ref) if a_upper_bound_abs is None else float(a_upper_bound_abs)
    )
    k_upper_bound_abs = float(2.0 * a_midpoint_radius)
    grid_upper_bound_abs = max(float(lambda_upper_bound_abs), float(k_upper_bound_abs))
    # Use one shared theorem-facing a-grid so the proposition checks really are taken over the
    # configured intervals per side.
    a_grid = theorem_a_grid(
        a_midpoint_radius=float(a_midpoint_radius),
        lambda_upper_bound_abs=float(lambda_upper_bound_abs),
        num_intervals_per_side=int(num_a_per_side_theorem),
    )
    if not np.any(np.isclose(a_grid, 0.0)):
        raise ValueError("the theorem a-grid must contain a=0.")
    z_grid = np.asarray([0.0], dtype=np.float64)

    ell = np.asarray(pair_geometry["ell"], dtype=np.float64)
    num_pairs = int(ell.shape[0])

    # read the raw diffusion buffers once so the theorem-time loop can stay purely numpy.
    alpha_bar_raw = diffusion.alphas_prod[: int(tau_target) + 1].detach().cpu().numpy().astype(np.float64)
    beta_vals_raw = diffusion.betas[: int(tau_target) + 1].detach().cpu().numpy().astype(np.float64)
    t_vals = np.arange(int(tau_target) + 1, dtype=np.int64)
    code_s_vals = int(tau_target) - t_vals

    per_t_rows = []
    for t, code_s in zip(t_vals.tolist(), code_s_vals.tolist()):
        # theorem time t and code step s run in opposite directions.
        alpha_bar_t = float(alpha_bar_raw[int(t)])
        beta_t = float(beta_vals_raw[int(t)])
        if drift_source != "exact_full_mixture":
            raise ValueError(f"Unsupported drift_source {drift_source!r}.")
        if full_gaussian_modes is None or full_mode_sigmas is None or full_std_dev is None:
            raise ValueError("exact_full_mixture requires full_gaussian_modes, full_mode_sigmas, and full_std_dev.")
        b_grid, a_physical, z_physical = fixed_midpoint_full_exact_mixture_drift(
            a_grid=a_grid,
            pair_geometry=pair_geometry,
            diffusion=diffusion,
            gaussian_modes=np.asarray(full_gaussian_modes, dtype=np.float64),
            mode_sigmas=np.asarray(full_mode_sigmas, dtype=np.float64),
            std_dev=float(full_std_dev),
            t=int(t),
            device=str(device),
            chunk_size=int(model_chunk_size),
        )

        # summarize $K(t)$ and $\\lambda_{\\mathrm{rep}}(t)$ over the theorem region in $a$.
        metrics = summarize_region_metrics(
            b_code_grid=b_grid,
            a_physical=a_physical,
            a_grid=a_grid,
            z_grid=z_grid,
            a_midpoint_radius=float(a_midpoint_radius),
            lambda_upper_bound_abs=float(lambda_upper_bound_abs),
        )
        per_t_rows.append(
            {
                "t": int(t),
                "code_s": int(code_s),
                "alpha_bar": float(alpha_bar_t),
                "beta_t": float(beta_t),
                "K_avg": float(metrics["K_avg"]),
                "K_worst": float(metrics["K_worst"]),
                "lambda_plus_avg": float(metrics["lambda_plus_avg"]),
                "lambda_plus_worst": float(metrics["lambda_plus_worst"]),
                "lambda_minus_avg": float(metrics["lambda_minus_avg"]),
                "lambda_minus_worst": float(metrics["lambda_minus_worst"]),
                "lambda_rep_avg_t": float(metrics["lambda_rep_avg"]),
                "lambda_rep_worst_t": float(metrics["lambda_rep_worst"]),
                "bmid_abs_avg": float(metrics["bmid_abs_avg"]),
                "bmid_abs_worst": float(metrics["bmid_abs_worst"]),
                "worst_pair_index": int(metrics["worst_pair_idx"]),
            }
        )

    lambda_rep_avg_arr = np.asarray([row["lambda_rep_avg_t"] for row in per_t_rows], dtype=np.float64)
    lambda_rep_worst_arr = np.asarray([row["lambda_rep_worst_t"] for row in per_t_rows], dtype=np.float64)
    k_avg_arr = np.asarray([row["K_avg"] for row in per_t_rows], dtype=np.float64)
    k_worst_arr = np.asarray([row["K_worst"] for row in per_t_rows], dtype=np.float64)

    # the saved summary records the physical-coordinate convention explicitly so downstream reports stay unambiguous.
    summary = {
        "coordinate_system": "fixed_midpoint_centerline",
        "drift_source": str(drift_source),
        "time_convention": "theorem_t_with_forward_code_s",
        "stored_drift_convention": "b_grid stores the reverse-process drift b(a,t) from Eq. (4)",
        "tau3_target": int(tau_target),
        "paper_tau3_ddim": None if paper_tau3_ddim is None else int(paper_tau3_ddim),
        "internal_tau3_raw_ddpm_steps": None
        if internal_tau3_raw_ddpm_steps is None
        else int(internal_tau3_raw_ddpm_steps),
        "num_pairs": int(num_pairs),
        "a_intervals_per_side": int(num_a_per_side_theorem),
        "a_range": [-float(grid_upper_bound_abs), float(grid_upper_bound_abs)],
        "a_midpoint_radius": float(a_midpoint_radius),
        "k_region_abs_a": [float(a_midpoint_radius), float(2.0 * a_midpoint_radius)],
        "lambda_region_abs_a": [float(a_midpoint_radius), float(lambda_upper_bound_abs)],
        "a_units": "physical coordinate along the pair axis in normalized ambient space",
        "segment_half_fraction": float(SEGMENT_HALF_FRACTION),
        "a_endpoint_extension": float(a_endpoint_extension),
        "closest_pair_distance": float(pair_geometry["closest_pair_distance"]),
        "pair_length_min": float(np.min(ell)),
        "pair_length_max": float(np.max(ell)),
        "lambda_rep_avg_global": float(np.nanmin(lambda_rep_avg_arr)),
        "lambda_rep_worst_global": float(np.nanmin(lambda_rep_worst_arr)),
        "lambda_rep_avg_global_t": int(t_vals[int(np.nanargmin(lambda_rep_avg_arr))]),
        "lambda_rep_worst_global_t": int(t_vals[int(np.nanargmin(lambda_rep_worst_arr))]),
        "K_avg_integral": float(np.nansum(k_avg_arr)),
        "K_worst_integral": float(np.nansum(k_worst_arr)),
    }

    folder_name = folder_name or coordinate_folder_name(drift_source)
    coordinate_folder = os.path.join(save_folder, folder_name)
    simplify_lambda_labels = False
    os.makedirs(coordinate_folder, exist_ok=True)
    with open(os.path.join(coordinate_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, sort_keys=True)

    write_csv(
        os.path.join(coordinate_folder, "per_t_summary.csv"),
        [
            "t",
            "code_s",
            "alpha_bar",
            "beta_t",
            "K_avg",
            "K_worst",
            "lambda_plus_avg",
            "lambda_plus_worst",
            "lambda_minus_avg",
            "lambda_minus_worst",
            "lambda_rep_avg_t",
            "lambda_rep_worst_t",
            "bmid_abs_avg",
            "bmid_abs_worst",
            "worst_pair_index",
        ],
        per_t_rows,
    )
    plot_worst_case_time_series(
        k_save_stem=os.path.join(coordinate_folder, "K_worst_time_series"),
        lambda_save_stem=os.path.join(coordinate_folder, "lambda_rep_worst_time_series"),
        t_vals=t_vals.astype(np.float64),
        per_t_rows=per_t_rows,
        dpi=int(dpi),
        simplify_lambda_labels=bool(simplify_lambda_labels),
    )
    return summary


def valid_sample_radius_scale(protocol_tag: str, num_dims: int) -> float:
    # match the current fixed-protocol sample-validity scaling used in the geometry classifier.
    return float(geometry_radius_scale(int(num_dims), scale_by_sqrt_varpi=True))


def count_sign_consistent_successes(
    *,
    learned_model,
    diffusion,
    x_start: np.ndarray,
    tau_target: int,
    num_rollouts: int,
    target_mode_idx: int,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    invalid_sigma_multiple: float,
    protocol_tag: str,
    num_dims: int,
    device: str,
    chunk_size: int,
    sampling_seed: int,
    midpoint: np.ndarray | None = None,
    u_vec: np.ndarray | None = None,
    a_midpoint_radius: float | None = None,
) -> dict:
    # count sign-consistent ddpm rollouts from one empirical start point $a$.
    # a rollout is successful only if it ends as a valid true-mode sample on the same side of the pair.
    # interpolations are true-mode samples that land on the wrong side or at another mode; invalids are tracked separately.
    radius_scale = valid_sample_radius_scale(protocol_tag, int(num_dims))
    sigma_multiple = float(invalid_sigma_multiple) * float(radius_scale)
    mode_radii = mode_radii_from_sigmas(np.asarray(mode_sigmas, dtype=np.float64), sigma_multiple=sigma_multiple)
    pair_radii = pair_radii_from_sigmas(np.asarray(mode_sigmas, dtype=np.float64), sigma_multiple=sigma_multiple)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(sampling_seed))
    x_start = np.asarray(x_start, dtype=np.float64)
    modes = np.asarray(gaussian_modes, dtype=np.float64)

    success_count = 0
    true_mode_count = 0
    invalid_count = 0
    return_after_exit_count = 0
    total = 0
    for start in range(0, int(num_rollouts), int(chunk_size)):
        # reuse one start point for a chunk of ddpm rollouts.
        bs = min(int(chunk_size), int(num_rollouts) - start)
        x = torch.as_tensor(
            np.repeat(x_start[None, :], bs, axis=0),
            device=device,
            dtype=torch.float32,
        )
        returned_mask = np.zeros((bs,), dtype=bool)
        for t_eff in range(int(tau_target), -1, -1):
            eps_pred = call_model(learned_model, x, int(t_eff))
            x = ddpm_step_from_eps_with_generator(
                diffusion=diffusion,
                x_t=x,
                eps_pred=eps_pred,
                t_eff=int(t_eff),
                generator=generator,
            )
            if midpoint is not None and u_vec is not None and a_midpoint_radius is not None:
                x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
                a_proj = np.sum((x_np - midpoint[None, :]) * u_vec[None, :], axis=1)
                returned_mask |= np.abs(a_proj) <= float(a_midpoint_radius) + 1e-12
        final_samples = x.detach().cpu().numpy().astype(np.float64, copy=False)
        # classify the endpoints using the same protocol-dependent geometric validity rule as the main pipeline.
        geom = classify_samples_by_geometry(
            final_samples,
            modes,
            threshold=None,
            mode_radii=mode_radii,
            pair_radii=pair_radii,
        )
        true_mode_mask = np.asarray(geom["true_mode_mask"], dtype=bool)
        invalid_mask = np.asarray(geom["invalid_mask"], dtype=bool)
        closest_mode_idx = np.asarray(geom["closest_mode_idx"], dtype=np.int64)
        success_mask = true_mode_mask & (closest_mode_idx == int(target_mode_idx))
        success_count += int(np.sum(success_mask))
        true_mode_count += int(np.sum(true_mode_mask))
        invalid_count += int(np.sum(invalid_mask))
        return_after_exit_count += int(np.sum(returned_mask))
        total += int(bs)
    return {
        "total": int(total),
        "success_count": int(success_count),
        "failure_count": int(total - success_count),
        "true_mode_count": int(true_mode_count),
        "invalid_count": int(invalid_count),
        "interpolation_count": int(total - success_count - invalid_count),
        "return_after_exit_count": int(return_after_exit_count),
        "success_rate": float(success_count / max(total, 1)),
    }


def run_truncated_fixed_midpoint_coordinate_system(
    *,
    drift_source: str,
    diffusion,
    pair_geometry: dict,
    a_midpoint_radius: float,
    tau_target: int,
    num_a_per_side_theorem: int,
    save_folder: str,
    device: str = "cpu",
    model_chunk_size: int = 8192,
    a_upper_bound_abs: float,
    full_gaussian_modes: np.ndarray | None = None,
    full_mode_sigmas: np.ndarray | None = None,
    full_std_dev: float | None = None,
    a_endpoint_extension: float = 0.0,
    paper_tau3_ddim: int | None = None,
    internal_tau3_raw_ddpm_steps: int | None = None,
) -> dict:
    # rerun one fixed-midpoint branch on the empirical truncated region in $a$.
    # this is only used after the empirical threshold search identifies a symmetric admissible interval in the physical coordinate.
    summary = run_fixed_midpoint_coordinate_system(
        drift_source=str(drift_source),
        diffusion=diffusion,
        pair_geometry=pair_geometry,
        a_midpoint_radius=float(a_midpoint_radius),
        tau_target=int(tau_target),
        num_a_per_side_theorem=int(num_a_per_side_theorem),
        save_folder=save_folder,
        dpi=180,
        device=str(device),
        model_chunk_size=int(model_chunk_size),
        a_upper_bound_abs=float(a_upper_bound_abs),
        folder_name=coordinate_folder_name(drift_source, truncated=True),
        full_gaussian_modes=full_gaussian_modes,
        full_mode_sigmas=full_mode_sigmas,
        full_std_dev=full_std_dev,
        a_endpoint_extension=float(a_endpoint_extension),
        paper_tau3_ddim=None if paper_tau3_ddim is None else int(paper_tau3_ddim),
        internal_tau3_raw_ddpm_steps=None
        if internal_tau3_raw_ddpm_steps is None
        else int(internal_tau3_raw_ddpm_steps),
    )
    truncated_folder = os.path.join(save_folder, coordinate_folder_name(drift_source, truncated=True))
    truncated_summary = {
        **summary,
        "a_threshold_global_abs": float(a_upper_bound_abs),
        "resolved": bool(np.isfinite(summary["lambda_rep_worst_global"])),
        "min_t_lambda_rep_worst_truncated": float(summary["lambda_rep_worst_global"]),
        "all_lambda_rep_worst_positive_truncated": bool(summary["lambda_rep_worst_global"] > 0.0),
    }
    with open(os.path.join(truncated_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(truncated_summary), f, indent=2, sort_keys=True)
    return truncated_summary


def run_tau3_empirical_a_threshold_companion(
    *,
    diffusion,
    gaussian_modes: np.ndarray,
    mode_sigmas: np.ndarray,
    std_dev: float,
    learned_model,
    save_folder: str,
    protocol_tag: str,
    tau_target: int,
    theta_fraction=None,
    a_endpoint_extension=None,
    num_a: int = 50,
    num_a_per_side_theorem: int = DEFAULT_THEOREM_A_INTERVALS_PER_SIDE,
    num_rollouts_per_start: int = 10_000,
    invalid_sigma_multiple: float = 5.0,
    device: str = "cpu",
    chunk_size: int = 4096,
    sampling_seed: int = 42,
    paper_tau3_ddim: int | None = None,
    internal_tau3_raw_ddpm_steps: int | None = None,
) -> dict:
    # run the empirical exp. l threshold sweep directly in the paper's physical coordinate $a$.
    # this companion checks how far one can move away from the midpoint neighborhood before interpolations cease to be negligible.
    modes = np.asarray(gaussian_modes, dtype=np.float64)
    mode_sigmas = np.asarray(mode_sigmas, dtype=np.float64)
    pair_geometry = build_pair_geometry(modes)
    a_midpoint_fraction = a_midpoint_fraction_from_protocol(theta_fraction, protocol_tag=protocol_tag)
    a_endpoint_extension = a_endpoint_extension_from_protocol(a_endpoint_extension, protocol_tag=protocol_tag)
    a_midpoint_radius, a_segment_half_abs, ell_ref = physical_a_window(
        pair_geometry=pair_geometry,
        a_midpoint_fraction=float(a_midpoint_fraction),
    )
    a_endpoint_abs = float(0.5 * (float(ell_ref) + float(a_endpoint_extension)))

    if int(num_a) < 2:
        raise ValueError("tau3_empirical_num_a_per_side must be at least 2.")

    # scan outward from $|a|=0.15\ell$ toward the fixed endpoint $|a|=(\ell+\varepsilon)/2$ on each side separately.
    positive_candidates = np.linspace(float(a_midpoint_radius), float(a_endpoint_abs), int(num_a), dtype=np.float64)
    negative_candidates = np.linspace(-float(a_midpoint_radius), -float(a_endpoint_abs), int(num_a), dtype=np.float64)
    empirical_root = os.path.join(save_folder, "empirical_a_threshold")
    os.makedirs(empirical_root, exist_ok=True)

    rows = []
    for side, candidates in (("positive", positive_candidates), ("negative", negative_candidates)):
        for cand_idx, a_val in enumerate(candidates.tolist()):
            # aggregate failures over all closest pairs for this one side and this one $a$ value.
            total_success = 0
            total_failures = 0
            total_rollouts = 0
            true_mode_total = 0
            invalid_total = 0
            return_after_exit_total = 0
            for pair_idx in range(int(len(pair_geometry["pair_i"]))):
                midpoint = pair_geometry["midpoint_fixed"][pair_idx]
                u_vec = pair_geometry["u"][pair_idx]
                x_start = midpoint + float(a_val) * u_vec
                target_mode_idx = int(
                    pair_geometry["pair_j"][pair_idx] if float(a_val) >= 0.0 else pair_geometry["pair_i"][pair_idx]
                )
                counts = count_sign_consistent_successes(
                    learned_model=learned_model,
                    diffusion=diffusion,
                    x_start=x_start,
                    tau_target=int(tau_target),
                    num_rollouts=int(num_rollouts_per_start),
                    target_mode_idx=int(target_mode_idx),
                    gaussian_modes=modes,
                    mode_sigmas=mode_sigmas,
                    invalid_sigma_multiple=float(invalid_sigma_multiple),
                    protocol_tag=str(protocol_tag),
                    num_dims=int(modes.shape[1]),
                    device=str(device),
                    chunk_size=int(chunk_size),
                    sampling_seed=int(sampling_seed + 100_000 * cand_idx + pair_idx),
                    midpoint=np.asarray(midpoint, dtype=np.float64),
                    u_vec=np.asarray(u_vec, dtype=np.float64),
                    a_midpoint_radius=float(a_midpoint_radius),
                )
                total_success += int(counts["success_count"])
                total_failures += int(counts["failure_count"])
                total_rollouts += int(counts["total"])
                true_mode_total += int(counts["true_mode_count"])
                invalid_total += int(counts["invalid_count"])
                return_after_exit_total += int(counts["return_after_exit_count"])
            interpolation_total = int(total_rollouts - total_success - invalid_total)
            interpolation_rate = float(interpolation_total / max(total_rollouts, 1))
            acceptable_interpolation = bool(interpolation_rate < float(MAX_INTERPOLATION_RATE_FOR_ASTAR))
            zero_return_after_exit = bool(return_after_exit_total == 0)
            acceptable_a_star_candidate = bool(acceptable_interpolation and zero_return_after_exit)
            rows.append(
                {
                    "side": str(side),
                    "a_value": float(a_val),
                    "a_abs": float(abs(a_val)),
                    "success_count_total": int(total_success),
                    "failure_count_total": int(total_failures),
                    "true_mode_count_total": int(true_mode_total),
                    "invalid_count_total": int(invalid_total),
                    "interpolation_count_total": int(interpolation_total),
                    "rollout_count_total": int(total_rollouts),
                    "success_rate_total": float(total_success / max(total_rollouts, 1)),
                    "interpolation_rate_total": float(interpolation_rate),
                    "return_after_exit_count_total": int(return_after_exit_total),
                    "return_after_exit_rate_total": float(return_after_exit_total / max(total_rollouts, 1)),
                    "acceptable_interpolation": bool(acceptable_interpolation),
                    "zero_return_after_exit": bool(zero_return_after_exit),
                    "acceptable_a_star_candidate": bool(acceptable_a_star_candidate),
                    "zero_interpolation": bool(interpolation_total == 0),
                }
            )

    pos_rows = [row for row in rows if row["side"] == "positive"]
    neg_rows = [row for row in rows if row["side"] == "negative"]

    def first_acceptable_threshold(side_rows: list[dict]):
        # identify the first physical coordinate where interpolations become negligible and return-after-exit disappears.
        # exp. l uses this as the empirical a* boundary of the region where return-to-midpoint after exit is still plausible.
        for row in side_rows:
            if bool(row["acceptable_a_star_candidate"]):
                return row
        return None

    threshold_positive = first_acceptable_threshold(pos_rows)
    threshold_negative = first_acceptable_threshold(neg_rows)
    acceptable_positive = [row for row in pos_rows if bool(row["acceptable_interpolation"])]
    acceptable_negative = [row for row in neg_rows if bool(row["acceptable_interpolation"])]
    acceptable_positive_astar = [row for row in pos_rows if bool(row["acceptable_a_star_candidate"])]
    acceptable_negative_astar = [row for row in neg_rows if bool(row["acceptable_a_star_candidate"])]
    largest_acceptable_positive = max(acceptable_positive, key=lambda row: float(row["a_value"])) if acceptable_positive else None
    largest_acceptable_negative = min(acceptable_negative, key=lambda row: float(row["a_value"])) if acceptable_negative else None
    largest_acceptable_positive_astar = (
        max(acceptable_positive_astar, key=lambda row: float(row["a_value"])) if acceptable_positive_astar else None
    )
    largest_acceptable_negative_astar = (
        min(acceptable_negative_astar, key=lambda row: float(row["a_value"])) if acceptable_negative_astar else None
    )

    a_threshold_global_abs = None
    if threshold_positive is not None and threshold_negative is not None:
        # use the larger side threshold so the symmetric cutoff covers the whole pre-asymptotic region on both sides.
        a_threshold_global_abs = max(float(threshold_positive["a_abs"]), float(threshold_negative["a_abs"]))

    summary = {
        "tau3_target": int(tau_target),
        "paper_tau3_ddim": None if paper_tau3_ddim is None else int(paper_tau3_ddim),
        "internal_tau3_raw_ddpm_steps": None
        if internal_tau3_raw_ddpm_steps is None
        else int(internal_tau3_raw_ddpm_steps),
        "a_midpoint_radius": float(a_midpoint_radius),
        "a_midpoint_fraction": float(a_midpoint_fraction),
        "a_units": "physical coordinate along the pair axis in normalized ambient space",
        "a_endpoint_extension": float(a_endpoint_extension),
        "num_a_per_side": int(num_a),
        "num_rollouts_per_start": int(num_rollouts_per_start),
        "num_closest_pairs": int(len(pair_geometry["pair_i"])),
        "rollout_count_total_per_candidate_side": int(len(pair_geometry["pair_i"])) * int(num_rollouts_per_start),
        "closest_pair_distance": float(ell_ref),
        "a_candidate_values_positive": [float(row["a_value"]) for row in pos_rows],
        "a_candidate_values_negative": [float(row["a_value"]) for row in neg_rows],
        "a_threshold_positive": None if threshold_positive is None else float(threshold_positive["a_value"]),
        "a_threshold_negative": None if threshold_negative is None else float(threshold_negative["a_value"]),
        "a_star_positive": None if threshold_positive is None else float(threshold_positive["a_value"]),
        "a_star_negative": None if threshold_negative is None else float(threshold_negative["a_value"]),
        "largest_acceptable_interpolation_candidate_positive": None if largest_acceptable_positive is None else float(largest_acceptable_positive["a_value"]),
        "largest_acceptable_interpolation_candidate_negative": None if largest_acceptable_negative is None else float(largest_acceptable_negative["a_value"]),
        "largest_acceptable_a_star_candidate_positive": None
        if largest_acceptable_positive_astar is None
        else float(largest_acceptable_positive_astar["a_value"]),
        "largest_acceptable_a_star_candidate_negative": None
        if largest_acceptable_negative_astar is None
        else float(largest_acceptable_negative_astar["a_value"]),
        "a_threshold_global_abs": None if a_threshold_global_abs is None else float(a_threshold_global_abs),
        "a_star_global_abs": None if a_threshold_global_abs is None else float(a_threshold_global_abs),
        "a_threshold_scalar": None if a_threshold_global_abs is None else float(a_threshold_global_abs),
        "interpolation_rate_threshold_for_a_star": float(MAX_INTERPOLATION_RATE_FOR_ASTAR),
        "threshold_definition": (
            "first candidate on each side, scanning outward from |a|=0.15 ell toward the fixed endpoint "
            "|a|=(ell+epsilon)/2, whose interpolation rate is strictly below 0.001 and whose "
            "return-after-exit count is exactly zero"
        ),
    }

    with open(os.path.join(empirical_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, sort_keys=True)
    write_csv(
        os.path.join(empirical_root, "success_vs_a.csv"),
        [
            "side",
            "a_value",
            "a_abs",
            "success_count_total",
            "failure_count_total",
            "true_mode_count_total",
            "invalid_count_total",
            "interpolation_count_total",
            "rollout_count_total",
            "success_rate_total",
            "interpolation_rate_total",
            "return_after_exit_count_total",
            "return_after_exit_rate_total",
            "acceptable_interpolation",
            "zero_return_after_exit",
            "acceptable_a_star_candidate",
            "zero_interpolation",
        ],
        rows,
    )
    with open(os.path.join(empirical_root, "success_vs_a.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"rows": rows, "summary": summary}), f, indent=2, sort_keys=True)

    outputs = {
        "summary": summary,
        "rows": rows,
    }
    if a_threshold_global_abs is not None:
        outputs["fixed_midpoint_exact_full_mixture_truncated"] = run_truncated_fixed_midpoint_coordinate_system(
            drift_source="exact_full_mixture",
            diffusion=diffusion,
            pair_geometry=pair_geometry,
            a_midpoint_radius=float(a_midpoint_radius),
            tau_target=int(tau_target),
            num_a_per_side_theorem=int(num_a_per_side_theorem),
            save_folder=empirical_root,
            device=str(device),
            model_chunk_size=int(chunk_size),
            a_upper_bound_abs=float(a_threshold_global_abs),
            full_gaussian_modes=modes,
            full_mode_sigmas=np.asarray(mode_sigmas, dtype=np.float64),
            full_std_dev=float(std_dev),
            a_endpoint_extension=float(a_endpoint_extension),
            paper_tau3_ddim=None if paper_tau3_ddim is None else int(paper_tau3_ddim),
            internal_tau3_raw_ddpm_steps=None
            if internal_tau3_raw_ddpm_steps is None
            else int(internal_tau3_raw_ddpm_steps),
        )
    return outputs
