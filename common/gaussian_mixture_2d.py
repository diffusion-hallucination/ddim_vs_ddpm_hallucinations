import json
import math
import os
from typing import Any

import numpy as np


SPEC_VERSION = "gaussian_mixture_2d/v1"


def as_float_array(x, *, shape_last: int | None = None) -> np.ndarray:
    # as float array.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    arr = np.asarray(x, dtype=np.float64)
    if shape_last is not None and arr.shape[-1] != shape_last:
        raise ValueError(f"Expected last dimension {shape_last}, got {arr.shape}")
    return arr


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    # normalize weights.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("mode_weights cannot be empty")
    if np.any(w < 0.0):
        raise ValueError("mode_weights must be non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("mode_weights must sum to a positive value")
    return w / total


def weighted_counts(num_samples: int, weights: np.ndarray) -> np.ndarray:
    # weighted counts.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    weights = normalize_weights(weights)
    exact = weights * float(num_samples)
    counts = np.floor(exact).astype(np.int64)
    remainder = int(num_samples - np.sum(counts))
    if remainder > 0:
        frac = exact - counts.astype(np.float64)
        order = np.argsort(-frac)
        counts[order[:remainder]] += 1
    return counts




def allocate_mode_counts(num_samples: int, weights: np.ndarray) -> np.ndarray:
    # allocate mode counts.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    return weighted_counts(int(num_samples), np.asarray(weights, dtype=np.float64))


def build_base_layout(base_layout: str = "grid25", mode_spacing: float = 2.0) -> np.ndarray:
    # build the 25-mode grid layout used by the paper's 2D Gaussian experiments.
    layout = str(base_layout).lower()
    if layout != "grid25":
        raise ValueError("Only base_layout='grid25' is supported.")
    pts = np.asarray([(i, j) for i in range(-2, 3) for j in range(-2, 3)], dtype=np.float64)
    return float(mode_spacing) * pts


def mixture_normalization_scale(raw_means: np.ndarray, raw_sigmas: np.ndarray, weights: np.ndarray) -> float:
    # mixture normalization scale.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    means = as_float_array(raw_means, shape_last=2)
    sigmas = np.asarray(raw_sigmas, dtype=np.float64).reshape(-1)
    weights = normalize_weights(weights)
    if means.shape[0] != sigmas.shape[0] or means.shape[0] != weights.shape[0]:
        raise ValueError("raw_means, raw_sigmas, and weights must agree in length")
    mean = np.sum(weights[:, None] * means, axis=0)
    second = np.sum(weights * np.sum(means * means, axis=1))
    trace_cov_modes = second - float(np.dot(mean, mean))
    trace_within = 2.0 * float(np.sum(weights * (sigmas ** 2)))
    scale2 = max((trace_cov_modes + trace_within) / 2.0, 1e-12)
    return math.sqrt(scale2)


def finalize_mixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    # finalize mixture spec.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    raw_means = as_float_array(spec["raw_means"], shape_last=2)
    raw_sigmas = np.asarray(spec["raw_sigmas"], dtype=np.float64).reshape(-1)
    mode_weights = normalize_weights(spec["mode_weights"])
    norm_scale = mixture_normalization_scale(raw_means, raw_sigmas, mode_weights)
    normalized_means = raw_means / norm_scale
    normalized_sigmas = raw_sigmas / norm_scale
    global_sigma_raw = float(math.sqrt(np.sum(mode_weights * (raw_sigmas ** 2))))
    global_sigma_normalized = float(math.sqrt(np.sum(mode_weights * (normalized_sigmas ** 2))))
    out = dict(spec)
    out["spec_version"] = SPEC_VERSION
    out["raw_means"] = raw_means.tolist()
    out["raw_sigmas"] = raw_sigmas.tolist()
    out["mode_weights"] = mode_weights.tolist()
    out["normalization_scale"] = float(norm_scale)
    out["normalized_means"] = normalized_means.tolist()
    out["normalized_sigmas"] = normalized_sigmas.tolist()
    out["global_sigma_raw"] = global_sigma_raw
    out["global_sigma_normalized"] = global_sigma_normalized
    return out


def build_mixture_spec_from_dataset_cfg(dataset_cfg) -> dict[str, Any]:
    # build the normalized gaussian-mixture specification that fixes the mode geometry for a run.
    cfg = dataset_cfg
    base_layout = str(cfg.get("base_layout", "grid25"))
    raw_means = build_base_layout(
        base_layout=base_layout,
        mode_spacing=float(cfg.get("mode_spacing", 2.0)),
    )
    n_modes = int(raw_means.shape[0])
    global_sigma = float(cfg.get("global_sigma", cfg.get("stdev", 0.02)))
    raw_sigmas = np.full((n_modes,), global_sigma, dtype=np.float64)
    mode_weights = np.full((n_modes,), 1.0 / float(n_modes), dtype=np.float64)

    spec = {
        "dataset_name": str(cfg.get("name", "gaussian_mixture_2d")),
        "kind": str(cfg.get("kind", "gaussian_mixture_2d")),
        "base_layout": base_layout,
        "mode_spacing": float(cfg.get("mode_spacing", 2.0)),
        "raw_means": raw_means.tolist(),
        "raw_sigmas": raw_sigmas.tolist(),
        "mode_weights": np.asarray(mode_weights, dtype=np.float64).tolist(),
    }
    return finalize_mixture_spec(spec)


def save_mixture_spec(spec: dict[str, Any], save_path: str) -> None:
    # save mixture spec for the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True)


def load_mixture_spec(spec_path: str) -> dict[str, Any]:
    # load mixture spec needed by the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec


def load_mixture_spec_for_run(run_dir: str) -> dict[str, Any] | None:
    # load mixture spec for run needed by the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    spec_path = os.path.join(run_dir, "mixture_spec.json")
    if not os.path.exists(spec_path):
        return None
    return load_mixture_spec(spec_path)


def mode_radii_from_sigmas(mode_sigmas, sigma_multiple: float = 3.0) -> np.ndarray:
    # mode radii from sigmas.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    return float(sigma_multiple) * np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)


def pair_radii_from_sigmas(mode_sigmas, sigma_multiple: float = 3.0) -> np.ndarray:
    # pair radii from sigmas.
    # this supports the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    sig = np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    return float(sigma_multiple) * 0.5 * (sig[:, None] + sig[None, :])


def geometry_radius_scale(num_dims: int, *, scale_by_sqrt_varpi: bool = False) -> float:
    # convert the geometric scaling rule into the effective radius factor.
    # for low-dimensional gaussian studies we keep the threshold at direct normalized $k\sigma$,
    # while high-dimensional runs can opt into $\\sqrt{\\varpi}$ scaling.
    dims = int(num_dims)
    if dims <= 0:
        raise ValueError(f"num_dims must be positive, got {dims}")
    if not bool(scale_by_sqrt_varpi):
        return 1.0
    if dims <= 10:
        return 1.0
    return math.sqrt(float(dims))


def sigma_shell_threshold_multiple(
    num_dims: int,
    *,
    shell_offset: float = 4.0,
) -> float:
    # compute the shell-aware threshold $\\sigma(\\sqrt{\\varpi} + C)$ in normalized coordinates.
    # because the dataset normalization divides both the radius and $\\sigma$ by the same factor,
    # the normalized threshold keeps the same dimensionless multiplier $\\sqrt{\\varpi} + C$.
    # This is the high-dimensional replacement for direct $k\\sigma$ clipping: genuine Gaussian samples
    # concentrate on a shell of radius about $\\sigma \\sqrt{\\varpi}$, with $O(\\sigma)$ radial fluctuation,
    # so the fixed-pipeline classifier switches to $\\sigma(\\sqrt{\\varpi} + 4)$ once the dimension is large enough.
    dims = int(num_dims)
    if dims <= 0:
        raise ValueError(f"num_dims must be positive, got {dims}")
    return math.sqrt(float(dims)) + float(shell_offset)


def sample_classification_config(
    mode_sigmas,
    *,
    classification_mode: str = "sigma_geometry",
    invalid_sigma_multiple: float = 5.0,
    num_dims: int = 2,
    scale_by_sqrt_varpi: bool = False,
    shell_offset: float | None = 4.0,
    shell_dimension_threshold: int = 10,
) -> dict[str, Any]:
    # sample classification config for the normalized gaussian-mixture geometry used throughout the paper's Gaussian experiments.
    sig = np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    mode = str(classification_mode).strip().lower()
    num_dims = int(num_dims)
    if num_dims <= 0:
        raise ValueError(f"num_dims must be positive, got {num_dims}")
    radius_scale = geometry_radius_scale(num_dims, scale_by_sqrt_varpi=bool(scale_by_sqrt_varpi))
    effective_scale_flag = bool(scale_by_sqrt_varpi) and int(num_dims) > 10
    shell_offset_value = None if shell_offset is None else float(shell_offset)
    shell_dim_threshold = int(shell_dimension_threshold)
    use_shell_threshold = (
        mode == "sigma_geometry"
        and shell_offset_value is not None
        and num_dims > shell_dim_threshold
    )
    if mode == "sigma_geometry":
        sigma_multiple = float(invalid_sigma_multiple)
        effective_sigma_multiple = sigma_multiple * radius_scale
        radius_policy = "sigma_multiple"
        radius_base_value = sigma_multiple
        radius_scale_factor = radius_scale
        sigma_clip = sigma_multiple
        if use_shell_threshold:
            # In higher dimensions we stop treating "valid" as a direct k-sigma ball and instead use the
            # shell-aware threshold sigma (sqrt(varpi) + 4). This keeps the classifier aligned with where
            # true Gaussian samples actually concentrate, rather than with a low-dimensional heuristic.
            effective_sigma_multiple = sigma_shell_threshold_multiple(
                num_dims,
                shell_offset=float(shell_offset_value),
            )
            radius_policy = "sigma_shell_plus_offset"
            radius_base_value = math.sqrt(float(num_dims))
            radius_scale_factor = 1.0
            sigma_clip = effective_sigma_multiple
            effective_scale_flag = False
        return {
            "classification_mode": mode,
            "threshold": None,
            "mode_radii": mode_radii_from_sigmas(sig, sigma_multiple=effective_sigma_multiple),
            "pair_radii": pair_radii_from_sigmas(sig, sigma_multiple=effective_sigma_multiple),
            "reference_radius_policy": radius_policy,
            "reference_radius_value": effective_sigma_multiple,
            "reference_radius_base_value": radius_base_value,
            "reference_sigma_clip": sigma_clip,
            "scale_by_sqrt_varpi": effective_scale_flag,
            "reference_radius_scale_factor": radius_scale_factor,
            "reference_radius_num_dims": num_dims,
            "reference_radius_shell_offset": shell_offset_value,
            "reference_radius_shell_dimension_threshold": shell_dim_threshold,
        }
    raise ValueError("classification_mode must be 'sigma_geometry'")


def classify_samples_by_geometry(
    final_samples,
    true_modes,
    *,
    threshold: float | None = None,
    mode_radii=None,
    pair_radii=None,
) -> dict[str, Any]:
    # classify final samples into true-mode, interpolated, and invalid outcomes using the paper's geometric rules.
    x = np.asarray(final_samples, dtype=np.float64)
    modes = np.asarray(true_modes, dtype=np.float64)
    if x.ndim != 2 or modes.ndim != 2:
        raise ValueError("final_samples and true_modes must both have shape (N, D)/(M, D)")
    diff = x[:, None, :] - modes[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    closest_idx = np.argmin(dists, axis=1)
    closest_dist = dists[np.arange(x.shape[0]), closest_idx]

    top2 = np.argsort(dists, axis=1)[:, :2]
    a = modes[top2[:, 0]]
    b = modes[top2[:, 1]]
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    ab2_safe = np.where(ab2 <= 0.0, 1.0, ab2)
    t = np.sum((x - a) * ab, axis=1) / ab2_safe
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    distance_to_segment = np.linalg.norm(x - proj, axis=1)

    if mode_radii is None:
        if threshold is None:
            raise ValueError("Either threshold or mode_radii must be provided")
        mode_radius_used = np.full((x.shape[0],), float(threshold), dtype=np.float64)
    else:
        mode_radii = np.asarray(mode_radii, dtype=np.float64).reshape(-1)
        mode_radius_used = mode_radii[closest_idx]

    if pair_radii is None:
        if threshold is None:
            raise ValueError("Either threshold or pair_radii must be provided")
        pair_radius_used = np.full((x.shape[0],), float(threshold), dtype=np.float64)
    else:
        pair_radii = np.asarray(pair_radii, dtype=np.float64)
        pair_radius_used = pair_radii[top2[:, 0], top2[:, 1]]

    true_mode_mask = closest_dist <= mode_radius_used
    invalid_mask = distance_to_segment > pair_radius_used
    interpolated_mask = ~(true_mode_mask | invalid_mask)

    bucket = np.full((x.shape[0],), "invalid", dtype=object)
    for idx in np.where(true_mode_mask)[0]:
        bucket[idx] = f"mode_{int(closest_idx[idx])}"
    for idx in np.where(interpolated_mask)[0]:
        i, j = sorted((int(top2[idx, 0]), int(top2[idx, 1])))
        bucket[idx] = f"segment_{i}_{j}"

    return {
        "closest_dist": closest_dist,
        "closest_mode_idx": closest_idx,
        "nearest_pair_idx": top2,
        "distance_to_segment": distance_to_segment,
        "mode_radius_used": mode_radius_used,
        "pair_radius_used": pair_radius_used,
        "true_mode_mask": true_mode_mask,
        "interpolated_mask": interpolated_mask,
        "invalid_mask": invalid_mask,
        "bucket": bucket,
    }
