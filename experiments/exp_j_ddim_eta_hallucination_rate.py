import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.artifact_io import write_csv
from common.gaussian_mixture_2d import classify_samples_by_geometry
from common.reverse_solvers import build_reverse_solver_stepper


def sample_final_with_solver(
    *,
    model,
    diffusion,
    num_samples: int,
    num_dims: int,
    device: str,
    sampling_mode: str,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
    ddim_eta: float,
    chunk_size: int = 25000,
):
    # sample final points from one solver configuration.
    model.eval()
    seq, step_fn = build_reverse_solver_stepper(
        diffusion=diffusion,
        sampling_mode=str(sampling_mode).lower(),
        timesteps=int(timesteps),
        ddim_steps=int(ddim_steps),
        skip_type=str(skip_type),
        ddim_eta=float(ddim_eta),
    )

    chunk_size = max(int(chunk_size), 1)
    outputs = []
    with torch.no_grad():
        for start in range(0, int(num_samples), chunk_size):
            # draw fresh noise for each chunk of final samples.
            bs = min(chunk_size, int(num_samples) - start)
            x = torch.randn(bs, int(num_dims), device=device)
            prev_x0 = None
            prev_h = None
            for s, t_eff in enumerate(seq):
                t_next = int(seq[s + 1]) if (s + 1) < len(seq) else -1
                x, prev_x0, prev_h = step_fn(
                    model=model,
                    x_t=x,
                    t_eff=int(t_eff),
                    t_next=int(t_next),
                    prev_x0=prev_x0,
                    prev_h=prev_h,
                )
            outputs.append(x.detach().cpu().numpy())
    model.train()
    return np.concatenate(outputs, axis=0)


def geometry_rates(final_samples, gaussian_modes, classification_cfg: dict) -> dict:
    # compute geometry-based rates for one sample cloud.
    geom = classify_samples_by_geometry(
        final_samples,
        gaussian_modes,
        threshold=classification_cfg.get("threshold"),
        mode_radii=classification_cfg.get("mode_radii"),
        pair_radii=classification_cfg.get("pair_radii"),
    )
    true_mode_mask = np.asarray(geom["true_mode_mask"], dtype=bool)
    interpolated_mask = np.asarray(geom["interpolated_mask"], dtype=bool)
    invalid_mask = np.asarray(geom["invalid_mask"], dtype=bool)

    total = int(final_samples.shape[0])
    true_mode_count = int(np.sum(true_mode_mask))
    interpolated_count = int(np.sum(interpolated_mask))
    invalid_count = int(np.sum(invalid_mask))
    hallucination_count = int(interpolated_count + invalid_count)

    return {
        "num_samples": total,
        "true_mode_count": true_mode_count,
        "interpolated_count": interpolated_count,
        "invalid_count": invalid_count,
        "hallucination_count": hallucination_count,
        "true_mode_rate": 100.0 * true_mode_count / max(total, 1),
        "interpolation_rate": 100.0 * interpolated_count / max(total, 1),
        "artifact_rate": 100.0 * invalid_count / max(total, 1),
        "hallucination_rate": 100.0 * hallucination_count / max(total, 1),
    }


def plot_eta_sweep(result_dict: dict, save_path: str) -> None:
    # plot the exp j hallucination curve over \eta.
    rows = list(result_dict.get("results", []))
    eta_vals = np.asarray([float(row["eta"]) for row in rows], dtype=np.float64)
    interp = np.asarray([float(row["interpolation_rate"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eta_vals, interp, "o-", linewidth=2.0, markersize=6)
    ax.set_xlabel(r"DDIM noise level $\eta$")
    ax.set_ylabel("Hallucination rate (%)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(float(np.min(eta_vals)) - 0.02, float(np.max(eta_vals)) + 0.02)
    ax.set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_ddim_eta_hallucination_rate_sweep(
    *,
    dataset_name: str,
    artifact_run_name: str,
    protocol_tag: str | None,
    source_run_name: str,
    diffusion,
    model,
    gaussian_modes,
    classification_cfg: dict,
    num_samples: int,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
    eta_values,
    save_folder: str,
    device: str,
    num_dims: int = 2,
    ddpm_baseline_model=None,
    ddpm_baseline_diffusion=None,
    ddpm_baseline_run_name: str | None = None,
    chunk_size: int = 25000,
):
    # run exp j by sweeping \eta and comparing against a ddpm baseline.
    os.makedirs(save_folder, exist_ok=True)
    eta_values = [float(v) for v in eta_values]

    rows = []
    for eta in eta_values:
        final_samples = sample_final_with_solver(
            model=model,
            diffusion=diffusion,
            num_samples=int(num_samples),
            num_dims=int(num_dims),
            device=device,
            sampling_mode="ddim",
            timesteps=int(timesteps),
            ddim_steps=int(ddim_steps),
            skip_type=str(skip_type),
            ddim_eta=float(eta),
            chunk_size=int(chunk_size),
        )
        stats = geometry_rates(final_samples, gaussian_modes, classification_cfg)
        rows.append(
            {
                "sampler": "ddim",
                "eta": float(eta),
                **stats,
            }
        )

    baseline_model = ddpm_baseline_model if ddpm_baseline_model is not None else model
    baseline_diffusion = ddpm_baseline_diffusion if ddpm_baseline_diffusion is not None else diffusion
    ddpm_stats = geometry_rates(
        sample_final_with_solver(
            model=baseline_model,
            diffusion=baseline_diffusion,
            num_samples=int(num_samples),
            num_dims=int(num_dims),
            device=device,
            sampling_mode="ddpm",
            timesteps=int(timesteps),
            ddim_steps=int(ddim_steps),
            skip_type=str(skip_type),
            ddim_eta=0.0,
            chunk_size=int(chunk_size),
        ),
        gaussian_modes,
        classification_cfg,
    )
    ddpm_row = {
        "sampler": "ddpm",
        "eta": None,
        **ddpm_stats,
    }

    payload = {
        "experiment": "exp_j_ddim_eta_hallucination_rate",
        "protocol": None if protocol_tag in (None, "") else str(protocol_tag),
        "dataset_name": str(dataset_name),
        "artifact_run_name": str(artifact_run_name),
        "source_run_name": str(source_run_name),
        "ddpm_baseline_run_name": str(ddpm_baseline_run_name or source_run_name),
        "timesteps": int(timesteps),
        "ddim_steps": int(ddim_steps),
        "skip_type": str(skip_type),
        "eta_values": eta_values,
        "num_samples": int(num_samples),
        "num_dims": int(num_dims),
        "classification": {
            "classification_mode": classification_cfg.get("classification_mode"),
            "threshold": classification_cfg.get("threshold"),
            "reference_radius_policy": classification_cfg.get("reference_radius_policy"),
            "reference_radius_value": classification_cfg.get("reference_radius_value"),
            "reference_radius_base_value": classification_cfg.get("reference_radius_base_value"),
            "reference_sigma_clip": classification_cfg.get("reference_sigma_clip"),
            "scale_by_sqrt_varpi": bool(classification_cfg.get("scale_by_sqrt_varpi", False)),
            "reference_radius_scale_factor": classification_cfg.get("reference_radius_scale_factor"),
            "reference_radius_num_dims": classification_cfg.get("reference_radius_num_dims"),
        },
        "results": rows,
        "ddpm_baseline": ddpm_row,
    }

    csv_rows = rows + [ddpm_row]
    fieldnames = [
        "sampler",
        "eta",
        "num_samples",
        "true_mode_count",
        "interpolated_count",
        "invalid_count",
        "hallucination_count",
        "true_mode_rate",
        "interpolation_rate",
        "artifact_rate",
        "hallucination_rate",
    ]
    csv_path = os.path.join(save_folder, "exp_j_ddim_eta_hallucination_rate_long.csv")
    json_path = os.path.join(save_folder, "exp_j_ddim_eta_hallucination_rate.json")
    pdf_path = os.path.join(save_folder, "exp_j_ddim_eta_hallucination_rate.pdf")
    write_csv(csv_path, fieldnames, csv_rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_eta_sweep(payload, pdf_path)
    payload["csv_path"] = csv_path
    payload["json_path"] = json_path
    payload["pdf_path"] = pdf_path
    return payload
