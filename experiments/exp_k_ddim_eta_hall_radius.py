import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common.artifact_io import write_csv
from experiments.exp_e_hall_with_radius import (
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    compute_hallucination_with_radius,
)


def plot_eta_cumulative_fixed_tau(
    results_by_eta: dict[float, dict],
    *,
    tau_target: int,
    save_path: str,
    band: str = "sem",
    alpha: float = 0.22,
) -> None:
    # Plot Exp. K as one eta overlay over midpoint radius for one fixed tau_3.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eta_keys = sorted(float(v) for v in results_by_eta.keys())
    if not eta_keys:
        raise ValueError("Exp. K requires at least one eta value.")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for eta in eta_keys:
        res = results_by_eta[eta]
        tau_vals = [int(t) for t in res["tau_vals"]]
        if int(tau_target) not in tau_vals:
            raise ValueError(f"tau_target={tau_target} not in tau_vals={tau_vals} for eta={eta}")
        t_idx = tau_vals.index(int(tau_target))

        r_vals = np.asarray(res["r_percentage"], dtype=np.float64)
        display_r_by_tau = res.get("display_r_percentage_by_tau", {}) or {}
        tau_display = display_r_by_tau.get(str(int(tau_target)))
        if tau_display is not None:
            tau_display_arr = np.asarray(tau_display, dtype=np.float64)
            if tau_display_arr.shape == r_vals.shape:
                r_vals = tau_display_arr

        hall_mean = np.asarray(res["hall_mean"], dtype=np.float64)
        hall_std = np.asarray(res["hall_std"], dtype=np.float64)
        y = hall_mean[:, t_idx]
        s = hall_std[:, t_idx] if hall_std.shape == hall_mean.shape else np.zeros_like(y)
        if str(band).lower() == "sem":
            n_pairs = len(res.get("mode_pairs", []))
            s = s / np.sqrt(float(n_pairs)) if n_pairs > 0 else np.zeros_like(y)
        if str(band).lower() != "none":
            ax.fill_between(r_vals, np.clip(y - s, 0.0, 1.0), np.clip(y + s, 0.0, 1.0), alpha=alpha)
        ax.plot(r_vals, y, marker="o", linewidth=2, label=rf"$\eta={eta:.1f}$")

    ax.set_xlabel(r"Radius from midpoint (% of $\ell_t$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hallucination rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 50.0)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(rf"$\tau_3={int(tau_target)}$", fontsize=LABEL_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_ddim_eta_hall_radius_sweep(
    *,
    diffusion,
    dataset_name: str,
    gaussian_modes,
    mode_sigmas,
    std_dev: float,
    model,
    save_folder: str,
    protocol_label: str,
    run_name: str,
    eta_values,
    tau_vals,
    tau_internal_vals,
    tau_targets,
    r_percentage,
    num_samples: int,
    ddim_steps: int,
    skip_type: str,
    hall_radius_sigma_multiple: float,
):
    # run exp k by sweeping \eta and reusing the exp e radius computation.
    # exp e places starts directly on the time-scaled pair segment L_t.
    os.makedirs(save_folder, exist_ok=True)
    eta_values = [float(v) for v in eta_values]
    tau_vals = [int(v) for v in tau_vals]
    tau_internal_vals = [int(v) for v in tau_internal_vals]
    tau_targets = [int(v) for v in tau_targets]
    r_percentage = [float(v) for v in r_percentage]
    if len(tau_internal_vals) != len(tau_vals):
        raise ValueError("Experiment K tau_internal_vals must match tau_vals.")

    results_by_eta = {}
    long_rows = []

    for eta in eta_values:
        hall_dict = compute_hallucination_with_radius(
            dataset_name=dataset_name,
            diffusion=diffusion,
            model=model,
            sampling_mode="ddim",
            num_samples=int(num_samples),
            std_dev=float(std_dev),
            r_percentage=r_percentage,
            tau_vals=tau_vals,
            tau_internal_vals=tau_internal_vals,
            ddim_steps=int(ddim_steps),
            skip_type=str(skip_type),
            ddim_eta=float(eta),
            hall_radius_sigma_multiple=float(hall_radius_sigma_multiple),
            gaussian_modes=gaussian_modes,
            mode_sigmas=mode_sigmas,
        )
        results_by_eta[float(eta)] = hall_dict

        means = np.asarray(hall_dict["hall_mean"], dtype=np.float64)
        stds = np.asarray(hall_dict["hall_std"], dtype=np.float64)
        for r_idx, r in enumerate(r_percentage):
            for t_idx, tau in enumerate(tau_vals):
                long_rows.append(
                    {
                        "protocol": str(protocol_label),
                        "run_name": str(run_name),
                        "ddim_eta": float(eta),
                        "tau": int(tau),
                        "r_percentage": float(r),
                        "hall_mean": float(means[r_idx, t_idx]),
                        "hall_std": float(stds[r_idx, t_idx]),
                        "num_mode_pairs": int(len(hall_dict.get("mode_pairs", []))),
                    }
                )

    if len(tau_targets) != 1:
        raise ValueError("Experiment K now writes one cumulative eta plot for one fixed tau target.")
    tau_target = int(tau_targets[0])
    cumulative_pdf_path = os.path.join(save_folder, f"exp_k_ddim_eta_tau{tau_target}_cumulative.pdf")
    plot_eta_cumulative_fixed_tau(
        results_by_eta,
        tau_target=tau_target,
        save_path=cumulative_pdf_path,
        band="sem",
        alpha=0.22,
    )

    csv_path = os.path.join(save_folder, "exp_k_ddim_eta_hall_radius_long.csv")
    json_path = os.path.join(save_folder, "exp_k_ddim_eta_hall_radius.json")
    write_csv(
        csv_path,
        ["protocol", "run_name", "ddim_eta", "tau", "r_percentage", "hall_mean", "hall_std", "num_mode_pairs"],
        long_rows,
    )
    payload = {
        "experiment": "exp_k_ddim_eta_hall_radius",
        "protocol": str(protocol_label),
        "run_name": str(run_name),
        "dataset_name": str(dataset_name),
        "sampling_mode": "ddim",
        "eta_values": eta_values,
        "tau_vals": tau_vals,
        "tau_internal_vals": tau_internal_vals,
        "tau_targets": tau_targets,
        "r_percentage": r_percentage,
        "num_samples": int(num_samples),
        "ddim_steps": int(ddim_steps),
        "skip_type": str(skip_type),
        "hall_radius_sigma_multiple": float(hall_radius_sigma_multiple),
        "cumulative_pdf_path": cumulative_pdf_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "cumulative_pdf_path": cumulative_pdf_path,
    }
