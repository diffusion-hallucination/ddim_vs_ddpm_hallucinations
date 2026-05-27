import os
import numpy as np
import torch
import matplotlib.pyplot as plt

RENAME_MAP = {
    "gaussian25": "Gaussian25-2D"
}


def visualize_gaussian_with_modes(final_samples, invalid_mask, true_modes,
                                  dataset_name, fig_save_folder, fig_name,
                                  dpi=200):

                                      # visualize gaussian with modes for the gaussian plotting helpers used across the synthetic experiments.
    os.makedirs(fig_save_folder, exist_ok=True)

    # Convert tensors to numpy
    if torch.is_tensor(final_samples):
        final_samples = final_samples.detach().cpu().numpy()
    if torch.is_tensor(true_modes):
        true_modes = true_modes.detach().cpu().numpy()
    if torch.is_tensor(invalid_mask):
        invalid_mask = invalid_mask.detach().cpu().numpy()
    invalid_mask = invalid_mask.astype(bool)

    if true_modes.ndim == 3:
        true_modes = true_modes.squeeze(0)

    fig, ax = plt.subplots(figsize=(6, 6))

    name = dataset_name.lower()
    xlim = (-4, 4) if ("2d" in name or "gaussian" in name) else None
    ylim = (-4, 4) if ("2d" in name or "gaussian" in name) else None

    valid_final_samples = final_samples[~invalid_mask]

    ax.scatter(
        valid_final_samples[:, 0],
        valid_final_samples[:, 1],
        s=20,
        alpha=0.5,
        label="Generated Samples",
        rasterized=True,   # <--- key line
        zorder=1,
    )

    ax.scatter(
        true_modes[:, 0],
        true_modes[:, 1],
        s=80,
        c="red",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label="True Modes",
        zorder=5,
    )

    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if xlim is not None:
        ax.set_xlim([-2.0, 2.0])
    if ylim is not None:
        ax.set_ylim([-2.0, 2.0])

    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(fig_save_folder, fig_name + ".pdf")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)  # dpi controls the rasterized layer resolution
    plt.close(fig)

    print(f"Saved samples to: {save_path}")
    return save_path
