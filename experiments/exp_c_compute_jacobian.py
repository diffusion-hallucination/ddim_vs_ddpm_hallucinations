import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.reverse_solvers import build_reverse_solver_stepper


def find_midpoints_between_closest_modes(modes: torch.Tensor, tol: float = 1e-8):
    dists = torch.cdist(modes, modes).clone()
    dists.fill_diagonal_(float("inf"))
    min_dist = dists.min()
    closest_mask = torch.triu(dists <= (min_dist + float(tol)), diagonal=1)
    pair_indices = torch.nonzero(closest_mask, as_tuple=False)
    if pair_indices.numel() == 0:
        return (
            torch.empty(0, modes.shape[1], device=modes.device),
            torch.empty(0, 2, dtype=torch.long, device=modes.device),
            float("nan"),
        )
    mode_i = modes[pair_indices[:, 0]]
    mode_j = modes[pair_indices[:, 1]]
    return 0.5 * (mode_i + mode_j), pair_indices, float(min_dist.item())


def sample_two_mode_gaussian_with_eig_values(
    model,
    diffusion,
    timesteps,
    dataset_name,
    num_samples=1000,
    device="cuda",
    vis_trajectory=False,
    init_samples=None,
    save_folder="./results",
    skip_type="quad",
    ddim_steps=50,
    sampling_mode="ddim",
    gaussian_modes=None,
):
    # Estimate the DDIM midpoint-jacobian spectrum for the closest-pair midpoints.
    mode = str(sampling_mode).lower()
    if mode != "ddim":
        raise ValueError("Experiment C now only supports sampling_mode='ddim'.")
    if gaussian_modes is None:
        raise ValueError("gaussian_modes must be provided for experiment C.")

    modes = torch.as_tensor(gaussian_modes, dtype=torch.float32)
    mode_midpoints, _, _ = find_midpoints_between_closest_modes(modes)
    if mode_midpoints.numel() == 0:
        raise ValueError("Experiment C requires at least one closest-pair midpoint.")

    was_training = bool(getattr(model, "training", False))
    model.eval()

    seq, step_fn = build_reverse_solver_stepper(
        diffusion=diffusion,
        sampling_mode="ddim",
        timesteps=int(timesteps),
        ddim_steps=int(ddim_steps),
        skip_type=str(skip_type),
        ddim_eta=0.0,
    )

    eval_points = mode_midpoints.to(device=device, dtype=torch.float32)
    all_eigvals = []
    t_axis = []
    dt_used = []

    for s, t_eff in enumerate(seq):
        t_eff = int(t_eff)
        t_next = int(seq[s + 1]) if (s + 1) < len(seq) else -1
        dt = float(t_eff - (t_next if t_next >= 0 else -1))
        if dt <= 0:
            continue

        def reverse_velocity(x_in):
            x_out, _, _ = step_fn(
                model=model,
                x_t=x_in,
                t_eff=t_eff,
                t_next=t_next,
                prev_x0=None,
                prev_h=None,
            )
            return (x_out - x_in) / dt

        eigvals_this_step = []
        for p_idx in range(int(eval_points.shape[0])):
            x_point = eval_points[p_idx : p_idx + 1].detach().requires_grad_(True)
            jac = torch.autograd.functional.jacobian(reverse_velocity, x_point, create_graph=False)
            jac = jac[0, :, 0, :]
            eigvals_this_step.append(torch.linalg.eigvals(jac).detach().cpu().numpy())

        all_eigvals.append(np.stack(eigvals_this_step, axis=0))
        t_axis.append(t_eff)
        dt_used.append(dt)

    if was_training:
        model.train()

    return {
        "seq": list(seq),
        "all_eigvals": all_eigvals,
        "midpoints": mode_midpoints,
        "eig_t_axis": t_axis,
        "eig_dt": dt_used,
    }


def visualize_eig_values_grouped_mean_std(
    all_eigvals,
    mode_midpoints,
    save_path,
    tol: float = 1e-12,
    plot_individual: bool = False,
    tau_kappa_dict: dict | None = None,
    kappa_target: float | None = None,
):
    eig = np.asarray(all_eigvals)
    if eig.ndim != 3 or eig.shape[2] != 2:
        raise ValueError(f"Expected all_eigvals shape (T, N, 2), got {eig.shape}")
    if len(mode_midpoints) != eig.shape[1]:
        raise ValueError(f"len(mode_midpoints)={len(mode_midpoints)} != N={eig.shape[1]}")

    eig_real = np.real(eig)
    pos_trajs = []
    neg_trajs = []
    for p in range(eig_real.shape[1]):
        for e in range(eig_real.shape[2]):
            traj = eig_real[:, p, e]
            if traj[-1] > tol:
                pos_trajs.append(traj)
            elif traj[-1] < -tol:
                neg_trajs.append(traj)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t = np.arange(eig_real.shape[0], -1, -1)

    def extend_last(y):
        return np.concatenate([y, [y[-1]]], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    if pos_trajs:
        arr = np.stack(pos_trajs, axis=0)
        mean = extend_last(arr.mean(axis=0))
        std = extend_last(arr.std(axis=0, ddof=0))
        if plot_individual:
            for tr in arr:
                ax.plot(t[:-1], tr, alpha=0.12, linewidth=1)
        ax.plot(t, mean, linewidth=2, label=r"$\mathbb{E}[\lambda_{+}]$", color="tab:blue")
        ax.fill_between(t, mean - std, mean + std, alpha=0.25, color="tab:blue")
    else:
        print("[eig plot] POS group empty (no Re(lambda_T) > tol).")

    if neg_trajs:
        arr = np.stack(neg_trajs, axis=0)
        mean = extend_last(arr.mean(axis=0))
        std = extend_last(arr.std(axis=0, ddof=0))
        if plot_individual:
            for tr in arr:
                ax.plot(t[:-1], tr, alpha=0.12, linewidth=1)
        ax.plot(t, mean, linewidth=2, label=r"$\mathbb{E}[\lambda_{-}]$", color="tab:green")
        ax.fill_between(t, mean - std, mean + std, alpha=0.25, color="tab:green")
    else:
        print("[eig plot] NEG group empty (no Re(lambda_T) < -tol).")

    tau2_val = None
    if tau_kappa_dict is not None and kappa_target is not None:
        kappa_vals = np.asarray(tau_kappa_dict.get("kappa", []), dtype=float)
        tau2_mean = np.asarray(tau_kappa_dict.get("tau_2_mean", []), dtype=float)
        mask = np.isfinite(kappa_vals) & np.isfinite(tau2_mean)
        if mask.sum() >= 2:
            tau2_val = float(np.interp(float(kappa_target), kappa_vals[mask], tau2_mean[mask]))
            ax.axvline(
                tau2_val,
                color="tab:orange",
                linestyle=":",
                linewidth=2,
                alpha=0.95,
                label=rf"$\tau_2\;(\kappa={float(kappa_target):g})$",
            )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(r"$t$", fontsize=22)
    ax.set_ylabel(r"$\lambda$", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=18)
    ax.set_xlim(eig_real.shape[0], 0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[eig plot] saved: {save_path}")
    print(f"[eig plot] counts: POS={len(pos_trajs)} NEG={len(neg_trajs)} (tol={tol:g})")
    if tau2_val is not None:
        print(f"[eig plot] tau2(kappa={float(kappa_target):g}) ~= {tau2_val:.3g}")
