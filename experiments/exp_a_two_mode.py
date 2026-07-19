'''
This script contains the experimental utils for running training.  
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from common.reverse_solvers import build_reverse_solver_stepper
from common.gaussian_mixture_2d import classify_samples_by_geometry

def schedule_sigma2_and_u_native(
    diffusion,
    *,
    timesteps: int,
    std_dev: float,
):
    # schedule sigma2 and u native.
    # this supports experiment a, which studies the times \tau_1 and \tau_2.
    alphas_prod = (
        diffusion.alphas_prod[: int(timesteps)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    betas = (
        diffusion.betas[: int(timesteps)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    sigma2_native = (float(std_dev) ** 2) * alphas_prod + (1.0 - alphas_prod)
    sigma2_native = np.clip(sigma2_native, 1e-12, None)
    u_increment = betas / (2.0 * sigma2_native)
    u_native = np.cumsum(u_increment[::-1])[::-1]
    return sigma2_native, u_native


def u_stats_from_crossing_indices( # used to check u time
    crossing_indices: np.ndarray,
    u_seq: np.ndarray,
):
    # u stats from crossing indices.
    # this supports experiment a, which studies the times \tau_1 and \tau_2.
    crossing_indices = np.asarray(crossing_indices)
    u_seq = np.asarray(u_seq, dtype=np.float64)
    u_samples = np.full(crossing_indices.shape, np.nan, dtype=np.float64)
    valid = crossing_indices >= 0
    if np.any(valid):
        u_samples[valid] = u_seq[crossing_indices[valid].astype(np.int64, copy=False)]
    u_mean = np.nanmean(u_samples, axis=0)
    u_std = np.nanstd(u_samples, axis=0)
    n = np.sum(~np.isnan(u_samples), axis=0).astype(np.float64)
    u_sem = np.divide(u_std, np.sqrt(n), out=np.full_like(u_std, np.nan), where=(n > 0))
    return u_samples, u_mean, u_std, u_sem


def plot_u_tau_1_tau_2_vs_kappa( # used to check u time
    tau_kappa_dict: dict,
    *,
    save_path=None,
):
    # plot u \tau 1 \tau 2 vs kappa for experiment a, which studies the stopping times \tau_1 and \tau_2.
    kappa = np.asarray(tau_kappa_dict["kappa"], dtype=np.float64)
    u_tau_1_mean = np.asarray(tau_kappa_dict["u_tau_1_mean"], dtype=np.float64)
    u_tau_1_std = np.asarray(tau_kappa_dict["u_tau_1_std"], dtype=np.float64)
    u_tau_2_mean = np.asarray(tau_kappa_dict["u_tau_2_mean"], dtype=np.float64)
    u_tau_2_std = np.asarray(tau_kappa_dict["u_tau_2_std"], dtype=np.float64)

    label_fontsize = 18
    tick_fontsize = 14
    legend_fontsize = 20

    plt.plot(kappa, u_tau_1_mean, linewidth=2, label=r"$\mathbb{E}[u(\tau_1)]$")
    plt.fill_between(
        kappa,
        u_tau_1_mean - u_tau_1_std,
        u_tau_1_mean + u_tau_1_std,
        alpha=0.25,
    )
    plt.plot(kappa, u_tau_2_mean, linewidth=2, label=r"$\mathbb{E}[u(\tau_2)]$")
    plt.fill_between(
        kappa,
        u_tau_2_mean - u_tau_2_std,
        u_tau_2_mean + u_tau_2_std,
        alpha=0.25,
    )
    plt.xlabel(r"$\kappa$", fontsize=label_fontsize)
    plt.ylabel(r"$u$", fontsize=label_fontsize)
    plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    plt.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def create_invalid_mask(modes, threshold=None, final_samples=None, mode_radii=None, pair_radii=None):
    # create invalid mask.
    # this supports experiment a, which studies the stopping times \tau_1 and \tau_2.
    if final_samples is None:
        raise ValueError("final_samples is required")
    if threshold is None and (mode_radii is None or pair_radii is None):
        raise ValueError("create_invalid_mask requires mode_radii/pair_radii or an explicit threshold.")

    geom = classify_samples_by_geometry(
        final_samples,
        modes,
        threshold=threshold,
        mode_radii=mode_radii,
        pair_radii=pair_radii,
    )
    return np.asarray(geom["invalid_mask"], dtype=bool)


def compute_interpolated_distances(
    final_samples, 
    true_modes, 
    threshold=None, 
    save_folder=None, 
    sampling_method="ddim", 
    ddim_timesteps=50,
    mode_radii=None,
    pair_radii=None,
):
    # compute interpolated distances for experiment a, which studies the stopping times \tau_1 and \tau_2.
    """
    Categorize samples and compute distances of interpolated samples to the line connecting
    the two closest modes (per sample).

    Saves:
      - hallucination.txt count file
    """
    os.makedirs(save_folder, exist_ok=True)

    # Convert to numpy
    if torch.is_tensor(final_samples):
        final_samples = final_samples.cpu().numpy()

    if torch.is_tensor(true_modes):
        true_modes = true_modes.cpu().numpy()

    if threshold is None and (mode_radii is None or pair_radii is None):
        raise ValueError(
            "compute_interpolated_distances requires mode_radii/pair_radii or an explicit threshold."
        )

    geom = classify_samples_by_geometry(
        final_samples,
        true_modes,
        threshold=threshold,
        mode_radii=mode_radii,
        pair_radii=pair_radii,
    )
    invalid_mask = np.asarray(geom["invalid_mask"], dtype=bool)
    true_mode_mask = np.asarray(geom["true_mode_mask"], dtype=bool)
    interpolated_mask = np.asarray(geom["interpolated_mask"], dtype=bool)
    distances_to_line = np.asarray(geom["distance_to_segment"], dtype=np.float64)[interpolated_mask]

    # --- Save results ---
    result_dict = {
        "true_mode": int(np.sum(true_mode_mask)),
        "interpolated": int(np.sum(interpolated_mask)),
        "invalid": int(np.sum(invalid_mask)),
        "distances_to_line": distances_to_line,
        "invalid_mask": invalid_mask,
        "true_mode_mask": true_mode_mask,
        "interpolated_mask": interpolated_mask,
        "closest_mode_idx": np.asarray(geom["closest_mode_idx"], dtype=np.int64),
        "nearest_pair_idx": np.asarray(geom["nearest_pair_idx"], dtype=np.int64),
    }
    if sampling_method == "ddpm": 
        save_path = os.path.join(save_folder, f"{sampling_method}_hallucination_{len(final_samples)}.txt")
    else: 
        save_path = os.path.join(save_folder, f"{sampling_method}_hallucination_{len(final_samples)}_{ddim_timesteps}.txt")
    
    with open(save_path, "w") as f:
        for k in ["true_mode", "interpolated", "invalid"]:
            f.write(f"{k}: {result_dict[k]}\n")

    print(f"Saved sample counts to {save_path}")
    return result_dict

@torch.no_grad()
def compute_tau_streaming(
    model,
    diffusion,
    gaussian_modes,              # (M, D) numpy or torch
    num_samples: int,
    num_dims: int,
    sampling_mode: str,          
    timesteps: int = 1000,       
    ddim_steps: int = 50,        
    skip_type: str = "quad",    
    ddim_eta: float = 0.0,       
    kappa_min: float = 1.0,
    kappa_max: float = 20.0,
    num_kappa_values: int = 100,
    std_dev: float = 0.02,
    device: str = "cuda",
    chunk_size: int = 8192,
    tau2_variance_scale_factor: float = 1.0,
    tau2_variance_scale_mode: str = "auto",
):
    # stream reverse trajectories and record the stopping times \tau_1 and \tau_2 used in experiment a.
    """
    To compute \tau_1, it suffices to find the time \tau_1 s.t. \max_{a \neq b} \Delta_t^{(a,b)} \geq 2 \kappa \sigma_t^2
    Here, \Delta_t^{(a,b)} := min_{k \neq a,b} ||x_t - \mu_t^{(k)}||_2^2 - \min(||x_t - \mu_t^{(a)}||_2^2, ||x_t - \mu_t^{(b)}||_2^2)
    Then, to get which i,j specifically the trajectory selects, one can take i, j = \argmax_{a, b; a \neq b} \Delta_t^{(a, b)}

    Then, can compute \tau_2 w.r.t. this i, j. Note that b/c RHS of assumptions is monotonically decreasing for our \sqrt{\bar{\alpha}_t}, suffices to just find when threshold is crossed

    The following does this in a streaming way, instead of allocating a massive (T, M, M, N) (M = # of modes, N = # of trajectories) tensor. 
    To do so, it does the following: 
        At each reverse step s = T-1, ..., 0 (or timestep_max, ..., 0 for DDIM): 
            Compute \mu_t
            Compute x_s^{(n)} of size [B,] for some batch size B <<< N
            Compute distances to all modes, take top 3; d1 < d2 < d3 
            Compute u^{(n)} = d3 - d1 = \max_{a \neq b} \Delta_t^{(a, b )}. Get v^{(n)} by dividing 2 \sigma_t^2.
            Compute tau_1: first time s.t. v^{(n)} \geq \kappa. 
                maintain next_g[n]; smallest \kappa s.t. this isn't satisfied for sample n. 
                when v^{(n)} increases enough s.t. this is satisfied for more \kappas, fill in tensor of \tau_1 vs. \kappa at this time 
                also maintain i, j
            Use i, j to compute \tau_2 w.r.t. precomputed \ell^2 matrix 
    
    N.B.: This computes trajectories online, so no need for trajectories from sample_two_mode_gaussian

    returns tau_kappa_dict for plotting: 
        {
            "kappa": kappa,
            "tau_1_mean": tau_1_mean,
            "tau_1_std": tau_1_std,
            "raw_tau_1": tau_1_samples,
            "tau_2_mean": tau_2_mean,
            "tau_2_std": tau_2_std,
            "raw_tau_2": tau_2_samples,
            "final_sample": final_sample,
            "metadata": {
                ...
            },
        }
    """

    sampling_mode = sampling_mode.lower()
    skip_type = skip_type.lower()
    model.eval()

    kappa = np.linspace(kappa_min, kappa_max, num_kappa_values).astype(np.float64)
    G = len(kappa)

    seq, step_fn = build_reverse_solver_stepper(
        diffusion=diffusion,
        sampling_mode=sampling_mode,
        timesteps=timesteps,
        ddim_steps=ddim_steps,
        skip_type=skip_type,
        ddim_eta=ddim_eta,
    )
    T = len(seq)
    eff_var = np.asarray(
        diffusion.effective_variance(
            num_steps=T,
            total_time_steps=timesteps,
            sigma=std_dev,
            seq=seq,
        ),
        dtype=np.float64,
    )
    sigma2_native, u_native = schedule_sigma2_and_u_native(
        diffusion,
        timesteps=int(timesteps),
        std_dev=float(std_dev),
    )
    reverse_timestep_seq = np.asarray(seq, dtype=np.int64)
    sigma2_seq = sigma2_native[reverse_timestep_seq]

    # fixed modes and ell2 (tau_2 uses fixed means)
    modes_fixed = np.asarray(gaussian_modes, dtype=np.float64)  # (M,D)
    M = modes_fixed.shape[0]
    ell2_mat = np.sum((modes_fixed[:, None, :] - modes_fixed[None, :, :]) ** 2, axis=-1)

    tau2_variance_scale_mode = str(tau2_variance_scale_mode).strip().lower()
    if tau2_variance_scale_mode == "auto":
        # the fixed protocol uses the \varpi-scaled rhs in asm. 4.4, so auto resolves to \varpi=d here.
        tau2_variance_scale_factor = float(num_dims)
        tau2_variance_scale_mode = "multiply_by_varpi"
    elif tau2_variance_scale_mode == "multiply_by_varpi":
        # if the caller requested the \varpi-scaled rule explicitly, make the factor match \varpi=d.
        tau2_variance_scale_factor = float(num_dims)
    elif tau2_variance_scale_mode == "none":
        tau2_variance_scale_factor = float(tau2_variance_scale_factor)
    else:
        raise ValueError(
            "tau2_variance_scale_mode must be one of {'auto', 'none', 'multiply_by_varpi'}."
        )

    tau2_variance_scale_factor = float(tau2_variance_scale_factor)
    if tau2_variance_scale_factor <= 0.0:
        raise ValueError(f"tau2_variance_scale_factor must be > 0, got {tau2_variance_scale_factor}")

    # precompute X2 lookup per (g,i,j): first s with ell2 >= 4*kappa*eff_var[s]*scale
    rhs2 = 4.0 * eff_var[:, None] * kappa[None, :] * tau2_variance_scale_factor  # (T,G)
    X2_pair = np.full((G, M, M), -1, dtype=np.int16)
    for g in range(G):
        thr = rhs2[:, g]
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                cond = (ell2_mat[i, j] >= thr)
                if cond.any():
                    X2_pair[g, i, j] = int(cond.argmax())

    X1   = np.full((num_samples, G), -1, dtype=np.int16)
    i_tau = np.full((num_samples, G), -1, dtype=np.int16)
    j_tau = np.full((num_samples, G), -1, dtype=np.int16)
    next_g = np.zeros((num_samples,), dtype=np.int16)  # per-sample frontier in kappa

    x = torch.randn(num_samples, num_dims, device=device) # x_{T - 1}
    prev_x0 = None
    prev_h = None

    # modes on GPU for distance computations
    orig_modes = torch.as_tensor(gaussian_modes, device=device, dtype=x.dtype)  # (M,D)

    # ensure model timestep dtype compatibility
    def call_model(xt, t_int):
        # call model.
        # this supports experiment a, which studies \tau_1(\kappa) and \tau_2(\kappa).
        t_long = torch.full((xt.shape[0],), t_int, device=xt.device, dtype=torch.long)
        try:
            return model(xt, t_long)
        except Exception:
            return model(xt, t_long.float())

    for s in range(T):
        t_eff = int(seq[s])  # current effective timestep (descending)

        # time-dependent modes for computing distances: mu_t = sqrt(alpha_bar(t_eff))*mu
        a_bar = diffusion.compute_alpha_t_bar(t_eff)  # tensor scalar
        modes_t = orig_modes * torch.sqrt(a_bar)

        # compute u_best and best (i,j) in chunks, update tau_1 
        # Asm. 4.1 is checked against the native VP marginal variance sigma_t^2, not the
        # effective-variance surrogate for the tau_2 threshold.
        denom = 2.0 * sigma2_seq[s]
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            xt = x[start:end]  # (B,D)

            # squared distances: (B,M)
            diff = xt[:, None, :] - modes_t[None, :, :]
            dist = (diff * diff).sum(dim=-1)

            vals, idx = torch.topk(dist, k=3, largest=False)  # (B,3)
            u = (vals[:, 2] - vals[:, 0]).detach().cpu().numpy().astype(np.float64)  # (B,)
            i_best = idx[:, 0].detach().cpu().numpy().astype(np.int16)
            j_best = idx[:, 1].detach().cpu().numpy().astype(np.int16)

            v = u / denom  # condition v >= kappa
            g_new = np.searchsorted(kappa, v, side="right").astype(np.int16)  # (B,) in [0,G]

            ng = next_g[start:end]
            upd = g_new > ng
            if np.any(upd):
                upd_idx = np.where(upd)[0]
                for loc in upd_idx:
                    a = int(ng[loc])
                    b = int(g_new[loc])
                    X1[start + loc, a:b] = s
                    i_tau[start + loc, a:b] = i_best[loc]
                    j_tau[start + loc, a:b] = j_best[loc]
                    ng[loc] = b
            next_g[start:end] = ng

        t_next = int(seq[s + 1]) if (s + 1) < T else -1
        x, prev_x0, prev_h = step_fn(
            model=model,
            x_t=x,
            t_eff=int(t_eff),
            t_next=int(t_next),
            prev_x0=prev_x0,
            prev_h=prev_h,
        )

    final_sample = x.detach().cpu().numpy()

    # compute tau_1 stats 
    X1f = X1.astype(np.float64)
    X1f[X1 < 0] = np.nan
    tau_1_samples = T - X1f
    tau_1_mean = np.nanmean(tau_1_samples, axis=0)
    tau_1_std  = np.nanstd(tau_1_samples, axis=0)

    # compute tau_2 via lookup 
    X2 = np.full((num_samples, G), -1, dtype=np.int16)
    for g in range(G):
        ii = i_tau[:, g]
        jj = j_tau[:, g]
        mask = (ii >= 0) & (jj >= 0)
        X2[mask, g] = X2_pair[g, ii[mask], jj[mask]]

    n1 = np.sum(~np.isnan(tau_1_samples), axis=0).astype(np.float64)  # (G,)
    tau_1_sem = np.divide(tau_1_std, np.sqrt(n1), out=np.full_like(tau_1_std, np.nan), where=(n1 > 0))


    X2f = X2.astype(np.float64)
    X2f[X2 < 0] = np.nan
    tau_2_samples = T - X2f
    tau_2_mean = np.nanmean(tau_2_samples, axis=0)
    tau_2_std  = np.nanstd(tau_2_samples, axis=0)

    n2 = np.sum(~np.isnan(tau_2_samples), axis=0).astype(np.float64)  # (G,)
    tau_2_sem = np.divide(tau_2_std, np.sqrt(n2), out=np.full_like(tau_2_std, np.nan), where=(n2 > 0))

    u_seq = u_native[reverse_timestep_seq]
    u_tau_1_samples, u_tau_1_mean, u_tau_1_std, u_tau_1_sem = u_stats_from_crossing_indices(X1, u_seq)
    u_tau_2_samples, u_tau_2_mean, u_tau_2_std, u_tau_2_sem = u_stats_from_crossing_indices(X2, u_seq)


    return {
        "kappa": kappa,
        "tau_1_mean": tau_1_mean,
        "tau_1_std": tau_1_std,
        "tau_1_sem": tau_1_sem,
        "raw_tau_1": tau_1_samples,
        "tau_2_mean": tau_2_mean,
        "tau_2_std": tau_2_std,
        "tau_2_sem": tau_2_sem,
        "raw_tau_2": tau_2_samples,
        "u_tau_1_mean": u_tau_1_mean,
        "u_tau_1_std": u_tau_1_std,
        "u_tau_1_sem": u_tau_1_sem,
        "raw_u_tau_1": u_tau_1_samples,
        "u_tau_2_mean": u_tau_2_mean,
        "u_tau_2_std": u_tau_2_std,
        "u_tau_2_sem": u_tau_2_sem,
        "raw_u_tau_2": u_tau_2_samples,
        "final_sample": final_sample,
        # "unsat_tau_1": unsat_tau_1,
        # "unsat_tau_2": unsat_tau_2,
        "seq": seq,
        "reverse_timestep_seq": reverse_timestep_seq,
        "eff_var": eff_var,
        "sigma2_native": sigma2_native,
        "sigma2_seq": sigma2_seq,
        "u_native": u_native,
        "u_seq": u_seq,
        "i_tau": i_tau,
        "j_tau": j_tau,
        "metadata": {
            "sampling_mode": sampling_mode,
            "skip_type": (skip_type if sampling_mode != "ddpm" else None),
            "ddim_steps": (ddim_steps if sampling_mode != "ddpm" else None),
            "timesteps": timesteps,
            "num_samples": num_samples,
            "T": T,
            "num_dims": num_dims,
            "tau1_variance_convention": "sigma_t_squared",
            "tau2_variance_scale_mode": str(tau2_variance_scale_mode),
            "tau2_variance_scale_factor": tau2_variance_scale_factor,
            "time_convention_step": "remaining_reverse_solver_steps",
            "time_convention_u": "u(t)=sum_{s=t}^{T-1} beta_s/(2 sigma_s^2), increasing as reverse time approaches 0",
        },
    }

def plot_tau_1_tau_2_vs_kappa(
    model,
    diffusion,
    dataset_name : str,
    num_samples: int,
    num_dims: int,
    sampling_mode: str,          
    timesteps: int = 1000,       
    ddim_steps: int = 50,        
    skip_type: str = "quad",    
    ddim_eta: float = 0.0,       
    kappa_min: float = 1.0,
    kappa_max: float = 20.0,
    num_kappa_values: int = 100,
    std_dev: float = 0.02,
    device: str = "cuda",
    chunk_size: int = 8192,
    save_path=None,
    gaussian_modes=None,
    tau2_variance_scale_factor: float = 1.0,
    tau2_variance_scale_mode: str = "auto",
): 
    # plot \tau 1 \tau 2 vs kappa for experiment a, which studies the stopping times \tau_1 and \tau_2.
    if gaussian_modes is None:
        raise ValueError("gaussian_modes must be provided for experiment A.")
    gaussian_modes = np.asarray(gaussian_modes, dtype=np.float64)

    tau_kappa_dict = compute_tau_streaming(
        model,
        diffusion,
        gaussian_modes,              
        num_samples,
        num_dims,
        sampling_mode,          
        timesteps,       
        ddim_steps,        
        skip_type,    
        ddim_eta,       
        kappa_min,
        kappa_max,
        num_kappa_values,
        std_dev,
        device,
        chunk_size,
        tau2_variance_scale_factor,
        tau2_variance_scale_mode,
    )

    kappa = tau_kappa_dict["kappa"]
    tau_1_mean = tau_kappa_dict["tau_1_mean"]
    tau_1_std =  tau_kappa_dict["tau_1_std"]
    tau_2_mean = tau_kappa_dict["tau_2_mean"]
    tau_2_std =  tau_kappa_dict["tau_2_std"]
    tau_1_sem = tau_kappa_dict["tau_1_sem"]
    tau_2_sem = tau_kappa_dict["tau_2_sem"]

    # --- styling knobs ---
    LABEL_FONTSIZE  = 18
    TICK_FONTSIZE   = 14
    LEGEND_FONTSIZE = 20

    # Plot \tau_1
    plt.plot(kappa, tau_1_mean, linewidth=2, label=r"$\mathbb{E}[\tau_1]$")
    plt.fill_between(
        kappa,
        tau_1_mean - tau_1_std,
        tau_1_mean + tau_1_std,
        alpha=0.25,
        # no label -> no "+/- std" entry in legend
    )

    # Plot \tau_2
    plt.plot(kappa, tau_2_mean, linewidth=2, label=r"$\mathbb{E}[\tau_2]$")
    plt.fill_between(
        kappa,
        tau_2_mean - tau_2_std,
        tau_2_mean + tau_2_std,
        alpha=0.25,
    )

    plt.xlabel(r"$\kappa$", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"$t$", fontsize=LABEL_FONTSIZE)

    plt.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plt.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)

    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return tau_kappa_dict


def sample_two_mode_gaussian(
    model,
    diffusion,
    timesteps,
    dataset_name,
    num_samples=1000,
    device="cuda",
    vis_trajectory=False,
    init_samples=None,
    save_folder="./results",
    sampling_mode="ddpm",
    skip_type="quad",
    ddim_steps=50,
    num_dims=2, 
    gaussian_modes=None,
):
    # sample two mode gaussian for experiment a, which studies the stopping times \tau_1 and \tau_2.
    model.eval()
    dim = num_dims
    trajectories_full = []

    if init_samples is None:
        x = torch.randn(num_samples, dim, device=device)
    else:
        x = torch.tensor(init_samples, device=device).float()

    seq = np.arange(0, timesteps, 1)
    if gaussian_modes is None:
        raise ValueError("gaussian_modes must be provided for Gaussian sampling.")
    gaussian_modes = np.asarray(gaussian_modes, dtype=np.float64)
    
    seq, step_fn = build_reverse_solver_stepper(
        diffusion=diffusion,
        sampling_mode=sampling_mode,
        timesteps=timesteps,
        ddim_steps=ddim_steps,
        skip_type=skip_type,
        ddim_eta=0.0,
    )
    prev_x0 = None
    prev_h = None
    with torch.no_grad():
        trajectories_full.append(x.cpu().clone())
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
            trajectories_full.append(x.cpu().clone())

    trajectories = trajectories_full[:-1]

    print(f"Final samples shape: {x.shape}")

    model.train()
    return_dict = {
        "final_sample": x.cpu().numpy(),
        "trajectories": trajectories_full,
        "seq": seq,
        "gaussian_modes": gaussian_modes,
    }

    return return_dict
