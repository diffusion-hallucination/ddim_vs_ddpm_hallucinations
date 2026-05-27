'''
This script contains the experimental utils for running training. 
There's a convergence to nearby line we test in this setup.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
sys.path.append(os.getcwd())
from common.reverse_solvers import build_reverse_solver_stepper

def true_score_noised_mixture(xb, modes_t, sigma2_eff):
    # xb: (B,D), modes_t: (M,D), sigma2_eff: scalar tensor
    diff = xb[:, None, :] - modes_t[None, :, :]          # (B,M,D)
    dist2 = (diff * diff).sum(dim=-1)                    # (B,M)
    logw = -0.5 * dist2 / sigma2_eff                     # (B,M)
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
    w = torch.exp(logw)                                  # (B,M)
    mu_bar = (w[:, :, None] * modes_t[None, :, :]).sum(dim=1)  # (B,D)
    return (mu_bar - xb) / sigma2_eff


# helper for computing distance to nearby line 
def dist_to_segment_batch(xb, a, b, tmin=0.0, tmax=1.0):
    # dist to segment batch.
    # this supports experiment b, which studies convergence back to the closest-pair segment.
    ab = b - a
    ab_norm_sq = (ab * ab).sum(dim=-1).clamp_min(1e-12)
    t = ((xb - a) * ab).sum(dim=-1) / ab_norm_sq
    tmin_t = torch.as_tensor(tmin, device=t.device, dtype=t.dtype)
    tmax_t = torch.as_tensor(tmax, device=t.device, dtype=t.dtype)
    t = torch.maximum(t, tmin_t)
    t = torch.minimum(t, tmax_t)
    proj = a + t[:, None] * ab
    return torch.linalg.norm(xb - proj, dim=-1)


@torch.no_grad()
def convergence_to_nearby_line_streaming(
    model,
    diffusion,
    dataset_name: str,
    num_samples: int,
    num_dims: int,
    sampling_mode: str,
    timesteps: int = 1000,
    ddim_steps: int = 50,
    skip_type: str = "quad",
    ddim_eta: float = 0.0,
    std_dev: float = 0.02,
    device: str = "cuda",
    chunk_size: int = 8192,
    tau_kappa_dict: dict | None = None,   
    kappa_target: float = 5.0,
    fig_save_path: str | None = None,
    gaussian_modes=None,
    mode_sigmas=None,
    invalid_sigma_multiple: float = 5.0,
):
    # stream reverse trajectories and measure how quickly they return to the closest-pair segment.
    """
    Sequential version of method to track convergence to the nearest (time-dependent) L_{t,\varepsilon}^{(i,j)}

    Generates x along reverse process (DDPM or DDIM) without storing trajectories.
    At each reverse step s:
        build time-dependent modes mu_t = sqrt(alpha_bar(t_eff)) * mu
        for each sample, pick 2 nearest modes at that time
        compute distance from x to the epsilon-extended line segment through those 2 modes
        filter distances using the dynamic normalized-space sigma neighborhood cutoff
    Returns mean/std distance vs reverse-step index s.

    Distances here are normalized as d_perp / sqrt(varpi).

    The cutoff is always a time-dependent, pairwise normalized sigma rule.
    For the two nearest time-dependent modes at step t, we use the normalized
    effective pair radius
        invalid_sigma_multiple * 0.5 * (sigma_eff_i(t) + sigma_eff_j(t)) / sqrt(varpi),
    where sigma_eff_i(t) is the forward-process effective sigma in normalized
    model coordinates for mode i at that step.

    epsilon = Nexp(-\kappa)
    notes:
      - For DDPM: T = timesteps, seq = [T-1,...,0], t_eff = seq[s]
      - For DDIM: T = len(seq)=ddim_steps, t_eff = seq[s] from the skip schedule
    """
    dim_scale = float(np.sqrt(num_dims)) 
    sampling_mode = sampling_mode.lower()
    skip_type = skip_type.lower()
    model.eval()

    if gaussian_modes is None:
        raise ValueError("gaussian_modes must be provided for experiment B.")
    modes_fixed = np.asarray(gaussian_modes, dtype=np.float64)
    modes_fixed = np.asarray(modes_fixed, dtype=np.float64)  # (M, D)
    M = modes_fixed.shape[0]
    if mode_sigmas is None:
        mode_sigmas_fixed = np.full((M,), float(std_dev), dtype=np.float64)
    else:
        mode_sigmas_fixed = np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
        if mode_sigmas_fixed.shape[0] != M:
            raise ValueError("mode_sigmas must have one value per mode")

    eps_base = float(M) * float(np.exp(-kappa_target))  # noncircular eps


    seq, step_fn = build_reverse_solver_stepper(
        diffusion=diffusion,
        sampling_mode=sampling_mode,
        timesteps=timesteps,
        ddim_steps=ddim_steps,
        skip_type=skip_type,
        ddim_eta=ddim_eta,
    )
    T = len(seq)

    x = torch.randn(num_samples, num_dims, device=device)
    prev_x0 = None
    prev_h = None

    base_modes = torch.as_tensor(modes_fixed, device=device, dtype=x.dtype)  # (M, D)
    base_sigmas = torch.as_tensor(mode_sigmas_fixed, device=device, dtype=x.dtype)  # (M,)
    cutoff_mode = "dynamic_pair_sigma_t_normalized"

    def call_model(xt, t_int: int):
        # call model.
        # this supports experiment b, which studies convergence toward the nearest mode pair segment.
        t_long = torch.full((xt.shape[0],), t_int, device=xt.device, dtype=torch.long)
        try:
            return model(xt, t_long)
        except Exception:
            return model(xt, t_long.float())

    # stats per reverse step
    mean_dist = np.full((T,), np.nan, dtype=np.float64)
    std_dist  = np.full((T,), np.nan, dtype=np.float64)
    # n_eff     = np.zeros((T,), dtype=np.int64)
    # sem_dist  = np.full((T,), np.nan, dtype=np.float64)
    alpha_bar_list = np.full((T,), np.nan, dtype=np.float64)

    # score error diagnostics
    eps_err_mean  = np.full(T, np.nan, dtype=np.float64)
    eps_err_p95   = np.full(T, np.nan, dtype=np.float64)
    eps_true_norm_mean = np.full(T, np.nan, dtype=np.float64)
    eps_norm_mean = np.full(T, np.nan, dtype=np.float64)
    x0_norm_mean  = np.full(T, np.nan, dtype=np.float64)

    diag_every = 5          # every 5 steps
    diag_bs = 1024          # only 1024 samples for diagnostics

    for s in range(T):
        t_eff = int(seq[s])  # effective diffusion timestep index

        # alpha_bar(t_eff) (scalar)
        # use diffusion.alphas_prod if available (it is in your DDIM code)
        a_bar = diffusion.alphas_prod[t_eff].to(device=device, dtype=x.dtype)
        alpha_bar_list[s] = float(a_bar.detach().cpu().item())

        # time-dependent modes and normalized effective sigmas
        modes_t = base_modes * torch.sqrt(a_bar)  # (M,D)
        sigma_eff_modes_t = torch.sqrt(a_bar * (base_sigmas ** 2) + (1.0 - a_bar))  # (M,)

        # accumulate filtered distances across all samples (in chunks)
        sum_d = 0.0
        sum_d2 = 0.0
        count = 0

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            xb = x[start:end]  # (B,D)

            # squared distances to all time-dependent modes: (B,M)
            diff = xb[:, None, :] - modes_t[None, :, :]
            dist2 = (diff * diff).sum(dim=-1)

            # two nearest modes at *this time*
            vals2, idx2 = torch.topk(dist2, k=2, largest=False)  # (B,2)
            i0 = idx2[:, 0]
            i1 = idx2[:, 1]

            a = modes_t[i0]  # (B,D)
            b = modes_t[i1]  # (B,D)
            # note, epsilon along parallel component is dimensionless, so dont rescale by \sqrt{d}
            d_line = dist_to_segment_batch(xb, a, b, tmin=-eps_base, tmax=1.0 + eps_base)  # (B,)

            # normalize distance by sqrt(d)
            d_line_norm = d_line / dim_scale

            pair_sigma_eff = 0.5 * (sigma_eff_modes_t[i0] + sigma_eff_modes_t[i1])
            active_cutoff = float(invalid_sigma_multiple) * pair_sigma_eff / dim_scale
            valid = d_line_norm <= active_cutoff

            if valid.any():
                dv = d_line_norm[valid].detach().cpu().numpy().astype(np.float64)
                sum_d += dv.sum()
                sum_d2 += (dv * dv).sum()
                count += dv.shape[0]

        if count > 0:
            m = sum_d / count

            # variance from streaming sums
            v_pop = max(sum_d2 / count - m * m, 0.0)
            std_pop = np.sqrt(v_pop)

            mean_dist[s] = m
            std_dist[s]  = std_pop

        if (s % diag_every) == 0:  # epsilon-space diagnostics (stable)
            xb_diag = x[:diag_bs]
            eps_pred = call_model(xb_diag, t_eff)

            # --- compute analytic "true eps" in float64 for stability ---
            xb64  = xb_diag.double()
            a_bar64 = diffusion.alphas_prod[t_eff].to(device=device, dtype=torch.float64)
            one_minus_ab64 = (1.0 - a_bar64).clamp_min(1e-18)

            # time-dependent modes already computed as modes_t = base_modes * sqrt(a_bar)
            # cast just the diag slice
            modes_t64 = modes_t.double()

            sigma2_eff64 = a_bar64 * (std_dev ** 2) + (1.0 - a_bar64)

            score_true64 = true_score_noised_mixture(xb64, modes_t64, sigma2_eff64)
            eps_true64 = -torch.sqrt(one_minus_ab64) * score_true64  # (B, D) float64

            eps_err = torch.linalg.norm(eps_pred.double() - eps_true64, dim=-1)  # (B,)
            eps_err_mean[s] = float(eps_err.mean().detach().cpu().item())
            eps_err_p95[s]  = float(torch.quantile(eps_err, 0.95).detach().cpu().item())

            eps_norm_mean[s] = float(torch.linalg.norm(eps_pred, dim=-1).mean().detach().cpu().item())
            eps_true_norm_mean[s] = float(torch.linalg.norm(eps_true64, dim=-1).mean().detach().cpu().item())

            # x0 implied by eps_pred (sanity / blow-up detector)
            a_bar32 = diffusion.alphas_prod[t_eff].to(device=device, dtype=xb_diag.dtype)
            one_minus_ab32 = (1.0 - a_bar32).clamp_min(1e-12)
            x0_hat = (xb_diag - eps_pred * torch.sqrt(one_minus_ab32)) / torch.sqrt(a_bar32.clamp_min(1e-12))
            x0_norm_mean[s] = float(torch.linalg.norm(x0_hat, dim=-1).mean().detach().cpu().item())

            print(
                f"[s={s:03d} t_eff={t_eff:04d}] "
                f"eps_err_mean={eps_err_mean[s]:.3g} "
                f"eps_err_p95={eps_err_p95[s]:.3g} "
                f"eps_norm_mean={eps_norm_mean[s]:.3g} "
                f"eps_true_norm_mean={eps_true_norm_mean[s]:.3g} "
                f"x0_norm_mean={x0_norm_mean[s]:.3g} "
            )

        t_next = int(seq[s + 1]) if (s + 1) < T else -1
        x, prev_x0, prev_h = step_fn(
            model=model,
            x_t=x,
            t_eff=int(t_eff),
            t_next=int(t_next),
            prev_x0=prev_x0,
            prev_h=prev_h,
        )

    # simple exponential reference bound (discrete steps)
    mean_up_bound = mean_dist[0] * np.exp(-np.arange(T))

    plt.figure(figsize=(7, 4))

    # exnted by one point so we can display timestep 0
    steps = np.arange(T + 1)  # s = 0..T
    mean_ext = np.concatenate([mean_dist, [mean_dist[-1]]])  # repeat t=1 value
    std_ext  = np.concatenate([std_dist,  [std_dist[-1]]])
    # sem_ext  = np.concatenate([sem_dist, [sem_dist[-1]]])

    plt.plot(
        steps,
        mean_ext,
        label=r"$\mathbb{E}\!\left[d_{\perp}(x_t, L_{t,\varepsilon}^{(i,j)})/\sqrt{\varpi}\right]$"
    )

    lower = np.maximum(mean_ext - std_ext, 0.0)
    upper = mean_ext + std_ext
    y_top = np.nanmax(upper[np.isfinite(upper)]) if np.any(np.isfinite(upper)) else 1.0
    # --- label staggering knobs (relative to plot height) ---
    label_dx_black = 0.15 * T     # move black label left/right as you like
    label_dx_red   = 0.35 * T     # make red much farther from black
    label_dy_black = 0.03 * y_top # vertical offset in data units
    label_dy_red   = 0.03 * y_top # bigger vertical offset for red

    y_tau = 0.92 * y_top
    dy_tau = 0.05 * y_top   # vertical staggering step

    plt.fill_between(
        steps,
        lower,
        upper,
        alpha=0.25,
        label="_nolegend_",
    )

    ax = plt.gca()
    ax.set_ylim(bottom=0.0)
    if np.isfinite(y_top) and y_top > 0:
        ax.set_ylim(0.0, 1.05 * y_top)
    ax.set_xlim(0, T)  
    x_text = 0.55 * T 

    def rev_time_formatter(x, pos):
        # rev time formatter.
        # this supports experiment b, which studies convergence toward the nearest mode pair segment.
        xi = int(round(x))
        if abs(x - xi) > 1e-6:
            return ""
        return str(T - xi)  # s=0 -> t=T, s=T -> t=0

    ax.xaxis.set_major_formatter(FuncFormatter(rev_time_formatter))

    xt = list(ax.get_xticks()) # force a tick so t = 0 appears
    if T not in xt:
        ax.set_xticks(sorted(set(xt + [T])))

    LABEL_FONTSIZE = 18
    MINI_LABEL_FONTSIZE = LABEL_FONTSIZE 
    TICK_FONTSIZE  = 18
    LEGEND_FONTSIZE = 20

    plt.xlabel(r"$t$", fontsize=LABEL_FONTSIZE)
    plt.ylabel(r"$d_{\perp}(x_t, L_{t,\varepsilon}^{(i,j)})/\sqrt{\varpi}$", fontsize=LABEL_FONTSIZE)

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)


    s_tau1 = None
    s_tau2 = None
    
    #  tau overlays (allow s=T now) 
    if tau_kappa_dict is not None:
        kappa_vals = np.asarray(tau_kappa_dict["kappa"], dtype=float)

        def step_at_kappa(tau_mean_arr):
            # step at kappa.
            # this supports experiment b, which studies convergence toward the nearest mode pair segment.
            if tau_mean_arr is None:
                return None
            tau_mean_arr = np.asarray(tau_mean_arr, dtype=float)
            mask = ~np.isnan(tau_mean_arr)
            if mask.sum() < 2:
                return None
            tau_k = float(np.interp(kappa_target, kappa_vals[mask], tau_mean_arr[mask]))
            s_k = int(np.clip(np.rint(T - tau_k), 0, T))  # NOTE: clamp to T (not T-1)
            return s_k

        s_tau1 = step_at_kappa(tau_kappa_dict.get("tau_1_mean", None))
        s_tau2 = step_at_kappa(tau_kappa_dict.get("tau_2_mean", None))

        # plot vertical dotted lines for taus
        if s_tau1 is not None:
            plt.axvline(
                s_tau1,
                color="tab:blue",
                linestyle=":",
                alpha=0.9,
                linewidth=2,
                label=rf"$\tau_1 \; (\kappa={kappa_target:g})$",
            )

        if s_tau2 is not None:
            plt.axvline(
                s_tau2,
                color="tab:orange",
                linestyle=":",
                alpha=0.9,
                linewidth=2,
                label=rf"$\tau_2 \; (\kappa={kappa_target:g})$",
            )

        # plot horizontal eps(kappa) line 
        if s_tau2 is not None:
                x0 = 0.85 * T
                x1 = T
                plt.hlines(
                    y=eps_base / dim_scale,
                    xmin=0,
                    xmax=T,          # right edge corresponds to t=0
                    colors="black",
                    linestyles=":",
                    linewidth=2,
                    label=rf"$\varepsilon/\sqrt{{\varpi}}\;(\kappa={kappa_target:g})$"
                )

                # eps (black) label
                plt.text(
                    x_text - label_dx_black,
                    (eps_base / dim_scale) + label_dy_black,
                    rf"{eps_base:.3f}",
                    color="black",
                    fontsize=MINI_LABEL_FONTSIZE,
                    ha="center",
                    va="bottom",
                )
    

     # residual distance at the end of the reverse process 
    if np.isfinite(mean_ext[-1]):
        y_resid = float(mean_ext[-1])

        # draw only near the right end (so it visually reads as "remaining at the end")
        x0 = 0.85 * T
        x1 = T
        plt.hlines(
            y=y_resid,
            xmin=0,
            xmax=T,          # right edge corresponds to t=0
            colors="red",
            linestyles=":",
            linewidth=2,
            label=r"$d_{\perp}(x_T, L_{T,\varepsilon}^{(i,j)})/\sqrt{\varpi}$",
        )


        # residual (red) label
        plt.text(
            x_text + label_dx_red,
            y_resid + label_dy_red,
            rf"{y_resid:.3f}",
            color="red",
            fontsize=MINI_LABEL_FONTSIZE,
            ha="right",
            va="bottom",
        )


    plt.grid(alpha=0.3)

    # tau labels
    fig = plt.gcf()
    fig.canvas.draw()  # needed so legend has a real size

    tau_specs = []
    if s_tau1 is not None:
        tau_specs.append((float(s_tau1), f"{int(round(T - s_tau1))}", "tab:blue"))   
    if s_tau2 is not None:
        tau_specs.append((float(s_tau2), f"{int(round(T - s_tau2))}", "tab:orange")) #
    tau_specs.sort(key=lambda z: z[0])

    pad_band = 0.05 * y_top
    dy_data  = 0.05 * y_top  # vertical stagger if the x's are very close

    level = 0

    min_dx_px = 18  # how close (in screen pixels) before we stagger
    last_x_px = None
    level = 0

    for x_tau, lab, color in tau_specs:
        x_px = ax.transData.transform((x_tau, 0.0))[0]

        if last_x_px is not None and abs(x_px - last_x_px) < min_dx_px:
            level += 1
        else:
            level = 0
        last_x_px = x_px

        idx = int(np.clip(int(round(x_tau)), 0, T))
        upper_here = float(upper[idx]) if np.isfinite(upper[idx]) else 0.0
        y_above_band = upper_here + pad_band

        y_cap = 0.98 * y_top

        # stagger vertically downward if too close
        y = min(max(y_above_band, 0.05 * y_top), y_cap) - level * dy_data
        y = max(y, 0.05 * y_top)

        # keep the DDIM tau_2 label a touch higher so it clears the red terminal residual annotation
        # near the right edge of the convergence plot.
        if color == "tab:orange":
            y = min(y + 0.06 * y_top, y_cap)

        ax.text(
            x_tau,
            y,
            lab,
            color=color,
            fontsize=MINI_LABEL_FONTSIZE,
            ha="center",
            va="bottom",
            rotation=90,
            bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=1.0),
            zorder=20,
            clip_on=True,
        )

    # Grab handles/labels (but do NOT draw legend on the main plot)
    handles, labels = ax.get_legend_handles_labels()

    # Save legend as a separate PNG in same folder
    legend_path = save_legend_pdf(
        handles, labels,
        fig_save_path=fig_save_path,
        prefix="legend_",
        fontsize=LEGEND_FONTSIZE,
        ncol=1
    )

    plt.tight_layout()

    if fig_save_path is not None and fig_save_path != "":
        os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
        plt.savefig(fig_save_path)
    plt.close()

    return {
        "mean_distance_to_line": mean_dist,
        "mean_upper_bound": mean_up_bound,
        "std_distance": std_dist,
        "alpha_bar_list": alpha_bar_list,
        "seq": seq,
        "metadata": {
            "sampling_mode": sampling_mode,
            "skip_type": (skip_type if sampling_mode != "ddpm" else None),
            "ddim_steps": (ddim_steps if sampling_mode != "ddpm" else None),
            "timesteps": timesteps,
            "num_samples": num_samples,
            "T": T,
            "distance_normalization": "sqrt_varpi",
            "invalid_cutoff_mode": cutoff_mode,
            "invalid_sigma_multiple": float(invalid_sigma_multiple),
            "sigma_space": "normalized",
        },
        "eps_err_mean": eps_err_mean,
        "eps_err_p95": eps_err_p95,
        "eps_norm_mean": eps_norm_mean,
        "x0_norm_mean": x0_norm_mean,
    }

def plot_convergence_bound_with_kappa(
    kappa_t_mapping_dict, 
    mean_distance_to_line,     # shape (T,)
    mean_upper_bound,           # shape (T,)
    alpha_t_bar_list=None, 
    fig_save_path=None
):
    # plot convergence bound with kappa for experiment b, which studies convergence back to the closest-pair segment.
    """
    Plot difference between empirical mean distance to line and
    theoretical upper bound as a function of κ.
    """

    T = len(mean_distance_to_line)
    kappa_vals = np.asarray(kappa_t_mapping_dict["kappa"])
    tau_vals = np.asarray(T - kappa_t_mapping_dict["tau_2_mean"]).astype(int)

    T = len(mean_distance_to_line)

    # Clamp \tau to valid range
    tau_vals = np.clip(tau_vals, 0, T - 1)

    # --------------------------------------------------
    # Compute gap at \tau(\kappa)
    # --------------------------------------------------
    empirical_vals = mean_distance_to_line[tau_vals]
    bound_vals = mean_upper_bound[tau_vals]
    alpha_t_bar_val = np.array(alpha_t_bar_list)[tau_vals]

    if alpha_t_bar_list is None: 
        raise ValueError("Scaling factor of alpha t bars is required")
    
    gap = (empirical_vals - bound_vals)/np.sqrt(np.array(alpha_t_bar_val))

    plt.figure(figsize=(6, 4))

    plt_label_ratio = (
        r"$\frac{"
        r"d_{\perp}(x_{\tau(\kappa)}, L^{(i,j)}_{\tau,\varepsilon})"
        r" - "
        r"d_{\perp}(\mathbf{x}_T, L^{(i,j)}_{T,\varepsilon})\,"
        r"\exp(-(T-\tau))"
        r"}{\sqrt{\bar{\alpha}_{\tau}}}$"
    )

    #######  Plot -1 Exponential Convergence with kappa # ### 
    plt.plot(
        kappa_vals,
        gap,
        marker="o",
        linewidth=2,
        label=plt_label_ratio
    )

    plt.axhline(0.0, linestyle="--", color="black", alpha=0.6)

    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"Upper bound of $\frac{varepsilon}{\sqrt{\varpi}}$")

    kappa_gap_dict = {
        "kappa": kappa_vals, 
        "gap": gap
    }

    if fig_save_path is not None:
        stem, _ = os.path.splitext(fig_save_path)
        np.save(stem + ".npy", kappa_gap_dict)
    
    plt.grid(alpha=0.3)
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),  # anchor to the axes' top-right corner
        fontsize=8,
        framealpha=0.9,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=2.0,
    )
    plt.tight_layout()
    
    if fig_save_path is not None: 
        plt.savefig(fig_save_path)

    return {
        "kappa": kappa_vals,
        "tau": tau_vals,
        "empirical": empirical_vals,
        "upper_bound": bound_vals,
        "gap": gap,
    }

def save_legend_pdf(handles, labels, fig_save_path, prefix="legend_", fontsize=18, ncol=1):
    # save legend pdf for experiment b, which studies convergence back to the closest-pair segment.
    """
    Save a standalone legend as a separate PDF in the same directory as fig_save_path.
    Returns the legend_path (or None if nothing to save).
    """
    if fig_save_path is None or fig_save_path == "":
        return None
    if len(handles) == 0 or len(labels) == 0:
        return None

    out_dir = os.path.dirname(fig_save_path)
    base = os.path.basename(fig_save_path)
    legend_path = os.path.join(out_dir, f"{prefix}{base}")
    stem, _ = os.path.splitext(legend_path)
    stale_png = stem + ".png"
    if os.path.exists(stale_png):
        os.remove(stale_png)

    fig_leg = plt.figure(figsize=(1, 1))
    fig_leg.legend(
        handles, labels,
        loc="center",
        frameon=True,
        fontsize=fontsize,
        ncol=ncol,
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)
    return legend_path
