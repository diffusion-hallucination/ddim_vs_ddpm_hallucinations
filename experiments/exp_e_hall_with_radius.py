import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from common.reverse_solvers import build_reverse_solver_stepper
from common.gaussian_mixture_2d import pair_radii_from_sigmas
from common.gaussian_pairs import find_all_closest_mode_pairs

LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12
LEGEND_FONTSIZE = 12
DEFAULT_EXP_E_R_PERCENTAGE = [5, 15, 25, 35, 45, 50]
EXP_E_OVERLAY_COLORS = {
    "DDIM": "#1f77b4",
    "DDIM + 2 DDPM steps": "#ff7f0e",
    "DDIM + 5 DDPM steps": "#2ca02c",
    "DDIM + 8 DDPM steps": "#d62728",
    "DDPM": "#9467bd",
}


def segment_points_on_L_t(
    mode_a: np.ndarray,
    mode_b: np.ndarray,
    start_alpha_bar: float,
    r_percentage: float,
    tol: float = 1e-10,
):
    # build the two symmetric start points on the pair segment $L_t$ used in exp. e.
    # the radius is expressed as a percentage of the time-$t$ pair length $\ell_t$.
    mode_a = np.asarray(mode_a, dtype=np.float64)
    mode_b = np.asarray(mode_b, dtype=np.float64)
    start_alpha_bar = float(start_alpha_bar)
    r_percentage = float(r_percentage)

    if r_percentage < 0.0:
        raise ValueError(f"r_percentage must be non-negative, got {r_percentage}")
    if r_percentage > 50.0 + float(tol):
        raise ValueError(
            f"r_percentage={r_percentage} lies off the line segment; valid values are <= 50."
        )

    # map the clean pair into the time-$t$ geometry before placing the start points.
    start_scale = float(np.sqrt(max(start_alpha_bar, 0.0)))
    a_t = start_scale * mode_a
    b_t = start_scale * mode_b
    midpoint_t = 0.5 * (a_t + b_t)
    delta_t = b_t - a_t
    segment_length_t = float(np.linalg.norm(delta_t))
    if not np.isfinite(segment_length_t) or segment_length_t <= float(tol):
        raise ValueError("Degenerate segment in exp_e; closest-pair distance is zero.")

    direction_t = delta_t / segment_length_t
    distance_from_midpoint_t = (r_percentage / 100.0) * segment_length_t
    if distance_from_midpoint_t > 0.5 * segment_length_t + float(tol):
        raise ValueError(
            f"distance_from_midpoint_t={distance_from_midpoint_t} exceeds half the segment length."
        )

    # place the two start points symmetrically about the midpoint on the pair axis.
    start_points = np.stack(
        [
            midpoint_t + distance_from_midpoint_t * direction_t,
            midpoint_t - distance_from_midpoint_t * direction_t,
        ],
        axis=0,
    )
    return {
        "start_points": start_points,
        "midpoint_t": midpoint_t,
        "a_t": a_t,
        "b_t": b_t,
        "direction_t": direction_t,
        "segment_length_t": segment_length_t,
        "distance_from_midpoint_t": distance_from_midpoint_t,
        "start_scale": start_scale,
    }


def is_pure_deterministic_ddim(model=None, sampling_mode: str = "ddim", z_ddpm: int = 0, ddim_eta: float = 0.0) -> bool:
    # detect the deterministic ddpm-free ddim regime used for the two-point line starts.
    # in this case we keep one copy of each symmetric start point instead of a large batch.
    if bool(getattr(model, "stochastic_model", False)):
        return False
    return (
        str(sampling_mode).lower() == "ddim"
        and int(z_ddpm) == 0
        and abs(float(ddim_eta)) <= 1e-12
    )


def build_segment_start_batch(start_points: np.ndarray, num_samples: int, deterministic: bool = False):
    # expand the two symmetric line-start points into a balanced batch for exp. e.
    # the side ids keep track of which half of the pair segment each start came from.
    start_points = np.asarray(start_points, dtype=np.float64)
    if start_points.shape[0] != 2:
        raise ValueError(f"Expected two symmetric segment start points, got shape {tuple(start_points.shape)}")

    if deterministic:
        counts = (1, 1)
    else:
        total = max(int(num_samples), 2)
        left = max(total // 2, 1)
        right = max(total - left, 1)
        counts = (left, right)

    # repeat each symmetric start point according to the side counts.
    batch = np.concatenate(
        [
            np.repeat(start_points[0:1], counts[0], axis=0),
            np.repeat(start_points[1:2], counts[1], axis=0),
        ],
        axis=0,
    )
    side_ids = np.concatenate(
        [
            np.zeros(counts[0], dtype=np.int64),
            np.ones(counts[1], dtype=np.int64),
        ],
        axis=0,
    )
    return batch, side_ids, counts


def num_pairs(res: dict) -> int:
    # return the number of mode pairs contributing to one exp. e summary.
    return int(len(res.get("mode_pairs", [])))

def band_from_std(res: dict, std_vec: np.ndarray, band: str) -> np.ndarray:
    # convert pairwise standard deviations into the plotting band.
    # exp. e/k plots can show either the raw pairwise std or the corresponding sem.
    """
    std_vec: shape (len(r_percentage),)
    band: "sem" or "std" or "none"
    """
    band = str(band).lower()
    if band == "none":
        return np.zeros_like(std_vec)

    if band == "std":
        return std_vec

    if band == "sem":
        n = num_pairs(res)
        # if n==0 or n==1, std_vec will typically be 0 anyway; keep it safe
        denom = np.sqrt(float(max(n, 1)))
        return std_vec / denom

    raise ValueError(f"Unknown band: {band}")


def check_hallucination(mode_a, mode_b, final_sample, pair_radius=None): 
    # classify whether the final sample lies outside the allowed pair neighborhood.
    # for exp. e this is the pairwise hallucination test attached to the selected mode pair.
    if torch.is_tensor(mode_a):
        mode_a = mode_a.detach().cpu().numpy()
    if torch.is_tensor(mode_b):
        mode_b = mode_b.detach().cpu().numpy()
    if torch.is_tensor(final_sample):
        final_sample = final_sample.detach().cpu().numpy()

    modes = np.asarray([mode_a, mode_b], dtype=np.float64)
    final_sample = np.asarray(final_sample, dtype=np.float64)

    if final_sample.ndim == 1:
        final_sample = final_sample[None, :]

    if pair_radius is None:
        raise ValueError("pair_radius is required for hallucination checks.")

    diff = final_sample[:, None, :] - modes[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dist = np.min(dists, axis=1)
    return min_dist > float(pair_radius)

@torch.no_grad()
def compute_hallucination_with_radius(
    dataset_name,
    diffusion, 
    r_percentage = DEFAULT_EXP_E_R_PERCENTAGE,
    tau_vals = [5, 20, 40],
    tau_internal_vals=None,
    sampling_mode="ddim", 
    num_samples=10000,
    std_dev=0.02,
    model=None,
    keep_percent=80,
    z_ddpm: int = 0,
    ddim_steps: int = 50,   
    skip_type: str = "quad",  
    ddim_eta: float = 0.0,  
    hall_radius_sigma_multiple: float = 5.0,
    gaussian_modes=None,
    mode_sigmas=None,
):
    # run exp. e by sweeping midpoint radii and reverse-time horizons for one solver family.
    # this is the main gaussian25 routine behind the midpoint-radius curves in the paper.
    # starts are placed directly on the time-scaled pair segment L_t.
    stepper_cache = {}
    execution_plan_cache = {}
    # tau_vals are the paper-facing labels shown in saved artifacts. Internally, we execute a different raw DDPM horizon
    # while keeping the displayed tau label tied to the DDIM time from exp B
    tau_labels = [int(v) for v in tau_vals]
    if tau_internal_vals is None:
        tau_internal_vals = list(tau_labels)
    tau_internal_vals = [int(v) for v in tau_internal_vals]
    if len(tau_internal_vals) != len(tau_labels):
        raise ValueError(
            f"tau_internal_vals must match tau_vals in length, got {len(tau_internal_vals)} vs {len(tau_labels)}."
        )

    
    def get_stepper(mode, ddim_steps, skip_type, ddim_eta):
        # cache the reverse solver so repeated $(\tau, r)$ evaluations reuse the same stepper.
        key = (
            str(mode).lower(),
            int(diffusion.steps),
            int(ddim_steps),
            str(skip_type),
            float(ddim_eta),
        )
        if key not in stepper_cache:
            stepper_cache[key] = build_reverse_solver_stepper(
                diffusion=diffusion,
                sampling_mode=str(mode).lower(),
                timesteps=int(diffusion.steps),
                ddim_steps=int(ddim_steps),
                skip_type=str(skip_type),
                ddim_eta=float(ddim_eta),
            )
        return stepper_cache[key]

    def build_tau_execution_plan(
        tau_steps,
        sampling_mode,
        z_ddpm: int = 0,
        ddim_steps: int = 50,
        skip_type: str = "quad",
        ddim_eta: float = 0.0,
    ):
        # build the exact reverse-time plan executed for one tau configuration.
        # hybrid DDIM variants replace the first z raw tail steps with DDPM steps.
        base_mode = str(sampling_mode).lower()
        if base_mode in {"ddim_ddpm_mix", "ddim+ddpm", "mix"}:
            base_mode = "ddim"
        if base_mode not in {"ddim", "ddpm"}:
            raise ValueError("Experiment E supports only sampling_mode='ddim' or 'ddpm'.")

        tau_steps = min(max(int(tau_steps), 0), int(diffusion.steps))
        cache_key = (
            base_mode,
            int(tau_steps),
            int(z_ddpm),
            int(ddim_steps),
            str(skip_type),
            float(ddim_eta),
        )
        if cache_key in execution_plan_cache:
            return execution_plan_cache[cache_key]

        # restrict the global reverse schedule to the first $\tau$ raw reverse steps.
        seq_full, _ = get_stepper(base_mode, ddim_steps, skip_type, ddim_eta)
        seq_local = [int(s) for s in seq_full if int(s) < tau_steps]
        transitions = []
        z_applied = 0

        if tau_steps > 0 and len(seq_local) > 0:
            if base_mode == "ddpm":
                # ddpm uses the full raw reverse chain with no skipped transitions.
                for idx, t_eff in enumerate(seq_local):
                    t_next = int(seq_local[idx + 1]) if (idx + 1) < len(seq_local) else -1
                    transitions.append({
                        "kind": "ddpm",
                        "t_from": int(t_eff),
                        "t_to": int(t_next),
                    })
            else:
                current_t = int(max(tau_steps - 1, 0))
                while z_applied < int(z_ddpm) and current_t > 0:
                    transitions.append({
                        "kind": "ddpm",
                        "t_from": int(current_t),
                        "t_to": int(current_t - 1),
                    })
                    current_t -= 1
                    z_applied += 1

                # execute the restricted DDIM tail after any DDPM prefix replacement.
                ddim_tail = [int(current_t)] + [int(s) for s in seq_local if int(s) < int(current_t)]
                for idx, t_eff in enumerate(ddim_tail):
                    t_next = int(ddim_tail[idx + 1]) if (idx + 1) < len(ddim_tail) else -1
                    transitions.append({
                        "kind": base_mode,
                        "t_from": int(t_eff),
                        "t_to": int(t_next),
                    })

        # keep the executed nodes explicitly so downstream reports can reconstruct the raw path.
        executed_nodes = []
        if transitions:
            executed_nodes.append(int(transitions[0]["t_from"]))
            for tr in transitions:
                if int(tr["t_to"]) >= 0:
                    executed_nodes.append(int(tr["t_to"]))

        hybrid_mode = "restricted_solver_tail"
        if base_mode == "ddpm":
            hybrid_mode = "ddpm_full_raw_chain"
        elif int(z_ddpm) > 0:
            hybrid_mode = f"{base_mode}_prefix_replaced_with_ddpm"

        plan = {
            "tau_interpretation": "raw_reverse_steps",
            "tau_raw_steps": int(tau_steps),
            "tau_raw_interval": [int(max(tau_steps - 1, 0)), 0] if tau_steps > 0 else [],
            "sampling_mode": base_mode,
            "restricted_solver_nodes": [int(s) for s in seq_local],
            "hybrid_mode": hybrid_mode,
            "z_ddpm_requested": int(z_ddpm),
            "z_ddpm_applied": int(z_applied),
            "executed_raw_timesteps": executed_nodes,
            "executed_raw_transitions": transitions,
        }
        execution_plan_cache[cache_key] = plan
        return plan

    def sample_tau_steps_of_reverse(
        x, diffusion, tau_steps, sampling_mode,
        z_ddpm: int = 0,
        ddim_steps: int = 50,
        skip_type: str = "quad",
        ddim_eta: float = 0.0,
    ):
        # execute the cached reverse-step plan for one batch of start points.
        # this is the routine that actually walks the solver from the chosen start set to $x_0$.
        base_mode = str(sampling_mode).lower()
        if base_mode in {"ddim_ddpm_mix", "ddim+ddpm", "mix"}:
            base_mode = "ddim"
        if base_mode not in {"ddim", "ddpm"}:
            raise ValueError("Experiment E supports only sampling_mode='ddim' or 'ddpm'.")

        plan = build_tau_execution_plan(
            tau_steps=tau_steps,
            sampling_mode=base_mode,
            z_ddpm=z_ddpm,
            ddim_steps=ddim_steps,
            skip_type=skip_type,
            ddim_eta=ddim_eta,
        )
        if len(plan["executed_raw_transitions"]) == 0:
            return x, plan

        # keep the DDPM stepper available when the solver is DDPM.
        _, step_fn_base = get_stepper(base_mode, ddim_steps, skip_type, ddim_eta)
        _, step_fn_ddpm = get_stepper("ddpm", ddim_steps, skip_type, ddim_eta)

        prev_x0 = None
        prev_h = None
        for tr in plan["executed_raw_transitions"]:
            # switch steppers only when the hybrid plan asks for ddpm.
            step_kind = str(tr["kind"]).lower()
            t_eff = int(tr["t_from"])
            t_next = int(tr["t_to"])
            if step_kind == "ddpm":
                x, _, _ = step_fn_ddpm(
                    model=model,
                    x_t=x,
                    t_eff=t_eff,
                    t_next=t_next,
                    prev_x0=None,
                    prev_h=None,
                )
                prev_x0 = None
                prev_h = None
            else:
                x, prev_x0, prev_h = step_fn_base(
                    model=model,
                    x_t=x,
                    t_eff=t_eff,
                    t_next=t_next,
                    prev_x0=prev_x0,
                    prev_h=prev_h,
                )
        return x, plan

    
    # exp. e only permits starts that stay on or within the segment, so $r \le 50$.
    raw_r_percentage = [float(r) for r in r_percentage]
    r_percentage = [float(r) for r in raw_r_percentage if float(r) <= 50.0 + 1e-9]
    if len(r_percentage) == 0:
        raise ValueError("exp_e requires at least one radius percentage <= 50.")
    dropped_r = [float(r) for r in raw_r_percentage if float(r) > 50.0 + 1e-9]
    if dropped_r:
        print(f"[exp_e] Dropping radius percentages above 50 because starts must lie on the segment: {dropped_r}")

    # load the normalized gaussian modes and the pairwise hallucination thresholds.
    if gaussian_modes is None:
        raise ValueError("gaussian_modes must be provided for experiment E.")
    modes = np.asarray(gaussian_modes, dtype=np.float64)
    mode_sigmas_arr = None if mode_sigmas is None else np.asarray(mode_sigmas, dtype=np.float64).reshape(-1)
    pair_radii = None if mode_sigmas_arr is None else pair_radii_from_sigmas(mode_sigmas_arr, sigma_multiple=float(hall_radius_sigma_multiple))

    adjacent_pairs, closest_pair_distance = find_all_closest_mode_pairs(modes)

    sampling_mode = sampling_mode.lower()
    if sampling_mode in {"ddim_ddpm_mix", "ddim+ddpm", "mix"}:
        sampling_mode = "ddim"
    if sampling_mode not in {"ddim", "ddpm"}:
        raise ValueError("Experiment E supports only sampling_mode='ddim' or 'ddpm'.")
    if model is None:
        if isinstance(diffusion, (tuple, list)) and len(diffusion) == 2:
            model, diffusion = diffusion
        elif hasattr(diffusion, "model"):
            model = diffusion.model
    if model is None:
        raise ValueError("compute_hallucination_with_radius requires a model or (model, diffusion).")

    # accumulate per-pair, per-radius, per-\tau artifacts before averaging over pairs.
    radius_hall_results = []
    hall_mean = np.zeros((len(r_percentage), len(tau_vals)), dtype=np.float64)
    hall_std = np.zeros((len(r_percentage), len(tau_vals)), dtype=np.float64)

    device = diffusion.betas.device
    model.eval()


    tau_execution_plans = {}
    tau_internal_lookup = {str(int(label)): int(internal) for label, internal in zip(tau_labels, tau_internal_vals)}
    per_midpoint_means = []
    display_r_percentage_by_tau = {
        str(int(tau)): [float(r) for r in r_percentage]
        for tau in tau_labels
    }
    for pair_idx, (i, j) in enumerate(adjacent_pairs):
        print("Computing pair", pair_idx)
        mode_a = modes[i]
        mode_b = modes[j]
        mid_means = np.zeros((len(r_percentage), len(tau_vals)), dtype=np.float64)

        # sweep the midpoint radii and reverse horizons for this one pair.
        for r_idx, r in enumerate(r_percentage):
            for t_idx, tau_label in enumerate(tau_labels):
                tau_steps = int(tau_internal_vals[t_idx])
                if tau_steps <= 0:
                    raise ValueError("Experiment E requires tau >= 1.")
                # Paper tau counts raw reverse steps remaining, i.e. x_tau -> x_{tau-1} -> ... -> x_0.
                # The solver buffers here are 0-indexed over the noisy states only, so code index 0 is
                # the final noisy state before the terminal x_0 reconstruction. Therefore the state that
                # corresponds to paper x_tau lives at raw code index tau-1.
                start_timestep = int(max(tau_steps - 1, 0))
                start_alpha_bar = float(diffusion.alphas_prod[start_timestep].detach().cpu().item())
                # use the two exact line-start points on L_t.
                segment_geom = segment_points_on_L_t(
                    mode_a=mode_a,
                    mode_b=mode_b,
                    start_alpha_bar=start_alpha_bar,
                    r_percentage=float(r),
                )
                deterministic_ddim = is_pure_deterministic_ddim(
                    model=model,
                    sampling_mode=sampling_mode,
                    z_ddpm=z_ddpm,
                    ddim_eta=ddim_eta,
                )
                init_samples, side_ids, side_counts = build_segment_start_batch(
                    segment_geom["start_points"],
                    num_samples=num_samples,
                    deterministic=deterministic_ddim,
                )
                radius_value = float(segment_geom["distance_from_midpoint_t"])
                midpoint_value = segment_geom["midpoint_t"].tolist()
                start_points_value = segment_geom["start_points"].tolist()
                segment_length_t_value = float(segment_geom["segment_length_t"])
                distance_from_midpoint_t_value = float(segment_geom["distance_from_midpoint_t"])
                side_rates = []
                if init_samples.shape[0] == 0:
                    hall_rate = float("nan")
                    hall_std_dev = float("nan")
                    pair_threshold = float(pair_radii[i, j]) if pair_radii is not None else float(hall_radius_sigma_multiple) * float(std_dev)
                    exec_plan = build_tau_execution_plan(
                        tau_steps=tau_steps,
                        sampling_mode=sampling_mode,
                        z_ddpm=z_ddpm,
                        ddim_steps=ddim_steps,
                        skip_type=skip_type,
                        ddim_eta=ddim_eta,
                    )
                else:
                    # propagate the chosen start set through the reverse-time horizon.
                    x_init = torch.tensor(init_samples, device=device, dtype=torch.float32)
                    x, exec_plan = sample_tau_steps_of_reverse(
                        x_init, diffusion, tau_steps, sampling_mode,
                        z_ddpm=z_ddpm,
                        ddim_steps=ddim_steps,
                        skip_type=skip_type,
                        ddim_eta=ddim_eta,
                    )
                    pair_threshold = float(pair_radii[i, j]) if pair_radii is not None else float(hall_radius_sigma_multiple) * float(std_dev)
                    hall_mask = check_hallucination(mode_a, mode_b, x, pair_radius=pair_threshold)
                    hall_rate = float(np.mean(hall_mask))
                    hall_std_dev = float(np.std(hall_mask))
                    side_rates = []
                    for side_idx in (0, 1):
                        mask = side_ids == side_idx
                        if np.any(mask):
                            side_rates.append(float(np.mean(hall_mask[mask])))
                        else:
                            side_rates.append(float("nan"))
                exec_plan = {
                    **exec_plan,
                    "paper_tau_label": int(tau_label),
                    "internal_tau_steps": int(tau_steps),
                    "internal_start_timestep": int(max(tau_steps - 1, 0)),
                }
                tau_execution_plans[str(int(tau_label))] = exec_plan

                # store the pairwise rate before later averaging across the selected mode pairs.
                mid_means[r_idx, t_idx] = hall_rate

                radius_hall_results.append(
                    {
                        "pair_index": int(pair_idx),
                        "mode_pair": [int(i), int(j)],
                        "midpoint": midpoint_value,
                        "r_percentage": float(r),
                        "radius": radius_value,
                        "tau": int(tau_label),
                        "internal_tau_steps": int(tau_steps),
                        "hall_mean": hall_rate,
                        "hall_std": hall_std_dev,
                        "start_points": start_points_value,
                        "start_timestep": int(start_timestep),
                        "start_alpha_bar": float(start_alpha_bar),
                        "segment_length_t": segment_length_t_value,
                        "distance_from_midpoint_t": distance_from_midpoint_t_value,
                        "pair_side_hall_rates": side_rates,
                        "pair_side_counts": [int(side_counts[0]), int(side_counts[1])],
                        "z_ddpm_requested": int(z_ddpm),
                        "z_ddpm_applied": int(exec_plan.get("z_ddpm_applied", 0)),
                    }
                )

        per_midpoint_means.append(mid_means)

    per_midpoint_means = np.asarray(per_midpoint_means, dtype=np.float64)
    if per_midpoint_means.size == 0:
        raise ValueError("No adjacent mode pairs found for hallucination computation.")
    # average over mode pairs.
    hall_mean = np.nanmean(per_midpoint_means, axis=0)
    hall_std = np.nanstd(per_midpoint_means, axis=0)
    model.train()
    return {
        "radius_hall_results": radius_hall_results,
        "hall_mean": hall_mean,
        "hall_std": hall_std,
        "r_percentage": list(r_percentage),
        "tau_vals": list(tau_labels),
        "tau_internal_vals": list(tau_internal_vals),
        "mode_pairs": [list(p) for p in adjacent_pairs],
        "ddim_steps": int(ddim_steps),
        "skip_type": str(skip_type),
        "ddim_eta": float(ddim_eta),
        "sampling_mode": str(sampling_mode),
        "z_ddpm": int(z_ddpm),
        "z_ddpm_requested": int(z_ddpm),
        "z_ddpm_applied_by_tau": {
            str(int(tau)): int((tau_execution_plans.get(str(int(tau))) or {}).get("z_ddpm_applied", 0))
            for tau in tau_labels
        },
        "tau_interpretation": "raw_reverse_steps",
        "paper_tau_labels": list(tau_labels),
        "internal_start_timestep_by_tau": {
            str(int(label)): int(max(int(internal) - 1, 0))
            for label, internal in zip(tau_labels, tau_internal_vals)
        },
        "hybrid_mode": (
            "ddpm_full_raw_chain"
            if str(sampling_mode).lower() == "ddpm"
            else ("ddim_prefix_replaced_with_ddpm" if int(z_ddpm) > 0 else "restricted_solver_tail")
        ),
        "tau_internal_lookup": tau_internal_lookup,
        "tau_execution_plans": tau_execution_plans,
        "hall_radius_sigma_multiple": float(hall_radius_sigma_multiple),
        "hall_radius_space": "normalized_x" if pair_radii is not None else "std_dev_units",
        "pair_selection": "all_closest_pairs",
        "midpoint_init": "segment_points_on_L_t",
        "start_geometry": "line_segment",
        "line_point_rule": "two_exact_points",
        "radius_percentage_basis": "full_pair_distance",
        "start_timestep": "tau_minus_one",
        "start_geometry_space": "normalized_x_t",
        "display_r_percentage_by_tau": display_r_percentage_by_tau,
        "num_mode_pairs": int(len(adjacent_pairs)),
        "closest_pair_distance": (None if not np.isfinite(closest_pair_distance) else float(closest_pair_distance)),
    }



def plot_hallucination_results(
    plot_result_dict,
    save_folder,
    sampling_mode,
    keep_percent=80,
    band: str = "std",     
    alpha: float = 0.18,    # nice looking fill
    max_tau: int | None = None,
):
    # plot the exp. e hallucination curves for one solver family across several $\tau$ values.
    # the x-axis is the midpoint radius as a percentage of the time-$t$ pair length $\ell_t$.
    # keep the legend slightly smaller and tighter so it interferes less with the radius curves.
    os.makedirs(save_folder, exist_ok=True)
    tau_vals = [int(t) for t in plot_result_dict["tau_vals"]]
    hall_mean = np.asarray(plot_result_dict["hall_mean"], dtype=np.float64)
    hall_std  = np.asarray(plot_result_dict["hall_std"], dtype=np.float64)
    display_r_by_tau = plot_result_dict.get("display_r_percentage_by_tau", {}) or {}
    if max_tau is not None:
        keep_idx = [idx for idx, tau in enumerate(tau_vals) if int(tau) <= int(max_tau)]
        if not keep_idx:
            raise ValueError(f"No tau values <= {int(max_tau)} available for {sampling_mode}.")
        tau_vals = [tau_vals[idx] for idx in keep_idx]
        if hall_mean.ndim == 2:
            hall_mean = hall_mean[:, keep_idx]
        if hall_std.ndim == 2:
            hall_std = hall_std[:, keep_idx]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for idx, tau in enumerate(tau_vals):
        r_vals = np.asarray(plot_result_dict["r_percentage"], dtype=np.float64)
        tau_display = display_r_by_tau.get(str(int(tau)))
        if tau_display is not None:
            tau_display_arr = np.asarray(tau_display, dtype=np.float64)
            if tau_display_arr.shape == r_vals.shape:
                r_vals = tau_display_arr

        # each column corresponds to one reverse horizon $\tau$.
        if hall_mean.ndim == 2:
            y = hall_mean[:, idx]
            s = hall_std[:, idx] if hall_std.shape == hall_mean.shape else np.zeros_like(y)
        else:
            y = hall_mean
            s = hall_std if hall_std.shape == hall_mean.shape else np.zeros_like(y)

        # convert the stored pairwise variability into either std or sem bands.
        band_vec = band_from_std(plot_result_dict, s, band=band)

        y_lo = np.clip(y - band_vec, 0.0, 1.0)
        y_hi = np.clip(y + band_vec, 0.0, 1.0)

        ax.plot(r_vals, y, marker="o", linewidth=2, label=rf"$\tau_3={int(tau)}$")
        if band.lower() != "none":
            ax.fill_between(r_vals, y_lo, y_hi, alpha=alpha)

    ax.set_xlabel(r"Radius from midpoint (% of $\ell_t$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hallucination rate", fontsize=LABEL_FONTSIZE)

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 50.0)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="best",
        fontsize=max(LEGEND_FONTSIZE - 2, 8),
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.4,
        handletextpad=0.45,
    )

    plt.tight_layout()

    save_path = os.path.join(save_folder, f"{sampling_mode}_hall_with_radius.pdf")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_hallucination_overlay_single_tau(
    results_by_label,
    save_path,
    tau_target: int = 11,
    band: str = "std",     # "std" or "sem"
    alpha: float = 0.22,
    show_title: bool = True,
):
    # overlay several exp. e curves at one fixed reverse horizon $\tau$.
    # this is the single-$\tau$ view used to compare solver families directly.
    """
    results_by_label: dict[label -> result_dict from compute_hallucination_with_radius]
    Plots one curve per label at a single tau (tau_target), with shaded confidence bands.

    band="std": uses hall_std directly (std across adjacent pairs)
    band="sem": uses hall_std / sqrt(#pairs)
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6.5, 4.2))
    ax = plt.gca()

    for label, res in results_by_label.items():
        r_vals = np.asarray(res["r_percentage"], dtype=np.float64)
        tau_vals = [int(t) for t in res["tau_vals"]]
        hall_mean = np.asarray(res["hall_mean"], dtype=np.float64)
        hall_std  = np.asarray(res["hall_std"], dtype=np.float64)

        if tau_target not in tau_vals:
            raise ValueError(f"{label}: tau_target={tau_target} not in tau_vals={tau_vals}")

        j = tau_vals.index(int(tau_target))

        display_r_by_tau = res.get("display_r_percentage_by_tau", {}) or {}
        tau_display = display_r_by_tau.get(str(int(tau_target)))
        if tau_display is not None:
            tau_display_arr = np.asarray(tau_display, dtype=np.float64)
            if tau_display_arr.shape == r_vals.shape:
                r_vals = tau_display_arr

        y = hall_mean[:, j]
        s = hall_std[:, j] if hall_std.shape == hall_mean.shape else None
        if s is None:
            # fallback: no band
            s = np.zeros_like(y)

        if band.lower() == "sem":
            n_pairs = len(res.get("mode_pairs", []))
            if n_pairs > 0:
                s = s / np.sqrt(float(n_pairs))

        # clip to [0,1] to avoid negative fill showing up like your current errorbars
        y_lo = np.clip(y - s, 0.0, 1.0)
        y_hi = np.clip(y + s, 0.0, 1.0)

        color = EXP_E_OVERLAY_COLORS.get(str(label))
        ax.plot(r_vals, y, marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(r_vals, y_lo, y_hi, alpha=alpha, color=color)

    ax.set_xlabel(r"Radius from midpoint (% of $\ell_t$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hallucination rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 50.0)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="best",
        fontsize=max(LEGEND_FONTSIZE - 2, 8),
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.4,
        handletextpad=0.45,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
