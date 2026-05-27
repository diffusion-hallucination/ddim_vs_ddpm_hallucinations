import os

import matplotlib.pyplot as plt
import numpy as np

from common.artifact_io import write_csv, write_json


def interp_at_target(x_vals: np.ndarray, y_vals: np.ndarray, target: float) -> float:
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    if mask.sum() == 1:
        return float(y[mask][0])
    return float(np.interp(float(target), x[mask], y[mask]))


def trapz_compat(y: np.ndarray, x: np.ndarray | None = None) -> float:
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.size < 2:
        return 0.0
    x_arr = np.arange(y_arr.size, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape for trapezoidal integration")
    return float(np.sum(np.diff(x_arr) * 0.5 * (y_arr[1:] + y_arr[:-1])))


def curve_summary(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return {
            "start": float("nan"),
            "mid": float("nan"),
            "end": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "auc": float("nan"),
        }
    x = np.arange(y.size, dtype=np.float64)
    return {
        "start": float(y[0]),
        "mid": float(y[y.size // 2]),
        "end": float(y[-1]),
        "mean": float(np.mean(y)),
        "max": float(np.max(y)),
        "min": float(np.min(y)),
        "auc": trapz_compat(y, x=x),
    }


def save_two_curve_pdf(
    *,
    x_vals: np.ndarray,
    y1_mean: np.ndarray,
    y1_std: np.ndarray,
    y2_mean: np.ndarray,
    y2_std: np.ndarray,
    x_label: str,
    y_label: str,
    y1_label: str,
    y2_label: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(6.4, 4.6))
    plt.plot(x_vals, y1_mean, linewidth=2, label=y1_label)
    plt.fill_between(x_vals, y1_mean - y1_std, y1_mean + y1_std, alpha=0.25)
    plt.plot(x_vals, y2_mean, linewidth=2, label=y2_label)
    plt.fill_between(x_vals, y2_mean - y2_std, y2_mean + y2_std, alpha=0.25)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_exp_a_solver_artifacts(
    root: str,
    solver_tag: str,
    sampling_mode: str,
    use_exact_score: bool,
    tau_kappa_dict: dict,
    kappa_target: float,
) -> None:
    kappa = np.asarray(tau_kappa_dict["kappa"], dtype=np.float64)
    tau1 = np.asarray(tau_kappa_dict["tau_1_mean"], dtype=np.float64)
    tau2 = np.asarray(tau_kappa_dict["tau_2_mean"], dtype=np.float64)
    payload = {
        "experiment": "a",
        "solver_tag": str(solver_tag),
        "sampling_mode": str(sampling_mode).lower(),
        "use_exact_score": bool(use_exact_score),
        "timesteps": int(tau_kappa_dict.get("metadata", {}).get("timesteps", 0)),
        "ddim_timesteps": tau_kappa_dict.get("metadata", {}).get("ddim_steps"),
        "skip_type": tau_kappa_dict.get("metadata", {}).get("skip_type"),
        "num_dims": tau_kappa_dict.get("metadata", {}).get("num_dims"),
        "tau1_variance_convention": tau_kappa_dict.get("metadata", {}).get("tau1_variance_convention"),
        "tau2_variance_scale_mode": tau_kappa_dict.get("metadata", {}).get("tau2_variance_scale_mode", "none"),
        "tau2_variance_scale_factor": tau_kappa_dict.get("metadata", {}).get("tau2_variance_scale_factor", 1.0),
        "kappa": kappa.tolist(),
        "tau_1_mean": tau1.tolist(),
        "tau_1_std": np.asarray(tau_kappa_dict.get("tau_1_std", []), dtype=np.float64).tolist(),
        "tau_1_sem": np.asarray(tau_kappa_dict.get("tau_1_sem", []), dtype=np.float64).tolist(),
        "tau_2_mean": tau2.tolist(),
        "tau_2_std": np.asarray(tau_kappa_dict.get("tau_2_std", []), dtype=np.float64).tolist(),
        "tau_2_sem": np.asarray(tau_kappa_dict.get("tau_2_sem", []), dtype=np.float64).tolist(),
        "summary": {
            "kappa_target": float(kappa_target),
            "tau_1_at_kappa_target": interp_at_target(kappa, tau1, float(kappa_target)),
            "tau_2_at_kappa_target": interp_at_target(kappa, tau2, float(kappa_target)),
        },
    }
    base = os.path.join(root, f"exp_a_tau_vs_kappa_{solver_tag}")
    write_json(f"{base}.json", payload)
    rows = []
    for idx, kval in enumerate(kappa):
        rows.append({
            "solver_tag": solver_tag,
            "sampling_mode": str(sampling_mode).lower(),
            "use_exact_score": bool(use_exact_score),
            "kappa": float(kval),
            "tau_1_mean": float(tau1[idx]),
            "tau_1_std": float(payload["tau_1_std"][idx]),
            "tau_1_sem": float(payload["tau_1_sem"][idx]),
            "tau_2_mean": float(tau2[idx]),
            "tau_2_std": float(payload["tau_2_std"][idx]),
            "tau_2_sem": float(payload["tau_2_sem"][idx]),
        })
    write_csv(
        f"{base}.csv",
        [
            "solver_tag",
            "sampling_mode",
            "use_exact_score",
            "kappa",
            "tau_1_mean",
            "tau_1_std",
            "tau_1_sem",
            "tau_2_mean",
            "tau_2_std",
            "tau_2_sem",
        ],
        rows,
    )
    save_two_curve_pdf(
        x_vals=kappa,
        y1_mean=tau1,
        y1_std=np.asarray(payload["tau_1_std"], dtype=np.float64),
        y2_mean=tau2,
        y2_std=np.asarray(payload["tau_2_std"], dtype=np.float64),
        x_label=r"$\kappa$",
        y_label=r"$t$",
        y1_label=r"$\mathbb{E}[\tau_1]$",
        y2_label=r"$\mathbb{E}[\tau_2]$",
        save_path=f"{base}.pdf",
    )


def save_exp_a_u_solver_artifacts(
    root: str,
    solver_tag: str,
    sampling_mode: str,
    use_exact_score: bool,
    tau_kappa_dict: dict,
    kappa_target: float,
) -> None:
    kappa = np.asarray(tau_kappa_dict["kappa"], dtype=np.float64)
    u_tau1 = np.asarray(tau_kappa_dict["u_tau_1_mean"], dtype=np.float64)
    u_tau2 = np.asarray(tau_kappa_dict["u_tau_2_mean"], dtype=np.float64)
    reverse_timestep_seq = np.asarray(
        tau_kappa_dict.get("reverse_timestep_seq", tau_kappa_dict.get("seq", [])),
        dtype=np.int64,
    )
    u_seq = np.asarray(tau_kappa_dict.get("u_seq", []), dtype=np.float64)
    payload = {
        "experiment": "a_u",
        "solver_tag": str(solver_tag),
        "sampling_mode": str(sampling_mode).lower(),
        "use_exact_score": bool(use_exact_score),
        "timesteps": int(tau_kappa_dict.get("metadata", {}).get("timesteps", 0)),
        "ddim_timesteps": tau_kappa_dict.get("metadata", {}).get("ddim_steps"),
        "skip_type": tau_kappa_dict.get("metadata", {}).get("skip_type"),
        "num_dims": tau_kappa_dict.get("metadata", {}).get("num_dims"),
        "tau1_variance_convention": tau_kappa_dict.get("metadata", {}).get("tau1_variance_convention"),
        "time_convention": tau_kappa_dict.get("metadata", {}).get(
            "time_convention_u",
            "u(t)=sum_{s=t}^{T-1} beta_s/(2 sigma_s^2), increasing as reverse time approaches 0",
        ),
        "kappa": kappa.tolist(),
        "u_tau_1_mean": u_tau1.tolist(),
        "u_tau_1_std": np.asarray(tau_kappa_dict.get("u_tau_1_std", []), dtype=np.float64).tolist(),
        "u_tau_1_sem": np.asarray(tau_kappa_dict.get("u_tau_1_sem", []), dtype=np.float64).tolist(),
        "u_tau_2_mean": u_tau2.tolist(),
        "u_tau_2_std": np.asarray(tau_kappa_dict.get("u_tau_2_std", []), dtype=np.float64).tolist(),
        "u_tau_2_sem": np.asarray(tau_kappa_dict.get("u_tau_2_sem", []), dtype=np.float64).tolist(),
        "u_seq": u_seq.tolist(),
        "reverse_timestep_seq": reverse_timestep_seq.tolist(),
        "summary": {
            "kappa_target": float(kappa_target),
            "u_tau_1_at_kappa_target": interp_at_target(kappa, u_tau1, float(kappa_target)),
            "u_tau_2_at_kappa_target": interp_at_target(kappa, u_tau2, float(kappa_target)),
        },
    }
    base = os.path.join(root, f"exp_a_u_vs_kappa_{solver_tag}")
    write_json(f"{base}.json", payload)
    rows = []
    for idx, kval in enumerate(kappa):
        rows.append({
            "solver_tag": solver_tag,
            "sampling_mode": str(sampling_mode).lower(),
            "use_exact_score": bool(use_exact_score),
            "kappa": float(kval),
            "u_tau_1_mean": float(u_tau1[idx]),
            "u_tau_1_std": float(payload["u_tau_1_std"][idx]),
            "u_tau_1_sem": float(payload["u_tau_1_sem"][idx]),
            "u_tau_2_mean": float(u_tau2[idx]),
            "u_tau_2_std": float(payload["u_tau_2_std"][idx]),
            "u_tau_2_sem": float(payload["u_tau_2_sem"][idx]),
        })
    write_csv(
        f"{base}.csv",
        [
            "solver_tag",
            "sampling_mode",
            "use_exact_score",
            "kappa",
            "u_tau_1_mean",
            "u_tau_1_std",
            "u_tau_1_sem",
            "u_tau_2_mean",
            "u_tau_2_std",
            "u_tau_2_sem",
        ],
        rows,
    )
    save_two_curve_pdf(
        x_vals=kappa,
        y1_mean=u_tau1,
        y1_std=np.asarray(payload["u_tau_1_std"], dtype=np.float64),
        y2_mean=u_tau2,
        y2_std=np.asarray(payload["u_tau_2_std"], dtype=np.float64),
        x_label=r"$\kappa$",
        y_label=r"$u$",
        y1_label=r"$\mathbb{E}[u(\tau_1)]$",
        y2_label=r"$\mathbb{E}[u(\tau_2)]$",
        save_path=f"{base}.pdf",
    )


def save_exp_b_solver_artifacts(
    root: str,
    solver_tag: str,
    sampling_mode: str,
    use_exact_score: bool,
    convergence_dict: dict,
    tau_kappa_dict: dict | None,
    kappa_target: float,
) -> None:
    curve = np.asarray(convergence_dict["mean_distance_to_line"], dtype=np.float64)
    upper = np.asarray(convergence_dict.get("mean_upper_bound", []), dtype=np.float64)
    std_distance = np.asarray(convergence_dict.get("std_distance", []), dtype=np.float64)
    alpha_bar_list = np.asarray(convergence_dict.get("alpha_bar_list", []), dtype=np.float64)
    eps_err_mean = np.asarray(convergence_dict.get("eps_err_mean", []), dtype=np.float64)
    eps_norm_mean = np.asarray(convergence_dict.get("eps_norm_mean", []), dtype=np.float64)
    seq = np.asarray(convergence_dict.get("seq", np.arange(curve.size)[::-1]), dtype=np.int64)
    progress = np.linspace(0.0, 1.0, curve.size, dtype=np.float64) if curve.size > 0 else np.asarray([], dtype=np.float64)
    summary = curve_summary(curve)
    summary["kappa_target"] = float(kappa_target)
    if tau_kappa_dict is not None:
        tau1_target = interp_at_target(tau_kappa_dict["kappa"], tau_kappa_dict["tau_1_mean"], float(kappa_target))
        tau2_target = interp_at_target(tau_kappa_dict["kappa"], tau_kappa_dict["tau_2_mean"], float(kappa_target))
        summary["tau_1_at_kappa_target"] = tau1_target
        summary["tau_2_at_kappa_target"] = tau2_target
        if curve.size > 0:
            tau1_step = int(np.clip(np.rint(curve.size - float(tau1_target)), 0, curve.size - 1))
            tau2_step = int(np.clip(np.rint(curve.size - float(tau2_target)), 0, curve.size - 1))
            summary["tau_1_overlay_reverse_step_at_kappa_target"] = tau1_step
            summary["tau_2_overlay_reverse_step_at_kappa_target"] = tau2_step
            if seq.size > 0:
                summary["tau_1_overlay_raw_timestep_at_kappa_target"] = int(seq[tau1_step])
                summary["tau_2_overlay_raw_timestep_at_kappa_target"] = int(seq[tau2_step])
        summary["tau_overlay_source_experiment"] = "exp_a_two_mode_assumption"
        summary["tau1_variance_convention"] = tau_kappa_dict.get("metadata", {}).get("tau1_variance_convention")
        summary["tau2_variance_scale_mode"] = tau_kappa_dict.get("metadata", {}).get("tau2_variance_scale_mode")
        summary["tau2_variance_scale_factor"] = tau_kappa_dict.get("metadata", {}).get("tau2_variance_scale_factor")
    else:
        summary["tau_1_at_kappa_target"] = float("nan")
        summary["tau_2_at_kappa_target"] = float("nan")
    payload = {
        "experiment": "b",
        "solver_tag": str(solver_tag),
        "sampling_mode": str(sampling_mode).lower(),
        "use_exact_score": bool(use_exact_score),
        "timesteps": int(convergence_dict.get("metadata", {}).get("timesteps", 0)),
        "ddim_timesteps": convergence_dict.get("metadata", {}).get("ddim_steps"),
        "skip_type": convergence_dict.get("metadata", {}).get("skip_type"),
        "reverse_step": np.arange(curve.size, dtype=np.int64).tolist(),
        "reverse_progress": progress.tolist(),
        "reverse_timestep": seq.tolist(),
        "mean_distance_to_line": curve.tolist(),
        "mean_upper_bound": upper.tolist(),
        "std_distance": std_distance.tolist(),
        "alpha_bar_list": alpha_bar_list.tolist(),
        "eps_err_mean": eps_err_mean.tolist(),
        "eps_norm_mean": eps_norm_mean.tolist(),
        "metadata": convergence_dict.get("metadata", {}),
        "summary": summary,
    }
    base = os.path.join(root, f"exp_b_convergence_curve_{solver_tag}")
    write_json(f"{base}.json", payload)
    rows = []
    for idx, val in enumerate(curve):
        rows.append({
            "solver_tag": solver_tag,
            "sampling_mode": str(sampling_mode).lower(),
            "use_exact_score": bool(use_exact_score),
            "reverse_step": int(idx),
            "reverse_progress": float(progress[idx]),
            "reverse_timestep": int(seq[idx]) if idx < seq.size else "",
            "mean_distance_to_line": float(val),
            "mean_upper_bound": float(upper[idx]) if idx < upper.size else float("nan"),
            "std_distance": float(std_distance[idx]) if idx < std_distance.size else float("nan"),
            "alpha_bar": float(alpha_bar_list[idx]) if idx < alpha_bar_list.size else float("nan"),
            "eps_err_mean": float(eps_err_mean[idx]) if idx < eps_err_mean.size else float("nan"),
            "eps_norm_mean": float(eps_norm_mean[idx]) if idx < eps_norm_mean.size else float("nan"),
        })
    write_csv(
        f"{base}.csv",
        [
            "solver_tag",
            "sampling_mode",
            "use_exact_score",
            "reverse_step",
            "reverse_progress",
            "reverse_timestep",
            "mean_distance_to_line",
            "mean_upper_bound",
            "std_distance",
            "alpha_bar",
            "eps_err_mean",
            "eps_norm_mean",
        ],
        rows,
    )


def exp_c_group_curves(result_dict: dict) -> dict:
    eig = np.asarray(result_dict["all_eigvals"], dtype=np.complex128)
    eig_real = np.real(eig)
    t_axis = np.asarray(result_dict.get("eig_t_axis", np.arange(eig_real.shape[0], dtype=np.int64)), dtype=np.int64)
    groups = {}
    for name, selector in [("positive", lambda v: v > 0), ("negative", lambda v: v < 0)]:
        curves = []
        for p in range(eig_real.shape[1]):
            for e in range(eig_real.shape[2]):
                traj = eig_real[:, p, e]
                if selector(traj[-1]):
                    curves.append(traj)
        if curves:
            arr = np.stack(curves, axis=0)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)
            groups[name] = {
                "count": int(arr.shape[0]),
                "reverse_timestep": t_axis.tolist(),
                "reverse_progress": np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64).tolist(),
                "mean": mean_curve.tolist(),
                "std": std_curve.tolist(),
                "summary": {
                    "terminal_mean": float(mean_curve[-1]),
                    "max_mean": float(np.max(mean_curve)),
                    "min_mean": float(np.min(mean_curve)),
                },
            }
        else:
            groups[name] = {
                "count": 0,
                "reverse_timestep": t_axis.tolist(),
                "reverse_progress": np.linspace(0.0, 1.0, eig_real.shape[0], dtype=np.float64).tolist(),
                "mean": [],
                "std": [],
                "summary": {
                    "terminal_mean": float("nan"),
                    "max_mean": float("nan"),
                    "min_mean": float("nan"),
                },
            }
    return groups


def save_exp_c_solver_artifacts(
    root: str,
    solver_tag: str,
    sampling_mode: str,
    use_exact_score: bool,
    result_dict: dict,
) -> None:
    groups = exp_c_group_curves(result_dict)
    payload = {
        "experiment": "c",
        "solver_tag": str(solver_tag),
        "sampling_mode": str(sampling_mode).lower(),
        "use_exact_score": bool(use_exact_score),
        "timesteps": int(result_dict.get("seq", [0])[0] + 1 if result_dict.get("seq") else 0),
        "ddim_timesteps": len(result_dict.get("seq", [])),
        "groups": groups,
    }
    base = os.path.join(root, f"exp_c_eig_curves_{solver_tag}")
    write_json(f"{base}.json", payload)
    rows = []
    for group_name, group in groups.items():
        mean = np.asarray(group.get("mean", []), dtype=np.float64)
        std = np.asarray(group.get("std", []), dtype=np.float64)
        progress = np.asarray(group.get("reverse_progress", []), dtype=np.float64)
        timesteps = np.asarray(group.get("reverse_timestep", []), dtype=np.int64)
        for idx in range(mean.size):
            rows.append({
                "solver_tag": solver_tag,
                "sampling_mode": str(sampling_mode).lower(),
                "use_exact_score": bool(use_exact_score),
                "group": group_name,
                "reverse_step": int(idx),
                "reverse_progress": float(progress[idx]) if idx < progress.size else float("nan"),
                "reverse_timestep": int(timesteps[idx]) if idx < timesteps.size else "",
                "mean": float(mean[idx]),
                "std": float(std[idx]) if idx < std.size else float("nan"),
                "count": int(group.get("count", 0)),
            })
    write_csv(
        f"{base}.csv",
        [
            "solver_tag",
            "sampling_mode",
            "use_exact_score",
            "group",
            "reverse_step",
            "reverse_progress",
            "reverse_timestep",
            "mean",
            "std",
            "count",
        ],
        rows,
    )


def save_exp_e_solver_artifacts(
    root: str,
    solver_tag: str,
    sampling_mode: str,
    use_exact_score: bool,
    hall_dict: dict,
) -> None:
    payload = {
        "experiment": "e",
        "solver_tag": str(solver_tag),
        "sampling_mode": str(sampling_mode).lower(),
        "use_exact_score": bool(use_exact_score),
        "r_percentage": np.asarray(hall_dict["r_percentage"], dtype=np.float64).tolist(),
        "tau_vals": [int(t) for t in hall_dict["tau_vals"]],
        "tau_internal_vals": [int(t) for t in hall_dict.get("tau_internal_vals", hall_dict["tau_vals"])],
        "hall_mean": np.asarray(hall_dict["hall_mean"], dtype=np.float64).tolist(),
        "hall_std": np.asarray(hall_dict["hall_std"], dtype=np.float64).tolist(),
        "mode_pairs": hall_dict.get("mode_pairs", []),
        "hall_radius_sigma_multiple": float(hall_dict.get("hall_radius_sigma_multiple", 5.0)),
        "tau_interpretation": hall_dict.get("tau_interpretation", "raw_reverse_steps"),
        "paper_tau_labels": [int(t) for t in hall_dict.get("paper_tau_labels", hall_dict["tau_vals"])],
        "internal_start_timestep_by_tau": hall_dict.get("internal_start_timestep_by_tau", {}),
        "tau_internal_lookup": hall_dict.get("tau_internal_lookup", {}),
        "hybrid_mode": hall_dict.get("hybrid_mode"),
        "ddim_eta": float(hall_dict.get("ddim_eta", 0.0)),
        "z_ddpm": int(hall_dict.get("z_ddpm", 0)),
        "z_ddpm_requested": int(hall_dict.get("z_ddpm_requested", 0)),
        "z_ddpm_applied_by_tau": hall_dict.get("z_ddpm_applied_by_tau", {}),
        "tau_execution_plans": hall_dict.get("tau_execution_plans", {}),
        "pair_selection": hall_dict.get("pair_selection"),
        "midpoint_init": hall_dict.get("midpoint_init"),
        "start_geometry": hall_dict.get("start_geometry"),
        "line_point_rule": hall_dict.get("line_point_rule"),
        "radius_percentage_basis": hall_dict.get("radius_percentage_basis"),
        "start_timestep": hall_dict.get("start_timestep"),
        "start_geometry_space": hall_dict.get("start_geometry_space"),
        "num_mode_pairs": hall_dict.get("num_mode_pairs"),
        "closest_pair_distance": hall_dict.get("closest_pair_distance"),
    }
    base = os.path.join(root, f"exp_e_hall_radius_{solver_tag}")
    write_json(f"{base}.json", payload)
    rows = []
    hall_mean = np.asarray(payload["hall_mean"], dtype=np.float64)
    hall_std = np.asarray(payload["hall_std"], dtype=np.float64)
    r_vals = np.asarray(payload["r_percentage"], dtype=np.float64)
    tau_vals = payload["tau_vals"]
    tau_internal_vals = payload["tau_internal_vals"]
    z_applied_by_tau = payload.get("z_ddpm_applied_by_tau", {}) or {}
    for r_idx, r in enumerate(r_vals):
        for t_idx, tau in enumerate(tau_vals):
            rows.append({
                "solver_tag": solver_tag,
                "sampling_mode": str(sampling_mode).lower(),
                "use_exact_score": bool(use_exact_score),
                "z_ddpm": int(payload.get("z_ddpm", 0)),
                "z_ddpm_applied": int(z_applied_by_tau.get(str(int(tau)), 0)),
                "tau": int(tau),
                "internal_tau_steps": int(tau_internal_vals[t_idx]) if t_idx < len(tau_internal_vals) else "",
                "r_percentage": float(r),
                "hall_mean": float(hall_mean[r_idx, t_idx]),
                "hall_std": float(hall_std[r_idx, t_idx]),
            })
    write_csv(
        f"{base}.csv",
        [
            "solver_tag",
            "sampling_mode",
            "use_exact_score",
            "z_ddpm",
            "z_ddpm_applied",
            "tau",
            "internal_tau_steps",
            "r_percentage",
            "hall_mean",
            "hall_std",
        ],
        rows,
    )
