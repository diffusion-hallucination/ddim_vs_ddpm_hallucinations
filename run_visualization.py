import os
import sys
import re
import json

import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd())

from common.artifact_io import write_csv


LABEL_FONTSIZE = 18
TICK_FONTSIZE  = 16
LEGEND_FONTSIZE = 14
LINEWIDTH = 2.5
MARKERSIZE = 7


def expected_ddim_grid(cfg: DictConfig) -> list[int]:
    # parse the DDIM-step grid for the interpolation-rate sweep.
    values = getattr(cfg, "expected_ddim_steps_grid", None)
    if values in (None, "", "null"):
        return list(range(50, 1001, 50))
    return sorted({int(v) for v in values})


def plot_ddim_interpolation_rate_sweep(cfg: DictConfig):
    # plot and tabulate the ddim hallucination rates over the step grid.

    results_dir = getattr(cfg, "results_dir", "eval_results")
    run_name = cfg.run_name

    if run_name is None:
        raise ValueError("Missing run_name (or dataset) in visualization config.")

    run_folder = os.path.join(results_dir, run_name)
    samples_folder = os.path.join(run_folder, "samples")
    if not os.path.isdir(samples_folder):
        raise ValueError(f"Missing learned samples folder under {run_folder}: expected {samples_folder}")

    def parse_counts(path):
        # parse counts.
        # this supports the visualization entrypoint for the paper's DDIM interpolation-rate figure.
        counts = {}
        with open(path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if not value:
                    continue
                try:
                    counts[key] = int(value)
                except ValueError:
                    continue
        return counts

    step_to_rates = {}
    ddim_pattern = re.compile(r"ddim_hallucination(?:_\d+)?_(\d+)\.txt$")

    for fname in os.listdir(samples_folder):
        # only collect ddim sample-count files here.
        if not fname.endswith(".txt"):
            continue
        match = ddim_pattern.search(fname)
        if match is None:
            continue
        ddim_steps = int(match.group(1))
        counts = parse_counts(os.path.join(samples_folder, fname))
        if not counts:
            continue
        total = sum(counts.get(k, 0) for k in ("true_mode", "interpolated", "invalid"))
        if total <= 0:
            continue
        interpolated = counts.get("interpolated", 0)
        rate = interpolated / total * 100.0
        step_to_rates.setdefault(ddim_steps, []).append(rate)

    if not step_to_rates:
        raise ValueError(f"No DDIM hallucination files found under {run_folder}")

    expected_steps = expected_ddim_grid(cfg)
    missing_steps = [step for step in expected_steps if step not in step_to_rates]
    if missing_steps:
        raise ValueError(
            "DDIM interpolation-rate sweep requires the full DDIM grid. "
            f"Missing step files for K={missing_steps} under {samples_folder}."
        )

    steps = [int(step) for step in expected_steps]
    rates = [float(np.mean(step_to_rates[int(step)])) for step in steps]
    plot_min_ddim_steps = int(getattr(cfg, "plot_min_ddim_steps", 200))
    plot_pairs = [
        (step, rate)
        for step, rate in zip(steps, rates)
        if int(step) >= int(plot_min_ddim_steps)
    ]
    if not plot_pairs:
        raise ValueError(
            f"No DDIM steps remain for plotting after plot_min_ddim_steps={plot_min_ddim_steps}."
        )
    plot_steps = [int(step) for step, _ in plot_pairs]
    plot_rates = [float(rate) for _, rate in plot_pairs]

    ddpm_rate = None
    ddpm_rates = []
    ddpm_pattern = re.compile(r"^ddpm_hallucination(?:_\d+)?\.txt$")
    for fname in os.listdir(samples_folder):
        if ddpm_pattern.match(fname) is None:
            continue
        counts = parse_counts(os.path.join(samples_folder, fname))
        total = sum(counts.get(k, 0) for k in ("true_mode", "interpolated", "invalid"))
        if total <= 0:
            continue
        ddpm_rates.append(counts.get("interpolated", 0) / total * 100.0)
    if ddpm_rates:
        ddpm_rate = float(np.mean(ddpm_rates))

    plt.figure(figsize=(7, 4))

    # bigger line/marker
    plt.plot(
        plot_steps,
        plot_rates,
        marker="o",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="DDIM",
        color="tab:blue",
    )

    if ddpm_rate is not None:
        plt.axhline(
            ddpm_rate,
            linestyle="--",
            linewidth=LINEWIDTH,
            label="DDPM",
            color="tab:red"
        )

    # bigger labels
    plt.xlabel("DDIM timesteps", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Interpolated samples (%)", fontsize=LABEL_FONTSIZE)

    # set the vertical range from the actual rates so the ddpm baseline stays
    # visually close to the bottom without pinning the axis to 0.
    all_rates = list(plot_rates)
    if ddpm_rate is not None:
        all_rates.append(float(ddpm_rate))
    y_min = float(min(all_rates))
    y_max = float(max(all_rates))
    y_span = max(y_max - y_min, 1e-6)
    lower_pad = max(0.06 * y_span, 0.05)
    upper_pad = max(0.12 * y_span, 0.05)

    # bigger ticks
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.set_ylim(bottom=y_min - lower_pad, top=y_max + upper_pad)

    # remove the 0 and 6 labels so the y axis only shows the interior rate scale
    # used by the current gaussian25 hallucination figure.
    y_ticks = [
        tick
        for tick in ax.get_yticks()
        if not np.isclose(float(tick), 0.0) and not np.isclose(float(tick), 6.0)
    ]
    if y_ticks:
        ax.set_yticks(y_ticks)

    plt.grid(alpha=0.3)

    # bigger legend
    if ddpm_rate is not None:
        plt.legend(loc="best", fontsize=LEGEND_FONTSIZE, frameon=True)

    plt.tight_layout()

    save_folder = os.path.join(run_folder, "hallucinations")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "ddim_interpolation_rate_sweep.pdf")
    plt.savefig(save_path)
    plt.close()

    grid_min = int(min(expected_steps))
    grid_max = int(max(expected_steps))
    stem = f"ddim_step_hallucination_rates_{grid_min}_to_{grid_max}"
    rows = []
    for step in expected_steps:
        step_rates = np.asarray(step_to_rates.get(int(step), []), dtype=np.float64)
        rows.append(
            {
                "ddim_steps": int(step),
                "num_files": int(step_rates.size),
                "interpolation_rate_mean": float(np.mean(step_rates)) if step_rates.size else float("nan"),
                "interpolation_rate_std": float(np.std(step_rates)) if step_rates.size else float("nan"),
            }
        )
    csv_path = os.path.join(save_folder, f"{stem}.csv")
    json_path = os.path.join(save_folder, f"{stem}.json")
    md_path = os.path.join(save_folder, f"{stem}.md")
    write_csv(
        csv_path,
        ["ddim_steps", "num_files", "interpolation_rate_mean", "interpolation_rate_std"],
        rows,
    )
    payload = {
        "run_name": str(run_name),
        "results_dir": str(results_dir),
        "expected_ddim_steps_grid": [int(v) for v in expected_steps],
        "plot_min_ddim_steps": int(plot_min_ddim_steps),
        "plotted_ddim_steps": [int(v) for v in plot_steps],
        "ddim_rows": rows,
        "ddpm_rate_mean": ddpm_rate,
        "figure_pdf": save_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md_lines = [
        "| DDIM Steps | Hallucination % | Std % |",
        "|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {int(row['ddim_steps'])} | {float(row['interpolation_rate_mean']):.4f} | {float(row['interpolation_rate_std']):.4f} |"
        )
    if ddpm_rate is not None:
        md_lines.extend(
            [
                "",
                f"DDPM baseline: {float(ddpm_rate):.4f}%",
            ]
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def cfg_list(value, default):
    if value in (None, "", "null"):
        return list(default)
    if isinstance(value, str):
        return [int(v) for v in re.split(r"[,\s]+", value.strip("[] ")) if v]
    return [int(v) for v in value]


def nearest_value_at_kappa(payload: dict, key: str, kappa_target: float) -> tuple[float, float]:
    kappa = np.asarray(payload["kappa"], dtype=np.float64)
    values = np.asarray(payload[key], dtype=np.float64)
    idx = int(np.nanargmin(np.abs(kappa - float(kappa_target))))
    return float(kappa[idx]), float(values[idx])


def load_high_dim_curve(root: str, dim: int, stem: str) -> dict:
    path = os.path.join(root, f"dim_{dim}", "two_mode_assumption", f"{stem}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_high_dim_convergence(root: str, dim: int, stem: str) -> dict:
    path = os.path.join(root, f"dim_{dim}", "exp_convergence", f"{stem}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_high_dim_assumption_summary(cfg: DictConfig):
    results_dir = getattr(cfg, "results_dir", "eval_results")
    artifact_root = getattr(cfg, "high_dim_artifact_root", "dimensional_tau2_varpi")
    dims = cfg_list(getattr(cfg, "high_dim_dims", [2, 4, 8, 32, 64]), [2, 4, 8, 32, 64])
    kappa_target = float(getattr(cfg, "high_dim_kappa_target", 7.0))

    root = os.path.join(results_dir, artifact_root)
    rows = []
    curves = {}
    for dim in dims:
        curves[dim] = {}
        for sampler, solver_tag in (("ddim", "ddim_quad_50"), ("ddpm", "ddpm_1000")):
            tau_payload = load_high_dim_curve(root, dim, f"exp_a_tau_vs_kappa_{solver_tag}")
            u_payload = load_high_dim_curve(root, dim, f"exp_a_u_vs_kappa_{solver_tag}")
            curves[dim][sampler] = {"tau": tau_payload, "u": u_payload}
            actual_kappa, tau_1 = nearest_value_at_kappa(tau_payload, "tau_1_mean", kappa_target)
            _, tau_2 = nearest_value_at_kappa(tau_payload, "tau_2_mean", kappa_target)
            _, u_tau_1 = nearest_value_at_kappa(u_payload, "u_tau_1_mean", kappa_target)
            _, u_tau_2 = nearest_value_at_kappa(u_payload, "u_tau_2_mean", kappa_target)
            rows.append(
                {
                    "dim": int(dim),
                    "sampler": sampler,
                    "solver_tag": solver_tag,
                    "requested_kappa": float(kappa_target),
                    "nearest_kappa": float(actual_kappa),
                    "tau_1_mean": float(tau_1),
                    "tau_2_mean": float(tau_2),
                    "tau_gap_tau2_minus_tau1": float(tau_2 - tau_1),
                    "u_tau_1_mean": float(u_tau_1),
                    "u_tau_2_mean": float(u_tau_2),
                    "u_gap_tau2_minus_tau1": float(u_tau_2 - u_tau_1),
                }
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    panel_specs = [
        (0, 0, "ddim", "tau", r"DDIM $\tau$"),
        (0, 1, "ddpm", "tau", r"DDPM $\tau$"),
        (1, 0, "ddim", "u", r"DDIM $u(\tau)$"),
        (1, 1, "ddpm", "u", r"DDPM $u(\tau)$"),
    ]
    for row_idx, col_idx, sampler, curve_kind, title in panel_specs:
        ax = axes[row_idx][col_idx]
        for dim in dims:
            payload = curves[dim][sampler][curve_kind]
            kappa = np.asarray(payload["kappa"], dtype=np.float64)
            if curve_kind == "tau":
                y1 = np.asarray(payload["tau_1_mean"], dtype=np.float64)
                y2 = np.asarray(payload["tau_2_mean"], dtype=np.float64)
                ylabel = r"Mean reverse step"
            else:
                y1 = np.asarray(payload["u_tau_1_mean"], dtype=np.float64)
                y2 = np.asarray(payload["u_tau_2_mean"], dtype=np.float64)
                ylabel = r"Mean $u$"
            ax.plot(kappa, y1, linewidth=1.8, label=fr"$d={dim}$ $\tau_1$")
            ax.plot(kappa, y2, linewidth=1.8, linestyle="--", label=fr"$d={dim}$ $\tau_2$")
        ax.axvline(kappa_target, color="black", linewidth=1.0, alpha=0.35)
        ax.set_title(title, fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel(r"$\kappa$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    os.makedirs(root, exist_ok=True)
    pdf_path = os.path.join(root, "high_dim_assumption_cumulative.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    csv_path = os.path.join(root, "high_dim_assumption_cumulative.csv")
    json_path = os.path.join(root, "high_dim_assumption_cumulative.json")
    md_path = os.path.join(root, "high_dim_assumption_cumulative.md")
    fieldnames = [
        "dim",
        "sampler",
        "solver_tag",
        "requested_kappa",
        "nearest_kappa",
        "tau_1_mean",
        "tau_2_mean",
        "tau_gap_tau2_minus_tau1",
        "u_tau_1_mean",
        "u_tau_2_mean",
        "u_gap_tau2_minus_tau1",
    ]
    write_csv(csv_path, fieldnames, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results_dir": str(results_dir),
                "artifact_root": str(artifact_root),
                "dims": [int(v) for v in dims],
                "kappa_target": float(kappa_target),
                "figure_pdf": pdf_path,
                "rows": rows,
            },
            f,
            indent=2,
        )
    md_lines = [
        "| Dim | Sampler | Kappa | Tau1 | Tau2 | Tau2 - Tau1 | u(Tau1) | u(Tau2) | u gap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {dim} | {sampler} | {nearest_kappa:.4f} | {tau_1_mean:.4f} | {tau_2_mean:.4f} | "
            "{tau_gap_tau2_minus_tau1:.4f} | {u_tau_1_mean:.6f} | {u_tau_2_mean:.6f} | "
            "{u_gap_tau2_minus_tau1:.6f} |".format(**row)
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def plot_high_dim_convergence_summary(cfg: DictConfig):
    results_dir = getattr(cfg, "results_dir", "eval_results")
    artifact_root = getattr(cfg, "high_dim_artifact_root", "dimensional_tau2_varpi")
    dims = cfg_list(getattr(cfg, "high_dim_dims", [2, 4, 8, 32, 64]), [2, 4, 8, 32, 64])
    log_y = str(getattr(cfg, "high_dim_convergence_log_y", "true")).strip().lower() not in {"0", "false", "no"}

    root = os.path.join(results_dir, artifact_root)
    curves = {}
    rows = []
    for dim in dims:
        curves[dim] = {}
        for sampler, solver_tag in (("ddim", "ddim_quad_50"), ("ddpm", "ddpm_1000")):
            payload = load_high_dim_convergence(root, dim, f"exp_b_convergence_curve_{solver_tag}")
            curves[dim][sampler] = payload
            y = np.asarray(payload["mean_distance_to_line"], dtype=np.float64)
            progress = np.asarray(payload.get("reverse_progress", np.arange(y.size)), dtype=np.float64)
            finite = np.isfinite(y)
            start = float(y[finite][0]) if finite.any() else float("nan")
            end = float(y[finite][-1]) if finite.any() else float("nan")
            rows.append(
                {
                    "dim": int(dim),
                    "sampler": sampler,
                    "solver_tag": solver_tag,
                    "num_points": int(y.size),
                    "start_distance": start,
                    "end_distance": end,
                    "min_distance": float(np.nanmin(y)) if y.size else float("nan"),
                    "max_distance": float(np.nanmax(y)) if y.size else float("nan"),
                    "end_over_start": float(end / start) if np.isfinite(start) and abs(start) > 1e-12 else float("nan"),
                    "auc_progress": float(np.trapz(y, x=progress)) if y.size > 1 else 0.0,
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, sampler, title in (
        (axes[0], "ddim", r"DDIM convergence"),
        (axes[1], "ddpm", r"DDPM convergence"),
    ):
        for dim in dims:
            payload = curves[dim][sampler]
            y = np.asarray(payload["mean_distance_to_line"], dtype=np.float64)
            x = np.asarray(payload.get("reverse_progress", np.arange(y.size)), dtype=np.float64)
            y_plot = np.clip(y, 1e-12, None) if log_y else y
            ax.plot(x, y_plot, linewidth=1.8, label=fr"$d={dim}$")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel("Reverse progress", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(r"Mean distance to $L_t$", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    os.makedirs(root, exist_ok=True)
    pdf_path = os.path.join(root, "high_dim_convergence_cumulative.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    csv_path = os.path.join(root, "high_dim_convergence_cumulative.csv")
    json_path = os.path.join(root, "high_dim_convergence_cumulative.json")
    md_path = os.path.join(root, "high_dim_convergence_cumulative.md")
    fieldnames = [
        "dim",
        "sampler",
        "solver_tag",
        "num_points",
        "start_distance",
        "end_distance",
        "min_distance",
        "max_distance",
        "end_over_start",
        "auc_progress",
    ]
    write_csv(csv_path, fieldnames, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results_dir": str(results_dir),
                "artifact_root": str(artifact_root),
                "dims": [int(v) for v in dims],
                "log_y": bool(log_y),
                "figure_pdf": pdf_path,
                "rows": rows,
            },
            f,
            indent=2,
        )
    md_lines = [
        "| Dim | Sampler | Start | End | End / Start | Min | Max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {dim} | {sampler} | {start_distance:.6g} | {end_distance:.6g} | "
            "{end_over_start:.6g} | {min_distance:.6g} | {max_distance:.6g} |".format(**row)
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="vis_config",
)
def run_visualization(cfg: DictConfig):
    # dispatch the visualization entrypoint.
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "ddim_interpolation_rate_sweep":
        plot_ddim_interpolation_rate_sweep(cfg)
    elif cfg.mode == "high_dim_assumption_summary":
        plot_high_dim_assumption_summary(cfg)
    elif cfg.mode == "high_dim_convergence_summary":
        plot_high_dim_convergence_summary(cfg)
    else:
        raise ValueError(f"{cfg.mode} is not recognized")


if __name__ == "__main__":
    run_visualization()
