import argparse
import copy
import json
import math
import os
from typing import List

import torch
import wandb
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from common.config_utils import cfg_get
from common.diffusion import Diffusion
from common.score_training import (
    gaussian_score_matching_losses,
    maybe_wrap_model_for_eps,
    score_training_method,
    validate_score_training_args,
)
from common.unet_new import update_ema
from common.utils import generate_image_organizer, sample, sample_gaussian, visualize_gaussian_samples


def flat_mean(x, start_dim=1):
    # flat mean.
    # this supports the shared training loop used by the paper's learned-score models.
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.mean(x, dim=reduce_dim)


def is_gaussian_dataset(args) -> bool:
    # is gaussian dataset.
    # this supports the shared training loop used by the paper's learned-score models.
    gaussian_datasets = {
        "gaussian25",
        "gaussian5d",
        "gaussian_mixture_2d",
    }
    dataset_name = str(cfg_get(args, "dataset.name", "")).lower()
    dataset_kind = str(cfg_get(args, "dataset.kind", "")).lower()
    return dataset_name in gaussian_datasets or dataset_kind == "gaussian_mixture_2d"


def wandb_enabled(args) -> bool:
    # use wandb.
    # this supports the shared training loop used by the paper's learned-score models.
    return bool(cfg_get(args, "use_wandb", cfg_get(args, "log.use_wandb", False)))


def run_path_for_args(args) -> str:
    # run path for the shared training loop used by the paper's learned-score models.
    save_folder = str(cfg_get(args, "dataset.save_folder", "./results"))
    run_name = str(cfg_get(args, "log.run_name", "") or "")
    return os.path.join(save_folder, run_name) if run_name else save_folder


def train_value(args, nested_key: str, flat_key: str | None = None, default=None):
    # train value.
    # this supports the shared training loop used by the paper's learned-score models.
    val = cfg_get(args, f"train.{nested_key}", None)
    if val is not None:
        return val
    if flat_key is not None:
        val = cfg_get(args, flat_key, None)
        if val is not None:
            return val
    return default


def optimizer_value(args, key: str, fallback_key: str | None = None, default=None):
    # optimizer value.
    # this supports the shared training loop used by the paper's learned-score models.
    val = cfg_get(args, f"optimizer.{key}", None)
    if val is not None:
        return val
    if fallback_key is not None:
        val = cfg_get(args, fallback_key, None)
        if val is not None:
            return val
    return default


def lr_for_step(
    step_idx: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
) -> float:
    # lr.
    # this supports step for the shared training loop used by the paper's learned-score models.
    if max_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step_idx < warmup_steps:
        return base_lr * float(step_idx + 1) / float(max(warmup_steps, 1))
    if max_steps <= warmup_steps:
        return min_lr
    progress = float(step_idx - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    if schedule in {"constant", "none"}:
        return base_lr
    if schedule in {"linear", "linear_decay"}:
        return base_lr + (min_lr - base_lr) * progress
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def checkpoint_sort_key(filename: str):
    # checkpoint sort key.
    # this supports the shared training loop used by the paper's learned-score models.
    for prefix in ("checkpoint_step_", "checkpoint_epoch_"):
        if filename.startswith(prefix) and filename.endswith(".pth"):
            try:
                value = int(filename[len(prefix) : -len(".pth")])
                return value
            except ValueError:
                return None
    return None


def find_latest_ckpt(ckpt_folder):
    # find latest ckpt used by the shared training loop used by the paper's learned-score models.
    if not os.path.isdir(ckpt_folder):
        return None

    latest_value = None
    latest_ckpt = None
    for filename in os.listdir(ckpt_folder):
        value = checkpoint_sort_key(filename)
        if value is None:
            continue
        if latest_value is None or value > latest_value:
            latest_value = value
            latest_ckpt = filename
    return latest_ckpt


def load_checkpoint(checkpoint_path, model, optimizer, device, ema_model=None):
    # load checkpoint needed by the shared training loop used by the paper's learned-score models.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class LambdaBinDiagnostics:
    def __init__(self, *, save_path: str, diffusion: Diffusion, num_bins: int = 20):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.save_path = save_path
        self.bin_ids = diffusion.lambda_bin_indices(num_bins=num_bins).detach().cpu()
        self.num_bins = int(num_bins)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.reset()

    def reset(self):
        # reset.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.counts = [0 for _ in range(self.num_bins)]
        self.weight_sums = [0.0 for _ in range(self.num_bins)]
        self.eps_err_sums = [0.0 for _ in range(self.num_bins)]
        self.eps_err_weighted_sums = [0.0 for _ in range(self.num_bins)]
        self.x0_err_sums = [0.0 for _ in range(self.num_bins)]
        self.x0_err_weighted_sums = [0.0 for _ in range(self.num_bins)]

    def update(self, *, t, weights, eps_err_norm, x0_err_norm):
        # update.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        t_cpu = t.detach().long().cpu().reshape(-1)
        w_cpu = weights.detach().float().cpu().reshape(-1)
        eps_cpu = eps_err_norm.detach().float().cpu().reshape(-1)
        x0_cpu = x0_err_norm.detach().float().cpu().reshape(-1)
        bin_assignments = self.bin_ids.index_select(0, t_cpu)

        for bin_id in range(self.num_bins):
            mask = bin_assignments == bin_id
            if not torch.any(mask):
                continue
            count = int(mask.sum().item())
            weights_bin = w_cpu[mask]
            eps_bin = eps_cpu[mask]
            x0_bin = x0_cpu[mask]
            self.counts[bin_id] += count
            self.weight_sums[bin_id] += float(weights_bin.sum().item())
            self.eps_err_sums[bin_id] += float(eps_bin.sum().item())
            self.eps_err_weighted_sums[bin_id] += float((weights_bin * eps_bin).sum().item())
            self.x0_err_sums[bin_id] += float(x0_bin.sum().item())
            self.x0_err_weighted_sums[bin_id] += float((weights_bin * x0_bin).sum().item())

    def overall_summary(self):
        # overall summary.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        total_count = sum(self.counts)
        total_weight = sum(self.weight_sums)
        eps_raw = sum(self.eps_err_sums) / float(max(total_count, 1))
        x0_raw = sum(self.x0_err_sums) / float(max(total_count, 1))
        eps_weighted = sum(self.eps_err_weighted_sums) / float(max(total_weight, 1e-12))
        x0_weighted = sum(self.x0_err_weighted_sums) / float(max(total_weight, 1e-12))
        return {
            "sample_count": int(total_count),
            "total_importance_mass": float(total_weight),
            "raw_eps_error_mean": float(eps_raw),
            "importance_corrected_eps_error_mean": float(eps_weighted),
            "raw_x0_error_mean": float(x0_raw),
            "importance_corrected_x0_error_mean": float(x0_weighted),
        }

    def flush(self, *, global_step: int, lr: float, loss_window_mean: float, batch_weight_stats: dict):
        # flush.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        bins = []
        for bin_id in range(self.num_bins):
            count = int(self.counts[bin_id])
            weight_sum = float(self.weight_sums[bin_id])
            bins.append(
                {
                    "bin_id": int(bin_id),
                    "sample_count": count,
                    "sampled_mean_weight": (weight_sum / float(count)) if count > 0 else None,
                    "total_importance_mass": weight_sum,
                    "raw_eps_error_mean": (self.eps_err_sums[bin_id] / float(count)) if count > 0 else None,
                    "importance_corrected_eps_error_mean": (
                        self.eps_err_weighted_sums[bin_id] / float(max(weight_sum, 1e-12))
                    )
                    if count > 0
                    else None,
                    "raw_x0_error_mean": (self.x0_err_sums[bin_id] / float(count)) if count > 0 else None,
                    "importance_corrected_x0_error_mean": (
                        self.x0_err_weighted_sums[bin_id] / float(max(weight_sum, 1e-12))
                    )
                    if count > 0
                    else None,
                }
            )

        record = {
            "global_step": int(global_step),
            "learning_rate": float(lr),
            "loss_window_mean": float(loss_window_mean),
            "batch_weight_stats": {
                "mean": float(batch_weight_stats["mean"]),
                "min": float(batch_weight_stats["min"]),
                "max": float(batch_weight_stats["max"]),
            },
            "overall": self.overall_summary(),
            "lambda_bins": bins,
        }
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.reset()
        return record


def save_periodic_samples(sample_model, diffusion, args, *, step: int, true_data_sample, run_path: str, device):
    # save periodic samples for the shared training loop used by the paper's learned-score models.
    is_gaussian = is_gaussian_dataset(args)
    if is_gaussian:
        samples = sample_gaussian(
            model=sample_model,
            diffusion=diffusion,
            args=args,
            num_samples=int(cfg_get(args, "train.sample_num_samples", cfg_get(args, "dataset.num_test_samples", 1000))),
            device=device,
            vis_trajectory=bool(cfg_get(args, "dataset.vis_trajectory", False)),
        )
        filename = f"samples_step_{step:07d}"
        visualize_gaussian_samples(
            samples=samples,
            save_folder=os.path.join(run_path, "train_samples"),
            filename=filename,
            title=f"{cfg_get(args, 'dataset.name', 'dataset')} - Step {step}",
            true_data=true_data_sample,
            xlim=(-4, 4) if "gaussian" in str(cfg_get(args, "dataset.name", "")).lower() else None,
            ylim=(-4, 4) if "gaussian" in str(cfg_get(args, "dataset.name", "")).lower() else None,
        )
        return None

    samples = sample(
        sample_model,
        diffusion,
        input_type="random_noise",
        num_samples=16,
        img_size=int(cfg_get(args, "dataset.img_size", 32)),
        diffusion_args=args,
        device=device,
    )
    filename = f"training_samples_step_{step:07d}"
    save_folder = os.path.join(run_path, "samples")
    os.makedirs(save_folder, exist_ok=True)
    return generate_image_organizer(samples, save_folder, filename, title=f"Training Samples - Step {step}")


def train(model, diffusion, dataloader, args, device):
    # train the model with the shared training loop and checkpointing logic.
    validate_score_training_args(args)
    max_steps = train_value(args, "max_steps", default=None)
    if max_steps is None:
        raise ValueError("train.max_steps is required for the current training pipeline.")
    return train_step_based(model, diffusion, dataloader, args, device)


def train_step_based(model, diffusion, dataloader, args, device):
    # run the modern step-based training loop used by the current paper workflows.
    active_training_method = score_training_method(args)
    print(f"Training objective: {active_training_method}")

    is_gaussian = is_gaussian_dataset(args)
    true_data_sample = None
    if is_gaussian:
        true_data_batch = next(iter(dataloader))
        true_data_sample = true_data_batch[: min(1000, true_data_batch.shape[0])]

    run_path = run_path_for_args(args)
    ckpt_dir = os.path.join(run_path, "checkpoints")
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    base_lr = float(optimizer_value(args, "lr", fallback_key="lr", default=1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(
            float(optimizer_value(args, "beta1", default=0.9)),
            float(optimizer_value(args, "beta2", default=0.999)),
        ),
        eps=float(optimizer_value(args, "eps", default=1e-8)),
        weight_decay=float(optimizer_value(args, "weight_decay", default=0.0)),
    )

    ema_decay = float(train_value(args, "ema_decay", default=0.0))
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    use_wandb = wandb_enabled(args)
    if use_wandb:
        wandb.init(
            project=str(cfg_get(args, "wandb_project", cfg_get(args, "log.project_name", "hallucination-in-diffusion-models"))),
            name=str(cfg_get(args, "wandb_run_name", cfg_get(args, "log.run_name", ""))),
            config=OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else args,
        )
        wandb.watch(model, log="all", log_freq=100)

    losses = []
    global_step = 0
    if cfg_get(args, "continue_training_path", None):
        ckpt_folder = ckpt_dir if str(cfg_get(args, "log.run_name", "") or "") != "" else os.path.join(
            str(cfg_get(args, "dataset.save_folder", "./results")),
            str(cfg_get(args, "continue_training_path")),
        )
        latest_ckpt = find_latest_ckpt(ckpt_folder)
        if latest_ckpt is not None:
            ckpt_path = os.path.join(ckpt_folder, latest_ckpt)
            checkpoint = load_checkpoint(ckpt_path, model, optimizer, device, ema_model=ema_model)
            losses = list(checkpoint.get("losses", []))
            global_step = int(checkpoint.get("global_step", checkpoint.get("epoch", 0)))
            diffusion = Diffusion.from_checkpoint_payload(checkpoint, args=args, device=device)
            print(f"Resumed training from global_step={global_step}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    max_steps = int(train_value(args, "max_steps", default=0))
    log_every_steps = int(train_value(args, "log_every_steps", flat_key=None, default=100))
    sample_every_steps = int(train_value(args, "sample_every_steps", flat_key="sample_interval", default=1000))
    checkpoint_every_steps = int(train_value(args, "checkpoint_every_steps", flat_key="checkpoint_interval", default=1000))
    grad_clip_norm = float(train_value(args, "grad_clip_norm", default=1.0))
    warmup_steps = int(train_value(args, "lr_warmup_steps", default=0))
    min_lr = float(train_value(args, "min_lr", default=base_lr))
    lr_schedule = str(train_value(args, "lr_schedule", default="cosine")).strip().lower()
    lambda_bins = int(train_value(args, "lambda_bins", default=20))

    diagnostics = LambdaBinDiagnostics(
        save_path=os.path.join(run_path, "training_diagnostics.jsonl"),
        diffusion=diffusion,
        num_bins=lambda_bins,
    )

    schedule_metadata_path = os.path.join(run_path, "schedule_metadata.json")
    schedule_metadata = diffusion.export_schedule_metadata()
    with open(schedule_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "beta_schedule": schedule_metadata["beta_schedule"],
                "schedule_params": schedule_metadata["schedule_params"],
                "high_snr_top_frac": schedule_metadata["high_snr_top_frac"],
                "betas": schedule_metadata["betas"].tolist(),
                "alphas": schedule_metadata["alphas"].tolist(),
                "alphas_prod": schedule_metadata["alphas_prod"].tolist(),
                "lambda_vals": schedule_metadata["lambda_vals"].tolist(),
                "q_probs": schedule_metadata["q_probs"].tolist(),
                "importance_weights": schedule_metadata["importance_weights"].tolist(),
            },
            f,
            indent=2,
        )

    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training")
    loss_window = []
    steps_per_epoch = max(len(dataloader), 1)
    last_batch_weight_stats = {"mean": 1.0, "min": 1.0, "max": 1.0}

    while global_step < max_steps:
        model.train()
        for batch in dataloader:
            if global_step >= max_steps:
                break

            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size = x.shape[0]

            t, importance_weights = diffusion.sample_timesteps(batch_size, device=device)
            losses_per_sample, details = train_losses(
                model,
                x,
                t,
                args,
                diffusion,
                importance_weights=importance_weights,
                return_details=True,
            )
            loss = losses_per_sample.mean()

            lr = lr_for_step(
                global_step,
                base_lr=base_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                schedule=lr_schedule,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            if ema_decay > 0.0:
                update_ema(list(ema_model.parameters()), list(model.parameters()), rate=ema_decay)
                for ema_buf, model_buf in zip(ema_model.buffers(), model.buffers()):
                    ema_buf.copy_(model_buf)
            else:
                ema_model.load_state_dict(model.state_dict())

            global_step += 1
            losses.append(float(loss.item()))
            loss_window.append(float(loss.item()))
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

            if details is not None and score_training_method(args) == "dsm":
                eps_err_norm = torch.linalg.norm((details["model_out"] - details["target"]).reshape(batch_size, -1), dim=1)
                x0_hat = diffusion.predict_x0_from_eps(details["x_t"], details["model_out"], t)
                x0_err_norm = torch.linalg.norm((x - x0_hat).reshape(batch_size, -1), dim=1)
                diagnostics.update(
                    t=t,
                    weights=importance_weights,
                    eps_err_norm=eps_err_norm,
                    x0_err_norm=x0_err_norm,
                )

            last_batch_weight_stats = {
                "mean": float(importance_weights.mean().item()),
                "min": float(importance_weights.min().item()),
                "max": float(importance_weights.max().item()),
            }

            if use_wandb and global_step % 10 == 0:
                wandb.log(
                    {
                        "train/step_loss": float(loss.item()),
                        "train/global_step": int(global_step),
                        "train/learning_rate": float(lr),
                        "train/batch_weight_mean": last_batch_weight_stats["mean"],
                        "train/batch_weight_min": last_batch_weight_stats["min"],
                        "train/batch_weight_max": last_batch_weight_stats["max"],
                    }
                )

            if global_step % log_every_steps == 0 or global_step == max_steps:
                loss_window_mean = float(sum(loss_window) / max(len(loss_window), 1))
                record = diagnostics.flush(
                    global_step=global_step,
                    lr=lr,
                    loss_window_mean=loss_window_mean,
                    batch_weight_stats=last_batch_weight_stats,
                )
                loss_window.clear()
                overall = record["overall"]
                print(
                    "step={step} loss={loss:.6f} lr={lr:.6f} weight_mean={wmean:.6f} "
                    "weight_min={wmin:.6f} weight_max={wmax:.6f} "
                    "eps_err={eps:.6f} x0_err={x0:.6f}".format(
                        step=global_step,
                        loss=loss_window_mean,
                        lr=lr,
                        wmean=last_batch_weight_stats["mean"],
                        wmin=last_batch_weight_stats["min"],
                        wmax=last_batch_weight_stats["max"],
                        eps=overall["importance_corrected_eps_error_mean"],
                        x0=overall["importance_corrected_x0_error_mean"],
                    )
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/global_step": int(global_step),
                            "train/loss_window_mean": loss_window_mean,
                            "train/learning_rate": float(lr),
                            "train/batch_weight_mean": last_batch_weight_stats["mean"],
                            "train/batch_weight_min": last_batch_weight_stats["min"],
                            "train/batch_weight_max": last_batch_weight_stats["max"],
                            "train/importance_corrected_eps_error_mean": overall["importance_corrected_eps_error_mean"],
                            "train/importance_corrected_x0_error_mean": overall["importance_corrected_x0_error_mean"],
                        }
                    )

            if sample_every_steps > 0 and (global_step % sample_every_steps == 0 or global_step == max_steps):
                sample_model = maybe_wrap_model_for_eps(ema_model, diffusion, args)
                save_periodic_samples(
                    sample_model,
                    diffusion,
                    args,
                    step=global_step,
                    true_data_sample=true_data_sample,
                    run_path=run_path,
                    device=device,
                )

            if checkpoint_every_steps > 0 and (global_step % checkpoint_every_steps == 0 or global_step == max_steps):
                checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pth")
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    global_step,
                    losses,
                    diffusion,
                    args,
                    ema_model=ema_model,
                    epoch=global_step // max(steps_per_epoch, 1),
                )
                print(f"Checkpoint saved: {checkpoint_path}")

    progress_bar.close()
    model_path = os.path.join(ckpt_dir, "final_model.pth")
    save_checkpoint(
        model_path,
        model,
        optimizer,
        global_step,
        losses,
        diffusion,
        args,
        ema_model=ema_model,
        epoch=global_step // max(steps_per_epoch, 1),
    )
    print(f"Model saved at: {model_path}\n")
    if use_wandb:
        wandb.finish()
    return losses


def train_losses(model, x_0, t, args, diffusion, noise=None, importance_weights=None, return_details=False):
    # train losses.
    # this supports the shared training loop used by the paper's learned-score models.
    if noise is None:
        noise = torch.randn_like(x_0)

    active_method = score_training_method(args)
    x_t = diffusion.forward_process(x_0, t, noise=noise)
    details = None

    if active_method != "dsm":
        losses = gaussian_score_matching_losses(
            model=model,
            x_0=x_0,
            t=t,
            args=args,
            diffusion=diffusion,
            noise=noise,
        )
    else:
        if str(cfg_get(args, "loss_type", "mse")) == "kl":
            losses = cfg_get(args, "_loss_term_bpd")(model, x_0=x_0, x_t=x_t, t=t, clip_denoised=False, return_pred=False)
        elif str(cfg_get(args, "loss_type", "mse")) == "mse":
            model_mean_type = str(cfg_get(args, "model_mean_type", "eps"))
            if model_mean_type == "mean":
                target = cfg_get(args, "q_posterior_mean_var")(x_0=x_0, x_t=x_t, t=t)[0]
            elif model_mean_type == "x_0":
                target = x_0
            elif model_mean_type == "eps":
                target = noise
            else:
                raise NotImplementedError(model_mean_type)

            model_out = model(x_t, t)
            losses = flat_mean((target - model_out).pow(2))
            if return_details:
                details = {"x_t": x_t, "target": target, "model_out": model_out}
        else:
            raise NotImplementedError(cfg_get(args, "loss_type"))

    if importance_weights is not None:
        losses = losses * importance_weights.reshape(-1)

    if return_details:
        return losses, details
    return losses


def parse_class_list(class_string: str) -> List[int]:
    # parse class list for the shared training loop used by the paper's learned-score models.
    try:
        return [int(c.strip()) for c in class_string.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid class list: {e}")


def load_config(config_path):
    # load config needed by the shared training loop used by the paper's learned-score models.
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(checkpoint_path, model, optimizer, global_step, losses, diffusion, args, ema_model=None, epoch=None):
    # save checkpoint for the shared training loop used by the paper's learned-score models.
    if OmegaConf.is_config(args):
        args_to_save = OmegaConf.to_container(args, resolve=True, enum_to_str=True)
    else:
        args_to_save = args

    schedule_metadata = diffusion.export_schedule_metadata()
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch if epoch is not None else global_step,
        "global_step": int(global_step),
        "losses": losses,
        "diffusion": diffusion,
        "args": args_to_save,
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "beta_schedule": schedule_metadata["beta_schedule"],
        "schedule_params": schedule_metadata["schedule_params"],
        "betas": schedule_metadata["betas"],
        "alphas": schedule_metadata["alphas"],
        "alphas_prod": schedule_metadata["alphas_prod"],
        "lambda_vals": schedule_metadata["lambda_vals"],
        "q_probs": schedule_metadata["q_probs"],
        "importance_weights": schedule_metadata["importance_weights"],
        "high_snr_top_frac": schedule_metadata["high_snr_top_frac"],
    }
    if ema_model is not None:
        payload["ema_state_dict"] = ema_model.state_dict()
    torch.save(payload, checkpoint_path)
