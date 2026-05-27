import json
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from common.gaussian_mixture_2d import build_mixture_spec_from_dataset_cfg
from common.gaussian_exact_score import oracle_eps_and_score_at_t


def plain_args(args):
    # as plain args.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    if OmegaConf.is_config(args):
        return OmegaConf.to_container(args, resolve=True)
    return args


def score_training_method(args) -> str:
    # score training method.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    plain = plain_args(args)
    return str(plain.get("score_training_method", "dsm")).strip().lower()


def model_output_parameterization(args) -> str:
    # model output parameterization.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    plain = plain_args(args)
    default = "score" if score_training_method(plain) in {"esm", "ism"} else "eps"
    return str(plain.get("model_output_parameterization", default)).strip().lower()


def validate_score_training_args(args) -> None:
    # validate score training args.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    method = score_training_method(args)
    output = model_output_parameterization(args)
    if method not in {"dsm", "esm", "ism"}:
        raise ValueError(f"Unsupported score_training_method={method!r}")
    if output not in {"eps", "score"}:
        raise ValueError(f"Unsupported model_output_parameterization={output!r}")
    if method == "dsm" and output != "eps":
        raise ValueError("DSM training requires model_output_parameterization='eps'.")
    if method in {"esm", "ism"} and output != "score":
        raise ValueError(f"{method.upper()} training requires model_output_parameterization='score'.")


def dataset_cfg_json(args) -> str:
    # dataset cfg json.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    plain = plain_args(args)
    dataset_cfg = plain.get("dataset", {})
    return json.dumps(dataset_cfg, sort_keys=True)


@lru_cache(maxsize=32)
def cached_gaussian_mixture_ctx(dataset_cfg_json: str) -> dict:
    # cached gaussian mixture ctx.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    dataset_cfg = json.loads(dataset_cfg_json)
    if str(dataset_cfg.get("kind", "")).lower() != "gaussian_mixture_2d":
        raise NotImplementedError(
            "Explicit and implicit score matching are currently implemented for "
            "dataset.kind='gaussian_mixture_2d' only."
        )
    spec = build_mixture_spec_from_dataset_cfg(dataset_cfg)
    return {
        "modes": np.asarray(spec["normalized_means"], dtype=np.float64),
        "mode_sigmas": np.asarray(spec["normalized_sigmas"], dtype=np.float64),
        "mode_weights": np.asarray(spec["mode_weights"], dtype=np.float64),
        "std_dev": float(spec["global_sigma_normalized"]),
        "num_dims": int(np.asarray(spec["normalized_means"]).shape[1]),
    }


def gaussian_mixture_training_context(args) -> dict:
    # gaussian mixture training context.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    return cached_gaussian_mixture_ctx(dataset_cfg_json(args))


def interp_alpha_bar(diffusion, t, *, device, dtype):
    # interp alpha bar.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device)
    t = t.to(device=device, dtype=torch.float64).reshape(-1)
    buf = diffusion.alphas_prod.to(device=device, dtype=torch.float64)
    t_clamped = torch.clamp(t, 0.0, float(buf.shape[0] - 1))
    t0 = torch.floor(t_clamped).long()
    t1 = torch.ceil(t_clamped).long()
    w = (t_clamped - t0.to(torch.float64)).to(torch.float64)
    a0 = buf.index_select(0, t0)
    a1 = buf.index_select(0, t1)
    a_bar = (1.0 - w) * a0 + w * a1
    return a_bar.to(dtype=dtype)


class ScoreModelAsEpsModel(nn.Module):
    def __init__(self, score_model: nn.Module, diffusion):
        # init.
        # this supports the score-training wrappers that connect model outputs to the paper's objectives.
        super().__init__()
        self.score_model = score_model
        self.diffusion = diffusion

    def forward(self, x_t: torch.Tensor, t):
        # forward.
        # this supports the score-training wrappers that connect model outputs to the paper's objectives.
        score = self.score_model(x_t, t)
        flat_t = t if torch.is_tensor(t) else torch.as_tensor(t, device=x_t.device)
        if torch.is_tensor(flat_t) and flat_t.ndim == 0:
            flat_t = flat_t.repeat(x_t.shape[0])
        elif torch.is_tensor(flat_t):
            flat_t = flat_t.reshape(-1)
            if flat_t.shape[0] == 1 and x_t.shape[0] != 1:
                flat_t = flat_t.repeat(x_t.shape[0])
        a_bar = interp_alpha_bar(self.diffusion, flat_t, device=x_t.device, dtype=score.dtype)
        view_shape = (a_bar.shape[0],) + (1,) * (score.ndim - 1)
        sqrt_one_minus = torch.sqrt((1.0 - a_bar).clamp_min(1e-18)).view(view_shape)
        return -sqrt_one_minus * score


def maybe_wrap_model_for_eps(model: nn.Module, diffusion, args) -> nn.Module:
    # maybe wrap model.
    # this supports eps for the score-training wrappers that connect the model parameterization to the paper's objectives.
    if model_output_parameterization(args) == "score":
        wrapped = ScoreModelAsEpsModel(model, diffusion)
        wrapped.to(next(model.parameters()).device)
        wrapped.eval()
        return wrapped
    return model


def batch_exact_divergence(score_pred: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    # batch exact divergence.
    # this supports the score-training wrappers that connect the model parameterization to the paper's objectives.
    if score_pred.ndim != 2 or x_t.ndim != 2:
        raise ValueError(
            f"ISM exact divergence expects 2D tensors shaped (B, D); got "
            f"{tuple(score_pred.shape)} and {tuple(x_t.shape)}"
        )
    div = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=score_pred.dtype)
    dim = int(score_pred.shape[1])
    for axis in range(dim):
        # Keep the graph alive for the outer loss.backward() call after we
        # differentiate each score component to form the exact divergence.
        grad_axis = torch.autograd.grad(
            outputs=score_pred[:, axis].sum(),
            inputs=x_t,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][:, axis]
        div = div + grad_axis
    return div


def gaussian_score_matching_losses(model, x_0, t, args, diffusion, noise=None):
    # compute the Gaussian score-training losses used by the paper's learned-score experiments.
    method = score_training_method(args)
    if method == "dsm":
        raise ValueError("gaussian_score_matching_losses should not be called for DSM.")

    if noise is None:
        noise = torch.randn_like(x_0)

    ctx = gaussian_mixture_training_context(args)
    x_t = diffusion.forward_process(x_0, t, noise=noise)

    modes = torch.as_tensor(ctx["modes"], device=x_t.device, dtype=torch.float64)
    mode_sigmas = torch.as_tensor(ctx["mode_sigmas"], device=x_t.device, dtype=torch.float64)
    mode_weights = torch.as_tensor(ctx["mode_weights"], device=x_t.device, dtype=torch.float64)
    std_dev = float(ctx["std_dev"])

    if method == "esm":
        with torch.no_grad():
            _, score_true = oracle_eps_and_score_at_t(
                diffusion=diffusion,
                x_t=x_t,
                gaussian_modes=modes,
                std_dev=std_dev,
                t_eff=t,
                mode_sigmas=mode_sigmas,
                mode_weights=mode_weights,
            )
        score_pred = model(x_t, t)
        score_true = score_true.to(dtype=score_pred.dtype)
        return torch.mean((score_pred - score_true) ** 2, dim=tuple(range(1, score_pred.ndim)))

    if method == "ism":
        x_t = x_t.detach().requires_grad_(True)
        score_pred = model(x_t, t)
        divergence = batch_exact_divergence(score_pred, x_t)
        norm_sq = torch.sum(score_pred * score_pred, dim=tuple(range(1, score_pred.ndim)))
        return divergence + 0.5 * norm_sq

    raise ValueError(f"Unsupported score_training_method={method!r}")
