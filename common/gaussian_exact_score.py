import math

import numpy as np
import torch


def interp_buffer_at_time(
    buf: torch.Tensor,
    t_eff,
    device,
    dtype,
) -> torch.Tensor:
    # interpolate one diffusion buffer at a possibly non-integer raw time.
    # the exact gaussian score helpers use this when a solver queries times between stored buffer entries.
    t = float(t_eff)
    n = int(buf.shape[0])
    if n <= 0:
        raise ValueError("Empty diffusion buffer.")
    if t <= 0.0:
        return buf[0].to(device=device, dtype=dtype)
    if t >= float(n - 1):
        return buf[n - 1].to(device=device, dtype=dtype)

    t0 = int(math.floor(t))
    t1 = int(math.ceil(t))
    if t1 == t0:
        return buf[t0].to(device=device, dtype=dtype)
    w = t - float(t0)
    v0 = buf[t0].to(device=device, dtype=dtype)
    v1 = buf[t1].to(device=device, dtype=dtype)
    return (1.0 - w) * v0 + w * v1


def prepare_mode_sigmas(gaussian_modes: torch.Tensor, std_dev: float | None, mode_sigmas=None) -> torch.Tensor:
    # prepare the per-mode standard deviations for an exact gaussian-mixture calculation.
    # the exact-score helpers allow either one shared $\\sigma$ or an explicit per-mode array.
    m = int(gaussian_modes.shape[0])
    if mode_sigmas is None:
        if std_dev is None:
            raise ValueError("Either std_dev or mode_sigmas must be provided")
        return torch.full((m,), float(std_dev), dtype=torch.float64, device=gaussian_modes.device)
    sig = torch.as_tensor(mode_sigmas, dtype=torch.float64, device=gaussian_modes.device).reshape(-1)
    if sig.shape[0] != m:
        raise ValueError(f"mode_sigmas must have length {m}, got {sig.shape[0]}")
    if torch.any(sig <= 0.0):
        raise ValueError("mode_sigmas must be positive")
    return sig


def prepare_mode_weights(gaussian_modes: torch.Tensor, mode_weights=None) -> torch.Tensor:
    # prepare the normalized mixture weights for the exact gaussian-mixture formulas.
    # by default the synthetic studies use equal weights across modes.
    m = int(gaussian_modes.shape[0])
    if mode_weights is None:
        return torch.full((m,), 1.0 / float(max(m, 1)), dtype=torch.float64, device=gaussian_modes.device)
    weights = torch.as_tensor(mode_weights, dtype=torch.float64, device=gaussian_modes.device).reshape(-1)
    if weights.shape[0] != m:
        raise ValueError(f"mode_weights must have length {m}, got {weights.shape[0]}")
    if torch.any(weights < 0.0):
        raise ValueError("mode_weights must be non-negative")
    total = torch.sum(weights)
    if float(total.item()) <= 0.0:
        raise ValueError("mode_weights must sum to a positive value")
    return weights / total


def mixture_log_prob_torch(
    x: torch.Tensor,
    gaussian_modes: torch.Tensor,
    *,
    std_dev: float | None = None,
    mode_sigmas=None,
    mode_weights=None,
) -> torch.Tensor:
    # evaluate the exact gaussian-mixture log density at a batch of points.
    # this is the shared exact-density primitive used by the explicit-score studies and diagnostics.
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (N, D), got {tuple(x.shape)}")
    if gaussian_modes.ndim != 2:
        raise ValueError(f"Expected gaussian_modes shape (M, D), got {tuple(gaussian_modes.shape)}")
    if x.shape[1] != gaussian_modes.shape[1]:
        raise ValueError("x and gaussian_modes dimensions do not match")

    modes64 = torch.as_tensor(gaussian_modes, device=x.device, dtype=torch.float64)
    x64 = x.to(dtype=torch.float64)
    sig64 = prepare_mode_sigmas(modes64, std_dev=std_dev, mode_sigmas=mode_sigmas)
    w64 = prepare_mode_weights(modes64, mode_weights=mode_weights)

    diff = x64[:, None, :] - modes64[None, :, :]
    dist2 = torch.sum(diff * diff, dim=-1)
    d = int(x.shape[1])
    sigma2 = sig64 * sig64
    log_comp = (
        torch.log(w64.clamp_min(1e-300))[None, :]
        - 0.5 * d * torch.log(2.0 * math.pi * sigma2)[None, :]
        - 0.5 * dist2 / sigma2[None, :]
    )
    return torch.logsumexp(log_comp, dim=1)


def true_score_noised_mixture(
    x_t: torch.Tensor,
    modes_t: torch.Tensor,
    sigma2_eff: torch.Tensor,
    mode_weights: torch.Tensor,
) -> torch.Tensor:
    # evaluate the exact score of the time-$t$ noised gaussian mixture.
    # the score is the posterior-weighted mean displacement divided by the effective variance.
    diff = x_t[:, None, :] - modes_t[None, :, :]
    dist2 = torch.sum(diff * diff, dim=-1)
    d = int(x_t.shape[1])
    log_comp = (
        torch.log(mode_weights.clamp_min(1e-300))[None, :]
        - 0.5 * d * torch.log(2.0 * math.pi * sigma2_eff)[None, :]
        - 0.5 * dist2 / sigma2_eff[None, :]
    )
    log_resp = log_comp - torch.logsumexp(log_comp, dim=1, keepdim=True)
    resp = torch.exp(log_resp)
    precision = 1.0 / sigma2_eff[None, :]
    score_terms = resp[:, :, None] * ((modes_t[None, :, :] - x_t[:, None, :]) * precision[:, :, None])
    return torch.sum(score_terms, dim=1)


def oracle_eps_and_score_at_t(
    diffusion,
    x_t: torch.Tensor,
    gaussian_modes: torch.Tensor,
    std_dev: float | None,
    t_eff,
    *,
    mode_sigmas=None,
    mode_weights=None,
):
    # return the exact $\\epsilon$ target and the exact score at raw time $t$.
    # this is the main exact oracle used by the synthetic explicit-score branches.
    x64 = x_t.to(dtype=torch.float64)
    modes64 = torch.as_tensor(gaussian_modes, device=x_t.device, dtype=torch.float64)
    sig64 = prepare_mode_sigmas(modes64, std_dev=std_dev, mode_sigmas=mode_sigmas)
    w64 = prepare_mode_weights(modes64, mode_weights=mode_weights)

    def single_time(x_batch: torch.Tensor, t_scalar: float):
        # evaluate one exact time slice after pushing the clean means forward to time $t$.
        a_bar64 = interp_buffer_at_time(
            diffusion.alphas_prod,
            t_eff=t_scalar,
            device=x_t.device,
            dtype=torch.float64,
        )
        modes_t64 = modes64 * torch.sqrt(a_bar64)
        sigma2_eff64 = a_bar64 * (sig64 ** 2) + (1.0 - a_bar64)
        score_true64 = true_score_noised_mixture(x_batch, modes_t64, sigma2_eff64, w64)
        eps_true64 = -torch.sqrt((1.0 - a_bar64).clamp_min(1e-18)) * score_true64
        return eps_true64, score_true64

    if not torch.is_tensor(t_eff):
        eps_true64, score_true64 = single_time(x64, float(t_eff))
        return eps_true64.to(dtype=x_t.dtype), score_true64

    t_tensor = t_eff.to(device=x_t.device)
    if t_tensor.ndim == 0:
        eps_true64, score_true64 = single_time(x64, float(t_tensor.item()))
        return eps_true64.to(dtype=x_t.dtype), score_true64

    t_tensor = t_tensor.reshape(-1)
    if t_tensor.shape[0] == 1 and x64.shape[0] != 1:
        eps_true64, score_true64 = single_time(x64, float(t_tensor[0].item()))
        return eps_true64.to(dtype=x_t.dtype), score_true64
    if t_tensor.shape[0] != x64.shape[0]:
        raise ValueError(f"Expected time shape {(x64.shape[0],)}, got {tuple(t_tensor.shape)}")

    t64 = t_tensor.to(dtype=torch.float64)
    eps_parts = []
    score_parts = []
    indices = []
    for t_val in torch.unique(t64.detach()).tolist():
        mask = (t64 == float(t_val))
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        eps_part, score_part = single_time(x64.index_select(0, idx), t_val)
        eps_parts.append(eps_part)
        score_parts.append(score_part)
        indices.append(idx)

    if len(eps_parts) == 1:
        return eps_parts[0].to(dtype=x_t.dtype), score_parts[0]

    idx_cat = torch.cat(indices, dim=0)
    perm = torch.argsort(idx_cat)
    eps_out = torch.cat(eps_parts, dim=0).index_select(0, perm)
    score_out = torch.cat(score_parts, dim=0).index_select(0, perm)
    return eps_out.to(dtype=x_t.dtype), score_out


class ExactGaussianMixtureEpsModel(torch.nn.Module):
    def __init__(
        self,
        diffusion,
        gaussian_modes: np.ndarray,
        std_dev: float | None,
        mode_weights=None,
        mode_sigmas=None,
    ):
        # wrap the exact gaussian-mixture oracle in the same interface as a learned $\\epsilon_\\theta$ model.
        # this lets the shared evaluation code swap between learned and exact scores without branching everywhere.
        super().__init__()
        self.diffusion = diffusion
        modes = torch.as_tensor(gaussian_modes, dtype=torch.float64)
        if modes.ndim != 2:
            raise ValueError(f"Expected modes shape (M, D), got {tuple(modes.shape)}")
        self.register_buffer("gaussian_modes", modes, persistent=False)
        sigmas = prepare_mode_sigmas(modes, std_dev=std_dev, mode_sigmas=mode_sigmas)
        weights = prepare_mode_weights(modes, mode_weights=mode_weights)
        self.register_buffer("mode_sigmas", sigmas, persistent=False)
        self.register_buffer("mode_weights", weights, persistent=False)
        self.std_dev = None if std_dev is None else float(std_dev)

    def forward_single_time(self, x_t: torch.Tensor, t_eff) -> torch.Tensor:
        # evaluate the exact $\\epsilon$ oracle at one raw time.
        eps_true, _ = oracle_eps_and_score_at_t(
            diffusion=self.diffusion,
            x_t=x_t,
            gaussian_modes=self.gaussian_modes,
            std_dev=self.std_dev,
            t_eff=t_eff,
            mode_sigmas=self.mode_sigmas,
            mode_weights=self.mode_weights,
        )
        return eps_true

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        # vectorize the exact $\\epsilon$ oracle over either scalar or batched timesteps.
        if x_t.ndim != 2:
            raise ValueError(f"Expected x_t shape (N, D), got {tuple(x_t.shape)}")

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x_t.device)
        t = t.to(device=x_t.device)
        if t.ndim == 0:
            return self.forward_single_time(x_t, float(t.item()))

        t = t.reshape(-1)
        if t.shape[0] == 1 and x_t.shape[0] != 1:
            return self.forward_single_time(x_t, float(t[0].item()))
        if t.shape[0] != x_t.shape[0]:
            raise ValueError(f"Expected time shape {(x_t.shape[0],)}, got {tuple(t.shape)}")

        t64 = t.to(dtype=torch.float64)
        unique_t = torch.unique(t64.detach())
        parts = []
        indices = []
        for t_val in unique_t.tolist():
            mask = (t64 == float(t_val))
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            parts.append(self.forward_single_time(x_t.index_select(0, idx), t_val))
            indices.append(idx)
        if len(parts) == 1:
            return parts[0]
        out_cat = torch.cat(parts, dim=0)
        idx_cat = torch.cat(indices, dim=0)
        perm = torch.argsort(idx_cat)
        return out_cat.index_select(0, perm)


class NoisyExactGaussianMixtureEpsModel(ExactGaussianMixtureEpsModel):
    def __init__(
        self,
        diffusion,
        gaussian_modes: np.ndarray,
        std_dev: float | None,
        *,
        score_noise_std: float,
        score_noise_seed: int = 42,
        mode_weights=None,
        mode_sigmas=None,
    ):
        # init.
        # this supports the exact gaussian-mixture score calculations used across the paper's synthetic experiments.
        super().__init__(
            diffusion=diffusion,
            gaussian_modes=gaussian_modes,
            std_dev=std_dev,
            mode_weights=mode_weights,
            mode_sigmas=mode_sigmas,
        )
        self.score_noise_std = float(score_noise_std)
        self.score_noise_seed = int(score_noise_seed)
        self.stochastic_model = bool(self.score_noise_std > 0.0)
        self.noise_space = "score"
        self.noise_generators = {}

    def generator_for(self, device: torch.device) -> torch.Generator:
        # generator for.
        # this supports the exact gaussian-mixture score calculations used across the paper's synthetic experiments.
        key = str(device)
        gen = self.noise_generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.score_noise_seed)
            self.noise_generators[key] = gen
        return gen

    def forward_single_time(self, x_t: torch.Tensor, t_eff) -> torch.Tensor:
        # forward single time.
        # this supports the exact gaussian-mixture score calculations used across the paper's synthetic experiments.
        eps_true, score_true = oracle_eps_and_score_at_t(
            diffusion=self.diffusion,
            x_t=x_t,
            gaussian_modes=self.gaussian_modes,
            std_dev=self.std_dev,
            t_eff=t_eff,
            mode_sigmas=self.mode_sigmas,
            mode_weights=self.mode_weights,
        )
        if self.score_noise_std <= 0.0:
            return eps_true

        a_bar64 = interp_buffer_at_time(
            self.diffusion.alphas_prod,
            t_eff=t_eff,
            device=x_t.device,
            dtype=torch.float64,
        )
        gen = self.generator_for(x_t.device)
        noise = torch.randn(
            score_true.shape,
            device=x_t.device,
            dtype=torch.float64,
            generator=gen,
        ) * self.score_noise_std
        score_noisy = score_true.to(dtype=torch.float64) + noise
        eps_noisy = -torch.sqrt((1.0 - a_bar64).clamp_min(1e-18)) * score_noisy
        return eps_noisy.to(dtype=x_t.dtype)
