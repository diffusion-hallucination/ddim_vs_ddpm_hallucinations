import torch


def call_model(model, x_t: torch.Tensor, t_eff: int) -> torch.Tensor:
    t = torch.full((x_t.shape[0],), int(t_eff), device=x_t.device, dtype=torch.long)
    try:
        return model(x_t, t)
    except Exception:
        return model(x_t, t.float())


def make_reverse_seq(
    diffusion,
    sampling_mode: str,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
):
    mode = str(sampling_mode).lower()
    if mode == "ddpm":
        return list(range(int(timesteps) - 1, -1, -1))

    if mode == "ddim":
        _, seq = diffusion.effective_variance(
            sampling_mode="ddim",
            sampling_type=str(skip_type),
            num_steps=int(ddim_steps),
            total_time_steps=int(timesteps),
            sigma=0.02,
            return_seq=True,
        )
        return list(seq)

    raise ValueError(f"Only DDIM and DDPM reverse solvers are supported, got {sampling_mode!r}.")


def ddpm_step_from_eps(
    diffusion,
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_eff: int,
) -> torch.Tensor:
    batch_size = x_t.shape[0]
    t = torch.full((batch_size,), int(t_eff), device=x_t.device, dtype=torch.long)

    sqrt_recip_alphas_t = diffusion.sqrt_recip_alphas.gather(-1, t)
    sqrt_one_minus_alphas_prod_t = diffusion.sqrt_one_minus_alphas_prod.gather(-1, t)
    betas_t = diffusion.betas.gather(-1, t)
    variance_t = diffusion.variance.gather(-1, t).clamp_min(1e-8)

    view_shape = (-1,) + (1,) * (x_t.ndim - 1)
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(view_shape)
    sqrt_one_minus_alphas_prod_t = sqrt_one_minus_alphas_prod_t.view(view_shape)
    betas_t = betas_t.view(view_shape)
    variance_t = variance_t.view(view_shape)

    mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_prod_t) * eps_pred)
    noise = torch.randn_like(x_t)
    nonzero_mask = (t > 0).view(view_shape).to(x_t.dtype)
    return mean + nonzero_mask * torch.sqrt(variance_t) * noise


def ddpm_step_from_eps_with_generator(
    diffusion,
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_eff: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = x_t.shape[0]
    t = torch.full((batch_size,), int(t_eff), device=x_t.device, dtype=torch.long)

    sqrt_recip_alphas_t = diffusion.sqrt_recip_alphas.gather(-1, t)
    sqrt_one_minus_alphas_prod_t = diffusion.sqrt_one_minus_alphas_prod.gather(-1, t)
    betas_t = diffusion.betas.gather(-1, t)
    variance_t = diffusion.variance.gather(-1, t).clamp_min(1e-8)

    view_shape = (-1,) + (1,) * (x_t.ndim - 1)
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(view_shape)
    sqrt_one_minus_alphas_prod_t = sqrt_one_minus_alphas_prod_t.view(view_shape)
    betas_t = betas_t.view(view_shape)
    variance_t = variance_t.view(view_shape)

    mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_prod_t) * eps_pred)
    noise = torch.randn(
        x_t.shape,
        device=x_t.device,
        dtype=x_t.dtype,
        generator=generator,
    )
    nonzero_mask = (t > 0).view(view_shape).to(x_t.dtype)
    return mean + nonzero_mask * torch.sqrt(variance_t) * noise


def ddim_step_from_eps(
    diffusion,
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_eff: int,
    t_next: int,
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    at = diffusion.alphas_prod[int(t_eff)].to(device=x_t.device, dtype=x_t.dtype)
    if int(t_next) < 0:
        at_next = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
    else:
        at_next = diffusion.alphas_prod[int(t_next)].to(device=x_t.device, dtype=x_t.dtype)

    x0 = (x_t - eps_pred * torch.sqrt((1.0 - at).clamp_min(1e-12))) / torch.sqrt(at.clamp_min(1e-12))

    if float(ddim_eta) == 0.0:
        return torch.sqrt(at_next) * x0 + torch.sqrt((1.0 - at_next).clamp_min(0.0)) * eps_pred

    c1 = float(ddim_eta) * torch.sqrt(
        ((1.0 - at / at_next).clamp_min(0.0) * (1.0 - at_next).clamp_min(0.0))
        / (1.0 - at).clamp_min(1e-12)
    )
    c2 = torch.sqrt((1.0 - at_next - c1 * c1).clamp_min(0.0))
    return torch.sqrt(at_next) * x0 + c1 * torch.randn_like(x_t) + c2 * eps_pred


def build_reverse_solver_seq(
    diffusion,
    sampling_mode: str,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
):
    return list(
        make_reverse_seq(
            diffusion=diffusion,
            sampling_mode=str(sampling_mode).lower(),
            timesteps=int(timesteps),
            ddim_steps=int(ddim_steps),
            skip_type=str(skip_type),
        )
    )


def build_reverse_solver_stepper(
    diffusion,
    sampling_mode: str,
    timesteps: int,
    ddim_steps: int,
    skip_type: str,
    ddim_eta: float = 0.0,
):
    mode = str(sampling_mode).lower()
    seq = build_reverse_solver_seq(
        diffusion=diffusion,
        sampling_mode=mode,
        timesteps=int(timesteps),
        ddim_steps=int(ddim_steps),
        skip_type=str(skip_type),
    )

    def step_fn(model, x_t, t_eff, t_next, prev_x0=None, prev_h=None):
        eps_pred = call_model(model, x_t, int(t_eff))

        if mode == "ddpm":
            x_next = ddpm_step_from_eps(
                diffusion=diffusion,
                x_t=x_t,
                eps_pred=eps_pred,
                t_eff=int(t_eff),
            )
            return x_next, None, None

        if mode == "ddim":
            x_next = ddim_step_from_eps(
                diffusion=diffusion,
                x_t=x_t,
                eps_pred=eps_pred,
                t_eff=int(t_eff),
                t_next=int(t_next),
                ddim_eta=float(ddim_eta),
            )
            return x_next, None, None

        raise ValueError(f"Only DDIM and DDPM reverse solvers are supported, got {sampling_mode!r}.")

    return seq, step_fn


def solver_tag_from_cfg(eval_cfg, sampling_mode: str | None = None) -> str:
    mode = str(sampling_mode or eval_cfg.sampling_mode).lower()
    if mode == "ddpm":
        return f"ddpm_{int(eval_cfg.timesteps)}"
    if mode == "ddim":
        return f"ddim_{eval_cfg.skip_type}_{int(eval_cfg.ddim_timesteps)}"
    raise ValueError(f"Only DDIM and DDPM reverse solvers are supported, got {mode!r}.")


def solver_display_name(sampling_mode: str) -> str:
    mode = str(sampling_mode).lower()
    if mode == "ddim":
        return "DDIM"
    if mode == "ddpm":
        return "DDPM"
    raise ValueError(f"Only DDIM and DDPM reverse solvers are supported, got {sampling_mode!r}.")
