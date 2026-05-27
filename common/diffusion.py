import math

import numpy as np
import torch
import torch.nn as nn

from common.config_utils import cfg_get


def to_float_tensor(value) -> torch.Tensor:
    # cast schedule inputs into float tensors before building the vp diffusion buffers.
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


class Diffusion(nn.Module):
    def __init__(self, args=None, device="cuda", schedule_metadata=None):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        super().__init__()
        if args is None and schedule_metadata is None:
            raise ValueError("Diffusion requires args or schedule_metadata.")

        self.sampling_type = str(cfg_get(args, "sampling_mode", "ddpm")).lower() if args is not None else "ddpm"

        if schedule_metadata is not None:
            betas = to_float_tensor(schedule_metadata["betas"])
            alphas = to_float_tensor(schedule_metadata.get("alphas", 1.0 - betas))
            alphas_prod = to_float_tensor(schedule_metadata["alphas_prod"])
            beta_schedule = str(schedule_metadata.get("beta_schedule", "linear")).lower()
            schedule_params = dict(schedule_metadata.get("schedule_params", {}))
            lambda_vals = to_float_tensor(
                schedule_metadata.get(
                    "lambda_vals",
                    torch.log(alphas_prod.clamp_min(1e-12) / (1.0 - alphas_prod).clamp_min(1e-12)),
                )
            )
            high_snr_top_frac = float(schedule_metadata.get("high_snr_top_frac", 0.0))
            q_probs = schedule_metadata.get("q_probs", None)
            importance_weights = schedule_metadata.get("importance_weights", None)
        else:
            steps = int(cfg_get(args, "timesteps", 1000))
            beta_start = float(cfg_get(args, "beta_start", 1e-4))
            beta_end = float(cfg_get(args, "beta_end", 2e-2))
            beta_schedule = str(cfg_get(args, "diffusion.beta_schedule", cfg_get(args, "beta_schedule", "linear"))).lower()
            cosine_s = float(cfg_get(args, "diffusion.cosine_s", 0.008))
            high_snr_top_frac = float(cfg_get(args, "loss.high_snr_top_frac", 0.0))
            betas, alphas, alphas_prod = self.build_schedule(
                steps=steps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                cosine_s=cosine_s,
            )
            schedule_params = {
                "timesteps": int(steps),
                "beta_start": float(beta_start),
                "beta_end": float(beta_end),
                "cosine_s": float(cosine_s),
            }
            lambda_vals = torch.log(alphas_prod.clamp_min(1e-12) / (1.0 - alphas_prod).clamp_min(1e-12))
            q_probs = None
            importance_weights = None

        self.steps = int(betas.shape[0])
        self.beta_schedule = beta_schedule
        self.schedule_params = dict(schedule_params)
        self.high_snr_top_frac = float(high_snr_top_frac)

        alphas_prev_prod = torch.cat(
            [torch.ones((1,), dtype=torch.float32), alphas_prod[:-1]],
            dim=0,
        )
        variance = betas * (1.0 - alphas_prev_prod) / (1.0 - alphas_prod).clamp_min(1e-12)

        if q_probs is None or importance_weights is None:
            q_probs, importance_weights = self.build_importance_sampling_probs(
                lambda_vals=lambda_vals,
                top_frac=self.high_snr_top_frac,
            )
        else:
            q_probs = to_float_tensor(q_probs)
            importance_weights = to_float_tensor(importance_weights)

        for name, tensor in {
            "betas": betas,
            "alphas": alphas,
            "alphas_prod": alphas_prod,
            "alphas_prev_prod": alphas_prev_prod,
            "sqrt_alphas_prod": torch.sqrt(alphas_prod),
            "sqrt_one_minus_alphas_prod": torch.sqrt((1.0 - alphas_prod).clamp_min(1e-12)),
            "sqrt_recip_alphas": torch.sqrt(1.0 / alphas.clamp_min(1e-12)),
            "variance": variance.clamp_min(1e-12),
            "lambda_vals": lambda_vals,
            "q_probs": q_probs,
            "importance_weights": importance_weights,
        }.items():
            self.register_buffer(name, tensor.to(device=device, dtype=torch.float32))

    @staticmethod
    def build_schedule(steps: int, beta_schedule: str, beta_start: float, beta_end: float, cosine_s: float):
        # build schedule.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        beta_schedule = str(beta_schedule).lower()
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
            x = (t / float(steps) + cosine_s) / (1.0 + cosine_s)
            alphas_bar = torch.cos(x * math.pi * 0.5).pow(2)
            alphas_bar = alphas_bar / alphas_bar[0].clamp_min(1e-12)
            betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1].clamp_min(1e-12))
            betas = betas.clamp(min=1e-8, max=0.999)
        else:
            raise ValueError(f"Unsupported beta_schedule={beta_schedule!r}")
        alphas = 1.0 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_prod

    @staticmethod
    def build_importance_sampling_probs(lambda_vals: torch.Tensor, top_frac: float):
        # build importance sampling probs.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        steps = int(lambda_vals.shape[0])
        base = torch.full((steps,), 1.0 / float(steps), dtype=torch.float32)
        if top_frac <= 0.0:
            return base, torch.ones_like(base)
        top_k = max(1, min(steps, int(round(float(top_frac) * float(steps)))))
        sorted_idx = torch.argsort(lambda_vals, descending=True)
        high_idx = sorted_idx[:top_k]
        q_probs = 0.5 * base
        q_probs[high_idx] += 0.5 / float(top_k)
        importance_weights = (1.0 / float(steps)) / q_probs.clamp_min(1e-12)
        return q_probs, importance_weights

    def export_schedule_metadata(self) -> dict:
        # export schedule metadata.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        beta_schedule = getattr(self, "beta_schedule", "linear")
        schedule_params = dict(getattr(self, "schedule_params", {}) or {})
        high_snr_top_frac = float(getattr(self, "high_snr_top_frac", 0.0))
        lambda_vals = getattr(self, "lambda_vals", None)
        if lambda_vals is None:
            alphas_prod = self.alphas_prod.detach().cpu().clamp(min=1e-12, max=1.0 - 1e-12)
            lambda_vals = torch.log(alphas_prod / (1.0 - alphas_prod))
        q_probs = getattr(self, "q_probs", None)
        importance_weights = getattr(self, "importance_weights", None)
        if q_probs is None or importance_weights is None:
            steps = int(self.betas.shape[0])
            q_probs = torch.full((steps,), 1.0 / float(steps), dtype=torch.float32)
            importance_weights = torch.ones((steps,), dtype=torch.float32)
        return {
            "beta_schedule": beta_schedule,
            "schedule_params": schedule_params,
            "betas": self.betas.detach().cpu(),
            "alphas": self.alphas.detach().cpu(),
            "alphas_prod": self.alphas_prod.detach().cpu(),
            "lambda_vals": lambda_vals.detach().cpu(),
            "q_probs": q_probs.detach().cpu(),
            "importance_weights": importance_weights.detach().cpu(),
            "high_snr_top_frac": high_snr_top_frac,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict, *, args=None, device="cpu"):
        # from checkpoint payload.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if payload is None:
            raise ValueError("Checkpoint payload is required.")

        schedule_keys = {
            "beta_schedule",
            "schedule_params",
            "betas",
            "alphas",
            "alphas_prod",
            "lambda_vals",
            "q_probs",
            "importance_weights",
            "high_snr_top_frac",
        }
        if schedule_keys.issubset(set(payload.keys())):
            return cls(
                args=args,
                device=device,
                schedule_metadata={key: payload[key] for key in schedule_keys},
            )

        diffusion_obj = payload.get("diffusion", None)
        if isinstance(diffusion_obj, Diffusion):
            return diffusion_obj.to(device)

        payload_args = payload.get("args", args)
        return cls(args=payload_args, device=device)

    def lambda_bin_indices(self, num_bins: int = 20) -> torch.Tensor:
        # lambda bin indices.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        num_bins = int(num_bins)
        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")
        sorted_idx = torch.argsort(self.lambda_vals, descending=True)
        bin_ids = torch.empty((self.steps,), device=self.lambda_vals.device, dtype=torch.long)
        for bin_id, idx in enumerate(torch.chunk(sorted_idx, num_bins)):
            bin_ids[idx] = int(bin_id)
        return bin_ids

    def sample_timesteps(self, batch_size: int, *, device=None):
        # sample timesteps.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        target_device = self.q_probs.device if device is None else torch.device(device)
        q_probs = self.q_probs.to(target_device)
        t = torch.multinomial(q_probs, int(batch_size), replacement=True)
        weights = self.importance_weights.to(target_device).index_select(0, t)
        return t, weights

    def importance_weights_for_t(self, t: torch.Tensor) -> torch.Tensor:
        # importance weights for t.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        t_long = t.long().reshape(-1)
        return self.importance_weights.index_select(0, t_long)

    def compute_alpha_t_bar(self, t):
        # compute alpha t bar.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return self.alphas_prod[t]

    def compute_ddim_alpha_t_bar_list(self, sampling_type="quad", ddim_steps=50, total_time_steps=1000, sigma=0.02):
        # compute ddim alpha t bar list.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        seq = reversed(np.linspace(0, np.sqrt(total_time_steps * 0.8), ddim_steps) ** 2)
        seq = [int(s) for s in list(seq)]
        return [self.compute_alpha_t_bar(t).item() for t in seq]

    def make_ddim_seq(self, skip_type: str, ddim_steps: int):
        # make ddim seq.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        skip_type = skip_type.lower()
        if skip_type == "uniform":
            seq = np.linspace(0, self.steps - 1, ddim_steps)
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(self.steps * 0.8), ddim_steps) ** 2
        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")

        seq = np.round(seq).astype(int)
        seq = np.clip(seq, 0, self.steps - 1)
        for k in range(1, len(seq)):
            if seq[k] <= seq[k - 1]:
                seq[k] = seq[k - 1] + 1
        overflow = seq[-1] - (self.steps - 1)
        if overflow > 0:
            seq = seq - overflow
        seq = np.clip(seq, 0, self.steps - 1)
        for k in range(len(seq) - 2, -1, -1):
            if seq[k] >= seq[k + 1]:
                seq[k] = seq[k + 1] - 1
        seq = np.clip(seq, 0, self.steps - 1)
        return list(seq.tolist())

    def effective_variance(
        self,
        sampling_mode="ddim",
        sampling_type="quad",
        num_steps=50,
        total_time_steps=1000,
        sigma=0.02,
        seq=None,
        return_seq=False,
    ):
        # effective variance.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if seq is not None:
            seq = list(seq)
        else:
            sampling_mode = sampling_mode.lower()
            sampling_type = sampling_type.lower()
            if sampling_mode == "ddpm":
                seq = list(range(total_time_steps - 1, -1, -1))
            elif sampling_mode == "ddim":
                seq_inc = self.make_ddim_seq(sampling_type, num_steps)
                seq = list(reversed(seq_inc))
            else:
                raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

        if num_steps is not None:
            if len(seq) < num_steps:
                raise ValueError(f"Schedule length {len(seq)} < num_steps {num_steps}")
            seq = seq[:num_steps]

        alphas_prod_cpu = self.alphas_prod.detach().cpu().numpy()
        eff_variance = np.zeros(len(seq), dtype=np.float64)
        for idx, t_eff in enumerate(seq):
            a_bar = float(alphas_prod_cpu[int(t_eff)])
            eff_variance[idx] = (sigma ** 2) + (1.0 / a_bar) * (1.0 - a_bar)

        return (eff_variance, seq) if return_seq else eff_variance

    @staticmethod
    def view_shape_like(x: torch.Tensor) -> tuple[int, ...]:
        # view shape like.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return (x.shape[0],) + (1,) * (x.ndim - 1)

    def gather_like(self, buf: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # gather like.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        values = buf.gather(-1, t.long().reshape(-1)).view(self.view_shape_like(x))
        return values.to(device=x.device, dtype=x.dtype)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # forward process.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        sqrt_alphas_prod_t = self.gather_like(self.sqrt_alphas_prod, t, x0)
        sqrt_one_minus_t = self.gather_like(self.sqrt_one_minus_alphas_prod, t, x0)
        return sqrt_alphas_prod_t * x0 + sqrt_one_minus_t * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # predict x0 from eps.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        sqrt_alphas_prod_t = self.gather_like(self.sqrt_alphas_prod, t, x_t)
        sqrt_one_minus_t = self.gather_like(self.sqrt_one_minus_alphas_prod, t, x_t)
        return (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alphas_prod_t.clamp_min(1e-12)

    def denoising_step(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # denoising step.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        sqrt_recip_alphas_t = self.gather_like(self.sqrt_recip_alphas, t, xt)
        sqrt_one_minus_alphas_prod_t = self.gather_like(self.sqrt_one_minus_alphas_prod, t, xt)
        betas_t = self.gather_like(self.betas, t, xt)

        predicted_noise = model(xt, t)
        predicted_mean = sqrt_recip_alphas_t * (xt - (betas_t / sqrt_one_minus_alphas_prod_t.clamp_min(1e-12)) * predicted_noise)
        variance = self.gather_like(self.variance, t, xt).clamp_min(1e-8)
        noise = torch.randn_like(xt)
        nonzero_mask = (t > 0).view(self.view_shape_like(xt)).to(dtype=xt.dtype, device=xt.device)
        return predicted_mean + nonzero_mask * torch.sqrt(variance) * noise

    def construct_image(
        self,
        model: nn.Module,
        shape,
        device,
        input=None,
        labels=None,
        save_intermediate=False,
        step_interval=1,
        callback=None,
    ):
        # construct image.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        model.eval()
        if input is not None:
            x = input.to(device)
        else:
            x = torch.randn(shape, device=device)

        batch_size = shape[0]
        if labels is None and hasattr(model, "y_embedder"):
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        for t in range(self.steps - 1, -1, -1):
            t_list = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if labels is not None and hasattr(model, "y_embedder"):
                predicted_noise = model(x, t_list, labels)
            else:
                predicted_noise = model(x, t_list)

            sqrt_recip_alphas_t = self.gather_like(self.sqrt_recip_alphas, t_list, x)
            sqrt_one_minus_alphas_prod_t = self.gather_like(self.sqrt_one_minus_alphas_prod, t_list, x)
            betas_t = self.gather_like(self.betas, t_list, x)
            variance = self.gather_like(self.variance, t_list, x).clamp_min(1e-8)
            predicted_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_one_minus_alphas_prod_t.clamp_min(1e-12)) * predicted_noise)

            noise = torch.randn_like(x)
            nonzero_mask = (t_list > 0).view(self.view_shape_like(x)).to(dtype=x.dtype)
            x = predicted_mean + nonzero_mask * torch.sqrt(variance) * noise

            if save_intermediate and t % step_interval == 0 and callback is not None:
                callback(x.clone(), t)

        model.train()
        return x

    def compute_alpha(self, t):
        # compute alpha.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        betas_padded = torch.cat([torch.zeros(1, device=self.betas.device), self.betas], dim=0)
        alphas_padded = 1.0 - betas_padded
        alphas_cumprod = alphas_padded.cumprod(dim=0)
        return alphas_cumprod[t + 1].view(-1, 1, 1, 1)

    @staticmethod
    def stack_xs_as_list(XS):
        # stack xs as list.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        out = [XS[0]]
        shape = XS[0].shape
        for x in XS[1:]:
            out.append(x.view(shape))
        return out

    def ddim_denoising_steps(self, x, seq, model, b, **kwargs):
        # ddim denoising steps.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])

            if kwargs["tau_start"] is not None:
                seq = [s for s in seq if s < kwargs["tau_start"]]
                seq_next = seq_next[: len(seq)]

            x0_preds = []
            xs = [x.detach().cpu()]

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((n,), i, device=x.device, dtype=torch.long)
                next_t = torch.full((n,), j, device=x.device, dtype=torch.long)

                at = self.compute_alpha(t.long()).view(-1, 1, 1, 1)
                at_next = self.compute_alpha(next_t.long()).view(-1, 1, 1, 1)

                xt = xs[-1].to(x.device)
                et = model(xt, t)
                at_v = at.view([et.shape[0], 1])
                at_next_v = at_next.view([at_next.shape[0], 1])

                at_v_exp = at_v.view(-1, *([1] * (xt.ndim - 1)))
                at_next_v_exp = at_next_v.view(-1, *([1] * (xt.ndim - 1)))

                x0_t = (xt - et * (1.0 - at_v_exp).sqrt()) / at_v_exp.sqrt()
                x0_preds.append(x0_t.detach().cpu())

                eta = kwargs.get("eta", 0)
                c1 = eta * ((1.0 - at_v / at_next_v) * (1.0 - at_next_v) / (1.0 - at_v)).sqrt()
                c2 = ((1.0 - at_next_v) - c1 ** 2).sqrt()

                c1_exp = c1.view(-1, *([1] * (xt.ndim - 1)))
                c2_exp = c2.view(-1, *([1] * (xt.ndim - 1)))
                xt_next = at_next_v_exp.sqrt() * x0_t + c1_exp * torch.randn_like(xt) + c2_exp * et
                xs.append(xt_next.detach().cpu())

        return xs, x0_preds

    def ddim_denoising_steps_with_jacobian(self, x, seq, model, b, compute_eig_values_at, **kwargs):
        # ddim denoising steps with jacobian.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        device = x.device
        n = x.size(0)
        eval_points = torch.as_tensor(compute_eig_values_at, dtype=x.dtype, device=device)
        if eval_points.ndim != 2:
            raise ValueError("compute_eig_values_at must have shape (P, D).")

        P, D = eval_points.shape
        seq = list(seq)
        seq_next = [-1] + seq[:-1]

        x0_preds = []
        xs = [x.detach().cpu()]
        all_eigvals = []
        all_t_cur = []
        all_dt = []

        def ddim_step_deterministic(xt, t_scalar, next_t_scalar):
            # ddim step deterministic.
            # this supports the shared training, model, and diffusion utilities used throughout the repo.
            t = torch.full((xt.shape[0],), t_scalar, device=xt.device, dtype=torch.long)
            next_t = torch.full((xt.shape[0],), next_t_scalar, device=xt.device, dtype=torch.long)

            at = self.compute_alpha(t).view(-1, 1)
            at_next = self.compute_alpha(next_t).view(-1, 1)
            et = model(xt, t)
            x0_t = (xt - et * (1.0 - at).sqrt()) / at.sqrt()
            c2 = (1.0 - at_next).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et
            return xt_next, x0_t

        for i, j in zip(reversed(seq), reversed(seq_next)):
            j_eff = j if j >= 0 else -1
            dt = float(i - j_eff)
            if dt <= 0:
                continue

            t_batch = torch.full((n,), i, device=device, dtype=torch.long)
            next_t_batch = torch.full((n,), j, device=device, dtype=torch.long)

            xt_traj = xs[-1].to(device)
            at_full = self.compute_alpha(t_batch).view(-1, 1)
            at_next_full = self.compute_alpha(next_t_batch).view(-1, 1)
            et_full = model(xt_traj, t_batch)
            x0_t_full = (xt_traj - et_full * (1.0 - at_full).sqrt()) / at_full.sqrt()
            c2_full = (1.0 - at_next_full).sqrt()
            xt_next_full = at_next_full.sqrt() * x0_t_full + c2_full * et_full

            x0_preds.append(x0_t_full.detach().cpu())
            xs.append(xt_next_full.detach().cpu())

            eigvals_this_step = []

            def F_single(x_in):
                # F single.
                # this supports the shared training, model, and diffusion utilities used throughout the repo.
                xt_next_single, _ = ddim_step_deterministic(x_in, i, j)
                return (xt_next_single - x_in) / dt

            for p_idx in range(P):
                x_point = eval_points[p_idx : p_idx + 1].detach().requires_grad_(True)
                J = torch.autograd.functional.jacobian(F_single, x_point, create_graph=False)
                J = J[0, :, 0, :]
                eigvals = torch.linalg.eigvals(J).detach().cpu().numpy()
                eigvals_this_step.append(eigvals)

            eigvals_this_step = np.stack(eigvals_this_step, axis=0)
            all_eigvals.append(eigvals_this_step)
            all_t_cur.append(i)
            all_dt.append(dt)

        return xs, x0_preds, all_eigvals, all_t_cur, all_dt

    def ddim_sampling(
        self,
        model,
        shape,
        device="cuda",
        ddim_steps=50,
        ddim_eta=0.0,
        skip_type="quad",
        retrieve_all_samples=False,
        input_noise=None,
        save_intermediate=False,
        step_interval=1,
        callback=None,
        compute_jacobian=False,
        tau_start=None,
        compute_eig_values_at=None,
    ):
        # ddim sampling.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        model.eval()
        device = self.betas.device

        if input_noise is not None:
            x = input_noise.to(device)
        else:
            x = torch.randn(shape, device=device)

        seq = self.make_ddim_seq(skip_type, ddim_steps)
        print(f"DDIM sampling: {ddim_steps} steps, eta={ddim_eta}, skip_type={skip_type}")
        print(f"Timestep sequence length: {len(seq)}")
        print(f"Sequence is {seq}")

        if not compute_jacobian:
            xs, x0_preds = self.ddim_denoising_steps(
                x=x,
                seq=seq,
                model=model,
                b=self.betas,
                tau_start=tau_start,
                eta=ddim_eta,
            )
            if retrieve_all_samples:
                xs_stacked = self.stack_xs_as_list(xs)
                model.train()
                return xs_stacked, list(reversed(seq))

            samples = xs[-1]
            model.train()
            return samples, list(reversed(seq))

        xs, x0_preds, all_eigvals, t_cur, dt_list = self.ddim_denoising_steps_with_jacobian(
            x=x,
            seq=seq,
            model=model,
            b=self.betas,
            compute_eig_values_at=compute_eig_values_at,
        )
        xs_stacked = self.stack_xs_as_list(xs)
        model.train()
        return xs_stacked, list(reversed(seq)), all_eigvals, t_cur, dt_list
