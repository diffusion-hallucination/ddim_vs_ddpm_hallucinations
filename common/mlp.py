import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal embedding for diffusion conditioning scalars.
    """

    def __init__(self, dim: int):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # forward.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.reshape(-1).float()
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            return t.unsqueeze(1)
        if half_dim == 1:
            freqs = torch.ones((1,), device=device, dtype=t.dtype)
        else:
            emb_scale = math.log(10000) / float(half_dim - 1)
            freqs = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb_scale)
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device, dtype=emb.dtype)], dim=1)
        return emb


class MLPBlock(nn.Module):
    """
    Residual MLP block:
    LN -> LeakyReLU -> Linear -> (+ conditioning) -> LeakyReLU -> Linear + skip
    """

    def __init__(self, hidden_dim: int, cond_dim: int | None = None):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim is not None else None
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # forward.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        h = self.norm(x)
        h = self.act(h)
        h = self.fc1(h)
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Conditioning tensor is required for per-block MLP conditioning.")
            h = h + self.cond_proj(cond)
        h = self.act(h)
        h = self.fc2(h)
        return x + h


class MLP(nn.Module):
    """
    Gaussian diffusion MLP.

    The main 2D Gaussian run uses raw-input, timestep-conditioned,
    single-input-injection settings. High-dimensional Gaussian runs switch to:
    - lambda-lookup conditioning
    - per-block conditioning injection
    - frozen random Fourier features on the input
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 128,
        out_dim: int = 2,
        num_blocks: int = 3,
        conditioning_type: str = "timestep",
        conditioning_strategy: str = "input_add",
        input_encoding: str = "raw",
        fourier_features: int = 256,
        fourier_scale: float = 0.5,
        fourier_seed: int | None = None,
        include_input_identity: bool = True,
        lambda_vals: torch.Tensor | None = None,
        num_diffusion_steps: int | None = None,
    ):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(time_dim)
        self.out_dim = int(out_dim)
        self.num_blocks = int(num_blocks)
        self.conditioning_type = str(conditioning_type).strip().lower()
        self.conditioning_strategy = str(conditioning_strategy).strip().lower()
        self.input_encoding = str(input_encoding).strip().lower()
        self.fourier_features = int(fourier_features)
        self.fourier_scale = float(fourier_scale)
        self.include_input_identity = bool(include_input_identity)

        if self.conditioning_type not in {"timestep", "lambda_lookup"}:
            raise ValueError(f"Unsupported conditioning_type={conditioning_type!r}")
        if self.conditioning_strategy not in {"input_add", "per_block"}:
            raise ValueError(f"Unsupported conditioning_strategy={conditioning_strategy!r}")
        if self.input_encoding not in {"raw", "fourier"}:
            raise ValueError(f"Unsupported input_encoding={input_encoding!r}")
        if self.input_encoding == "fourier" and self.fourier_features <= 0:
            raise ValueError("fourier_features must be positive when input_encoding='fourier'.")

        encoded_in_dim = self.in_dim
        if self.input_encoding == "fourier":
            generator = torch.Generator(device="cpu")
            if fourier_seed is not None:
                generator.manual_seed(int(fourier_seed))
            matrix = torch.randn((self.fourier_features, self.in_dim), generator=generator, dtype=torch.float32)
            matrix = matrix * self.fourier_scale
            self.register_buffer("fourier_matrix", matrix)
            encoded_in_dim = 2 * self.fourier_features
            if self.include_input_identity:
                encoded_in_dim += self.in_dim

        if self.conditioning_type == "lambda_lookup":
            if lambda_vals is None:
                num_steps = int(num_diffusion_steps or 0)
                if num_steps <= 0:
                    raise ValueError("num_diffusion_steps must be positive for lambda_lookup conditioning.")
                lambda_vals = torch.zeros((num_steps,), dtype=torch.float32)
            lambda_vals = torch.as_tensor(lambda_vals, dtype=torch.float32).reshape(-1)
            self.register_buffer("lambda_lookup", lambda_vals)

        if self.conditioning_strategy == "input_add":
            self.condition_mlp = nn.Sequential(
                SinusoidalPosEmb(self.condition_dim),
                nn.Linear(self.condition_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )
            block_cond_dim = None
        else:
            self.condition_mlp = nn.Sequential(
                SinusoidalPosEmb(self.condition_dim),
                nn.Linear(self.condition_dim, self.condition_dim),
                nn.LeakyReLU(0.2),
            )
            block_cond_dim = self.condition_dim

        self.input_layer = nn.Linear(encoded_in_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim, cond_dim=block_cond_dim) for _ in range(self.num_blocks)])
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_inputs(self, x: torch.Tensor) -> torch.Tensor:
        # encode inputs.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if self.input_encoding != "fourier":
            return x
        phases = x @ self.fourier_matrix.t()
        feats = [torch.sin(phases), torch.cos(phases)]
        if self.include_input_identity:
            feats.insert(0, x)
        return torch.cat(feats, dim=1)

    def resolve_condition_scalar(self, t: torch.Tensor, batch_size: int, device) -> torch.Tensor:
        # resolve condition scalar.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device)
        t = t.to(device=device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        else:
            t = t.reshape(-1)
            if t.shape[0] == 1 and batch_size != 1:
                t = t.repeat(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError(f"Conditioning batch mismatch: got {t.shape[0]} for batch_size={batch_size}.")

        if self.conditioning_type == "timestep":
            return t.float()

        lookup = self.lambda_lookup.to(device=device)
        if lookup.numel() == 0:
            raise ValueError("lambda_lookup conditioning requested without stored lambda values.")
        t_float = t.float().clamp(0.0, float(lookup.shape[0] - 1))
        t0 = torch.floor(t_float).long()
        t1 = torch.ceil(t_float).long()
        w = t_float - t0.float()
        v0 = lookup.index_select(0, t0)
        v1 = lookup.index_select(0, t1)
        return (1.0 - w) * v0 + w * v1

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # forward.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        cond_scalar = self.resolve_condition_scalar(t, batch_size=x.shape[0], device=x.device)
        cond = self.condition_mlp(cond_scalar)
        h = self.input_layer(self.encode_inputs(x))
        if self.conditioning_strategy == "input_add":
            h = h + cond
            for block in self.blocks:
                h = block(h, None)
        else:
            for block in self.blocks:
                h = block(h, cond)
        return self.output_layer(h)


def create_model(model_type="mlp", **kwargs):
    # construct the mlp score model with the requested conditioning and input encoding choices.
    if model_type.lower() == "mlp":
        return MLP(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
