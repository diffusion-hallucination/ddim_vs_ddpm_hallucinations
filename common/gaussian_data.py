import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common.gaussian_mixture_2d import (
    allocate_mode_counts,
    build_mixture_spec_from_dataset_cfg,
    save_mixture_spec,
)

__all__ = [
    "Gaussian25",
    "GaussianMixture2D",
    "DataStreamer",
    "GenToyDataset",
    "GaussianFPS25D",
]


def load_gaussian_data(
    dataset_name,
    batch_size,
    num_samples,
    num_dims,
    modes=None,
    dataset_cfg=None,
    mixture_spec_path=None,
    resample=False,
    sample_seed=None,
):
    # build the synthetic gaussian dataloader used in the paper's training runs.
    num_batches = int(math.ceil(float(num_samples) / float(batch_size)))
    datastreamer = DataStreamer(
        dataset=dataset_name,
        batch_size=batch_size,
        num_batches=num_batches,
        num_samples=num_samples,
        num_dims=num_dims,
        modes=modes, 
        dataset_cfg=dataset_cfg,
        mixture_spec_path=mixture_spec_path,
        resample=resample,
        sample_seed=sample_seed,
    )
    
    return datastreamer

class ToyDataset(Dataset):
    def __init__(self, size: int, stdev: float, random_state: int = None):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.size = size
        self.noise = stdev
        self.random_state = random_state
        self.stdev = self.calc_stdev()
        self.data = self.sample()
        
    def calc_stdev(self):
        # calc stdev.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        pass

    def sample(self):
        # sample.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        pass

    def resample(self):
        # resample.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.data = self.sample()

    def __len__(self):
        # len.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return self.size

    def __getitem__(self, idx):
        # getitem.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return torch.from_numpy(self.data[idx])

class GenToyDataset(Dataset):
    def __init__(self, data):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.size = len(data)
        self.data = data

    def __len__(self):
        # len.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return self.size

    def __getitem__(self, idx):
        # getitem.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return torch.from_numpy(self.data[idx])

'''
Gaussian Farthest Point Sampling 
'''
class GaussianFPS25D(ToyDataset):
    """
    Pick exactly `num_modes` modes from the D-dim 5^D lattice using farthest-point sampling (greedy max-min),
    then sample isotropic Gaussians around those modes.

    Lattice coords are in {-2,-1,0,1,2}^D, scaled by `scale`.
    """
    scale = 2.0

    def __init__(
        self,
        size,
        d,
        stdev=0.05,
        noise=None,
        random_state=1234,
        fps_seed=None,
        num_modes=25,
        candidate_pool=None,      # None => exact enumerate if feasible, else random pool
        start="corner",           # "corner" or "random"
        chunk_size=20000,         # for distance eval
        max_enum_points=500_000,  # enumerate full lattice only if 5^d <= this
        max_candidate_bytes=256 * 1024 * 1024,  # cap candidate pool memory for large d
        shuffle_modes=False,      # if True, randomize mode assignment order
        min_normalized_pair_distance_floor=None,
        normalization_mode="fixed_2sqrtd",
    ):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        assert d >= 1
        assert num_modes >= 1
        self.d = int(d)
        self.sample_seed = None if random_state is None else int(random_state)
        self.fps_seed = self.sample_seed if fps_seed is None else int(fps_seed)
        self.resample_index = 0
        self.noise = float(stdev if noise is None else noise)
        self.num_modes = int(num_modes)
        self.chunk_size = int(chunk_size)
        self.shuffle_modes = bool(shuffle_modes)
        self.start = str(start)
        self.normalization_mode = str(normalization_mode).strip().lower()
        self.min_normalized_pair_distance_floor = (
            None if min_normalized_pair_distance_floor is None else float(min_normalized_pair_distance_floor)
        )

        rng = np.random.default_rng(self.fps_seed)

        # ------------------------------------------------------------
        # Build candidate set (exact if small enough, else random pool)
        # ------------------------------------------------------------
        lattice_size = 5 ** self.d
        exact_enumeration = False
        if candidate_pool is None:
            if lattice_size <= max_enum_points:
                candidates = self.enumerate_lattice(self.d)  # (5^d, d) int8
                exact_enumeration = True
            else:
                max_pool_by_mem = max(1_000, int(max_candidate_bytes / (4 * self.d)))
                candidate_pool = min(
                    max(50_000, 10_000 * self.d),
                    500_000,
                    max_pool_by_mem,
                )
                candidates = self.random_lattice_samples(rng, self.d, candidate_pool)
        else:
            candidates = self.random_lattice_samples(rng, self.d, int(candidate_pool))

        candidates = (self.scale * candidates.astype(np.float32))  # (M, d)
        self.candidate_pool_size = int(candidates.shape[0])
        self.exact_enumeration = bool(exact_enumeration)

        self.modes = self.farthest_point_sampling(
            candidates=candidates,
            k=self.num_modes,
            rng=rng,
            start=start,
        ).astype(np.float32)  # (num_modes, d)

        super().__init__(size, self.noise, self.sample_seed)
        self.finalize_geometry_metadata()

    # ---------------- ToyDataset hooks ----------------

    def calc_stdev(self):
        # calc stdev.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if self.normalization_mode == "fixed_2sqrtd":
            return float(self.scale) * math.sqrt(float(self.d))
        raise ValueError("gaussian5d normalization_mode must be 'fixed_2sqrtd'")

    def sample(self):
        # sample.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        """
        Returns:
            data: (size, d) float32
        Sampling matches your original pattern:
          data = noise * N(0, I) + mode_assigned
          data /= stdev
        """
        rng = self.sample_rng()

        # base Gaussian noise in d dims
        data = self.noise * rng.standard_normal((self.size, self.d), dtype=np.float32)

        # choose a mode per sample
        if self.shuffle_modes:
            mode_idx = rng.integers(0, self.modes.shape[0], size=(self.size,), endpoint=False)
        else:
            # deterministic cycling like your original code
            mode_idx = np.arange(self.size) % self.modes.shape[0]

        data += self.modes[mode_idx]

        # normalize like original
        data /= self.stdev
        return data

    def sample_rng(self):
        # sample rng.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if self.sample_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.sample_seed + self.resample_index)

    def resample(self):
        # resample.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.resample_index += 1
        self.data = self.sample()

    def finalize_geometry_metadata(self):
        # finalize geometry metadata.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        normalized_modes = self.modes / self.stdev
        diff = normalized_modes[:, None, :] - normalized_modes[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_distance = float(np.min(dists))
        if (
            self.min_normalized_pair_distance_floor is not None
            and min_pair_distance < self.min_normalized_pair_distance_floor
        ):
            raise ValueError(
                "Curated Gaussian FPS modes violated the normalized spacing floor: "
                f"{min_pair_distance:.6f} < {self.min_normalized_pair_distance_floor:.6f}"
            )
        self.fps_metadata = {
            "num_dims": int(self.d),
            "num_modes": int(self.num_modes),
            "fps_seed": None if self.fps_seed is None else int(self.fps_seed),
            "sample_seed": None if self.sample_seed is None else int(self.sample_seed),
            "candidate_pool_size": int(self.candidate_pool_size),
            "exact_enumeration": bool(self.exact_enumeration),
            "start_policy": self.start,
            "normalization_mode": str(self.normalization_mode),
            "normalization_scale": float(self.stdev),
            "raw_noise_std": float(self.noise),
            "min_normalized_pair_distance": min_pair_distance,
            "min_normalized_pair_distance_floor": self.min_normalized_pair_distance_floor,
        }
        sigma_normalized = float(self.noise / self.stdev)
        normalized_modes = np.asarray(normalized_modes, dtype=np.float64)
        raw_modes = np.asarray(self.modes, dtype=np.float64)
        mode_weights = np.full((self.num_modes,), 1.0 / float(self.num_modes), dtype=np.float64)
        normalized_sigmas = np.full((self.num_modes,), sigma_normalized, dtype=np.float64)
        raw_sigmas = np.full((self.num_modes,), float(self.noise), dtype=np.float64)
        self.mixture_spec = {
            "spec_version": "gaussian_fps_d/v1",
            "dataset_name": "gaussian5d",
            "kind": "gaussian_fps_d",
            "num_dims": int(self.d),
            "num_modes": int(self.num_modes),
            "normalization_mode": str(self.normalization_mode),
            "normalization_scale": float(self.stdev),
            "raw_means": raw_modes.tolist(),
            "raw_sigmas": raw_sigmas.tolist(),
            "normalized_means": normalized_modes.tolist(),
            "normalized_sigmas": normalized_sigmas.tolist(),
            "mode_weights": mode_weights.tolist(),
            "global_sigma_raw": float(self.noise),
            "global_sigma_normalized": sigma_normalized,
            "fps_metadata": dict(self.fps_metadata),
        }

    # ---------------- FPS utilities ----------------

    @staticmethod
    def enumerate_lattice(d: int) -> np.ndarray:
        # enumerate lattice.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        """
        Enumerate all points in {-2,-1,0,1,2}^d.
        Returns int8 array of shape (5^d, d).
        """
        vals = np.array([-2, -1, 0, 1, 2], dtype=np.int8)

        grids = np.meshgrid(*([vals] * d), indexing="ij")
        lattice = np.stack(grids, axis=-1).reshape(-1, d)

        return lattice


    @staticmethod
    def random_lattice_samples(rng: np.random.Generator, d: int, n: int) -> np.ndarray:
        # random lattice samples.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        """
        Sample n points uniformly from {-2,-1,0,1,2}^d (with replacement).
        Returns int8 array of shape (n, d).
        """
        vals = np.array([-2, -1, 0, 1, 2], dtype=np.int8)
        idx = rng.integers(0, 5, size=(n, d), endpoint=False)
        return vals[idx]

    def farthest_point_sampling(
        self,
        candidates: np.ndarray,  # (M, d) float32
        k: int,
        rng: np.random.Generator,
        start: str = "corner",
    ) -> np.ndarray:
        # farthest point sampling.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        """
        Greedy max-min FPS using squared Euclidean distances.
        """
        M, _ = candidates.shape
        if k >= M:
            return candidates.copy()

        # initial point
        if start == "random":
            first = int(rng.integers(0, M))
        else:
            # "corner": max L2 norm
            first = int(np.argmax(np.sum(candidates * candidates, axis=1)))

        selected_idx = np.empty((k,), dtype=np.int64)
        selected_idx[0] = first

        # min_dist2[i] = min_j ||candidates[i] - selected[j]||^2
        min_dist2 = np.full((M,), np.inf, dtype=np.float32)

        # update with first center
        self.update_min_dist2(min_dist2, candidates, candidates[first], self.chunk_size)

        for t in range(1, k):
            nxt = int(np.argmax(min_dist2))
            selected_idx[t] = nxt
            self.update_min_dist2(min_dist2, candidates, candidates[nxt], self.chunk_size)
            min_dist2[nxt] = -np.inf  # avoid reselection

        return candidates[selected_idx]

    @staticmethod
    def update_min_dist2(min_dist2, candidates, center, chunk_size: int):
        # update min dist2.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        """
        min_dist2 = min(min_dist2, ||candidates - center||^2) computed in chunks.
        """
        M = candidates.shape[0]
        for s in range(0, M, chunk_size):
            e = min(s + chunk_size, M)
            diff = candidates[s:e] - center
            dist2 = np.sum(diff * diff, axis=1, dtype=np.float32)
            min_dist2[s:e] = np.minimum(min_dist2[s:e], dist2)

class GaussianMixture2D(ToyDataset):
    def __init__(self, size, mixture_spec, random_state=1234):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        self.mixture_spec = dict(mixture_spec)
        self.modes = np.asarray(self.mixture_spec["raw_means"], dtype=np.float32)
        self.mode_weights = np.asarray(self.mixture_spec["mode_weights"], dtype=np.float64)
        self.raw_sigmas = np.asarray(self.mixture_spec["raw_sigmas"], dtype=np.float64)
        self.normalized_modes = np.asarray(self.mixture_spec["normalized_means"], dtype=np.float32)
        self.normalized_sigmas = np.asarray(self.mixture_spec["normalized_sigmas"], dtype=np.float32)
        super().__init__(size, stdev=float(self.mixture_spec["global_sigma_raw"]), random_state=random_state)

    def calc_stdev(self):
        # calc stdev.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return float(self.mixture_spec["normalization_scale"])

    def sample(self):
        # sample.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        rng = np.random.default_rng(self.random_state)
        mode_counts = allocate_mode_counts(self.size, self.mode_weights)
        mode_idx = np.repeat(np.arange(self.modes.shape[0], dtype=np.int64), mode_counts)
        if mode_idx.shape[0] != self.size:
            raise ValueError("Mode count allocation did not match dataset size")
        rng.shuffle(mode_idx)
        noise = rng.standard_normal((self.size, 2), dtype=np.float32)
        data = self.modes[mode_idx] + self.raw_sigmas[mode_idx, None].astype(np.float32) * noise
        data /= np.float32(self.stdev)
        return data.astype(np.float32, copy=False)


class Gaussian25(GaussianMixture2D):
    def __init__(self, size, stdev=0.05, random_state=1234):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        spec = build_mixture_spec_from_dataset_cfg({
            "name": "gaussian25",
            "kind": "gaussian_mixture_2d",
            "base_layout": "grid25",
            "mode_spacing": 2.0,
            "global_sigma": float(stdev),
        })
        super().__init__(size=size, mixture_spec=spec, random_state=random_state)


class DataStreamer:

    # Data Streamer # 
    def __init__(
        self,
        dataset: ToyDataset,
        batch_size: int,
        num_batches: int,
        resample: bool = False,
        num_dims=None,
        modes=None,
        dataset_cfg=None,
        mixture_spec_path=None,
        num_samples=None,
        sample_seed=None,
    ):
        # init.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if isinstance(dataset, str):
            dataset_name = dataset
            dataset_size = int(num_samples) if num_samples is not None else int(batch_size * num_batches)
            if dataset_cfg is not None and str(dataset_cfg.get("kind", "")).lower() == "gaussian_mixture_2d":
                mixture_spec = build_mixture_spec_from_dataset_cfg(dataset_cfg)
                self.dataset = GaussianMixture2D(dataset_size, mixture_spec=mixture_spec, random_state=sample_seed)
                if mixture_spec_path is not None:
                    self.save_mixture_spec(mixture_spec_path)
            else:
                dataset = self.dataset_map(dataset)
                if dataset is None:
                    raise ValueError(f"Unsupported Gaussian dataset {dataset_name!r}")
                if dataset_name == "gaussian5d":
                    assert num_dims is not None, "Num Dims must be provided for 5D gaussian"
                    self.dataset = dataset(
                        dataset_size,
                        d=num_dims,
                        stdev=float(dataset_cfg.get("stdev", 0.02)) if dataset_cfg is not None else 0.02,
                        random_state=sample_seed,
                        fps_seed=(None if dataset_cfg is None else dataset_cfg.get("fps_seed", None)),
                        min_normalized_pair_distance_floor=(
                            None if dataset_cfg is None else dataset_cfg.get("min_normalized_pair_distance_floor", None)
                        ),
                        normalization_mode=(
                            "fixed_2sqrtd"
                            if dataset_cfg is None
                            else dataset_cfg.get("normalization_mode", "fixed_2sqrtd")
                        ),
                    )
                    if mixture_spec_path is not None:
                        self.save_mixture_spec(mixture_spec_path)
                else:
                    self.dataset = dataset(dataset_size, random_state=sample_seed)
        else:
            self.dataset = GenToyDataset(dataset)
        
        
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.resample = resample


    def save_mixture_spec(self, mixture_spec_path):
        # save mixture spec.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        if not hasattr(self.dataset, "mixture_spec"):
            return
        save_mixture_spec(self.dataset.mixture_spec, mixture_spec_path)

    def __iter__(self):
        # iter.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        cnt = 0
        while True:
            start = cnt * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            yield torch.from_numpy(self.dataset.data[start:end])
            cnt += 1
            if cnt >= self.num_batches:
                break
        if self.resample:
            self.dataset.resample()

    def __len__(self):
        # len.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return self.num_batches
        
    @staticmethod
    def dataset_map(dataset):
        # dataset map.
        # this supports the shared training, model, and diffusion utilities used throughout the repo.
        return {
            "gaussian5d": GaussianFPS25D,
            "gaussian25": Gaussian25,
            "gaussian_mixture_2d": GaussianMixture2D,
        }.get(dataset, None)
