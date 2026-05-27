import numpy as np


def find_all_closest_mode_pairs(modes: np.ndarray, tol: float = 1e-8):
    # return every mode pair that attains the closest pair distance.
    # experiments e, i, and l all use this to define the nearest pair geometry.
    modes = np.asarray(modes, dtype=np.float64)
    if modes.ndim != 2:
        raise ValueError(f"Expected modes shape (K, D), got {tuple(modes.shape)}")
    num_modes = int(modes.shape[0])
    if num_modes < 2:
        return [], float("nan")

    # build the full pairwise distance table in normalized data coordinates.
    diffs = modes[:, None, :] - modes[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)

    # the closest pair distance \ell is the minimum off-diagonal entry.
    min_dist = float(np.min(dists))
    if not np.isfinite(min_dist):
        return [], float("nan")

    # keep every unordered pair whose distance matches \ell up to numerical tolerance.
    ii, jj = np.where(np.abs(dists - min_dist) <= float(tol))
    pairs = {
        tuple(sorted((int(i), int(j))))
        for i, j in zip(ii.tolist(), jj.tolist())
        if int(i) != int(j)
    }
    return sorted(pairs), min_dist


def build_pair_geometry_table(modes: np.ndarray, pair_list=None, tol: float = 1e-8) -> dict:
    # build midpoint, direction, and pair-length tables for a collection of mode pairs.
    # this is the common pair-axis geometry used throughout the gaussian experiments.
    modes = np.asarray(modes, dtype=np.float64)
    if pair_list is None:
        pair_list, closest_pair_distance = find_all_closest_mode_pairs(modes, tol=tol)
    else:
        pair_list = [tuple(map(int, pair)) for pair in pair_list]
        closest_pair_distance = float("nan")
    if len(pair_list) == 0:
        raise ValueError("No Gaussian mode pairs were supplied.")

    pair_i = np.asarray([pair[0] for pair in pair_list], dtype=np.int64)
    pair_j = np.asarray([pair[1] for pair in pair_list], dtype=np.int64)
    mu_i = modes[pair_i]
    mu_j = modes[pair_j]

    # \ell is the pair distance in the normalized data coordinates used by the repo.
    ell = np.linalg.norm(mu_j - mu_i, axis=1)
    if np.any(~np.isfinite(ell)) or np.any(ell <= 0.0):
        raise ValueError("Encountered a degenerate mode pair with zero pair distance.")

    # u is the unit direction from mode i to mode j, and m is the midpoint.
    u = (mu_j - mu_i) / ell[:, None]
    midpoint = 0.5 * (mu_i + mu_j)
    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "mu_i": mu_i,
        "mu_j": mu_j,
        "ell": ell,
        "u": u,
        "midpoint": midpoint,
        "pair_list": pair_list,
        "closest_pair_distance": closest_pair_distance,
    }
