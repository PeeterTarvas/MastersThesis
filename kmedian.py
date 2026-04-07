import numpy as np
from typing import Optional



def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two points."""
    return float(np.sum(np.abs(a - b)))


def pairwise_l1(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    compute pairwise L1 distances between all points in x and all centers.
    Returns shape (n, k).
    """
    return np.sum(np.abs(x[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis=2)


def assignment_cost(
        x: np.ndarray, centers: np.ndarray, _weights: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """assign each point to its nearest center and return total L1 cost."""
    if _weights is None:
        _weights = np.ones(len(x))
    D = pairwise_l1(x, centers)
    labels = np.argmin(D, axis=1)
    min_dists = D[np.arange(len(x)), labels]
    cost = float(np.dot(_weights, min_dists))
    return labels, cost


def kmedian_plus_plus_seed(
        x: np.ndarray,
        k: int,
        rng: np.random.Generator,
        _weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    probabilistic seeding: choose first center uniformly at random, then
    each subsequent center with probability proportional to its L1 distance
    to the nearest already-chosen center.
    """
    if _weights is None:
        _weights = np.ones(len(x))

    n = len(x)
    chosen_indices = []

    probs = _weights / _weights.sum()
    idx = int(rng.choice(n, p=probs))
    chosen_indices.append(idx)

    for _ in range(1, k):
        current_centers = x[chosen_indices]
        d = pairwise_l1(x, current_centers)
        min_dists = d.min(axis=1)
        weighted_dists = _weights * min_dists
        total = weighted_dists.sum()
        if total == 0:
            probs = np.ones(n) / n
        else:
            probs = weighted_dists / total
        idx = int(rng.choice(n, p=probs))
        chosen_indices.append(idx)

    return x[chosen_indices].copy()


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_v = values[order]
    cum_w = np.cumsum(weights[order])
    half = cum_w[-1] / 2.0
    idx = np.searchsorted(cum_w, half)
    return float(sorted_v[min(idx, len(sorted_v) - 1)])


def local_search_kmedian(
        x: np.ndarray,
        k: int,
        _weights: np.ndarray,
        init_centers: np.ndarray,
        max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    lloyd's iteration for weighted k-median under L1 distance.
    each centre is relocated to the coordinate-wise weighted median of
    its cluster

    """
    n, d = x.shape
    centers = init_centers.copy()

    for _it in range(max_iter):
        D = pairwise_l1(x, centers)
        labels = np.argmin(D, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                continue
            cX = x[mask]
            cW = _weights[mask]
            for dim in range(d):
                new_centers[j, dim] = _weighted_median_1d(cX[:, dim], cW)

        if np.allclose(centers, new_centers, atol=1e-12):
            break
        centers = new_centers

    D = pairwise_l1(x, centers)
    labels = np.argmin(D, axis=1)
    cost = float(np.dot(_weights, D[np.arange(n), labels]))

    return centers, labels, cost


def kmedian(
        x: np.ndarray,
        k: int,
        _weights: Optional[np.ndarray] = None,
        n_trials: int = 5,
        max_iter: int = 10,
        random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    k-median clustering via k-median++ seeding + Lloyd's iteration.
    n_trials independent restarts and returns the best result.
    """
    x = np.asarray(x, dtype=float)

    if _weights is not None:
        _weights = np.asarray(_weights, dtype=float)
    else:
        _weights = np.ones(len(x))

    rng = np.random.default_rng(random_seed)

    best_centers, best_labels, best_cost = None, None, np.inf

    for trial in range(n_trials):
        init_centers = kmedian_plus_plus_seed(x, k, rng, _weights)
        centers, labels, cost = local_search_kmedian(
            x, k, _weights, init_centers, max_iter
        )
        if cost < best_cost:
            best_centers = centers
            best_labels = labels
            best_cost = cost

    return best_centers, best_labels, best_cost