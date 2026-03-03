"""
k-Median Clustering
====================
Standalone implementation of vanilla k-median clustering.
Used as a baseline and as the first step in fair clustering.

Algorithm: Single-swap local search (5-approximation, Arya et al. 2004),
           seeded with k-median++ (adapted D^2 sampling for L1).
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two points."""
    return float(np.sum(np.abs(a - b)))


def pairwise_l1(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute pairwise L1 distances between all points in X and all centers.

    Parameters
    ----------
    X       : (n, d) array of data points
    centers : (k, d) array of centers

    Returns
    -------
    D : (n, k) distance matrix
    """
    # Expand dims for broadcasting: (n,1,d) - (1,k,d) -> (n,k,d)
    return np.sum(np.abs(X[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis=2)


def assignment_cost(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Assign each point to its nearest center and return total L1 cost.

    Returns
    -------
    labels : (n,) integer array of center indices
    cost   : total assignment cost (sum of L1 distances)
    """
    D = pairwise_l1(X, centers)
    labels = np.argmin(D, axis=1)
    cost = float(np.sum(D[np.arange(len(X)), labels]))
    return labels, cost


# ---------------------------------------------------------------------------
# Seeding: k-median++ (D^1 sampling, L1 analog of k-means++)
# ---------------------------------------------------------------------------

def kmedian_plus_plus_seed(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Probabilistic seeding: choose first center uniformly at random, then
    each subsequent center with probability proportional to its L1 distance
    to the nearest already-chosen center.

    Returns
    -------
    centers : (k, d) array of initial center coordinates (rows of X)
    """
    n = len(X)
    chosen_indices = []

    # First center: uniform random
    idx = int(rng.integers(0, n))
    chosen_indices.append(idx)

    for _ in range(1, k):
        current_centers = X[chosen_indices]
        D = pairwise_l1(X, current_centers)
        min_dists = D.min(axis=1)
        probs = min_dists / min_dists.sum()
        idx = int(rng.choice(n, p=probs))
        chosen_indices.append(idx)

    return X[chosen_indices].copy()


# ---------------------------------------------------------------------------
# Core: single-swap local search k-median
# ---------------------------------------------------------------------------

def _local_search_kmedian(
    X: np.ndarray,
    k: int,
    init_centers: np.ndarray,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Single-swap local search for k-median (Arya et al. 2004 style).

    At each iteration, try swapping each current center with each non-center
    point. Accept the swap that gives the greatest cost reduction.
    Stops when no improving swap exists or max_iter is reached.

    Returns
    -------
    centers : (k, d) final centers
    labels  : (n,) assignment of each point to a center index
    cost    : total L1 cost
    """
    n, d = X.shape
    centers = init_centers.copy()
    labels, cost = assignment_cost(X, centers)

    # Track which points are currently centers (by index)
    # We work with actual coordinate copies; centers need not be data points
    # but for k-median it is standard to restrict centers to data points.
    center_set = set(map(tuple, centers.tolist()))

    for iteration in range(max_iter):
        best_gain = 0.0
        best_swap = None  # (old_center_idx_in_centers, new_point_idx_in_X)

        for ci in range(k):
            old_center = centers[ci]
            for xi in range(n):
                candidate = X[xi]
                if tuple(candidate.tolist()) in center_set:
                    continue  # already a center

                # Build trial centers with the swap
                trial_centers = centers.copy()
                trial_centers[ci] = candidate
                _, trial_cost = assignment_cost(X, trial_centers)
                gain = cost - trial_cost
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (ci, xi)

        if best_swap is None:
            break  # local optimum

        ci, xi = best_swap
        center_set.discard(tuple(centers[ci].tolist()))
        centers[ci] = X[xi].copy()
        center_set.add(tuple(centers[ci].tolist()))
        labels, cost = assignment_cost(X, centers)

    return centers, labels, cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kmedian(
    X: np.ndarray,
    k: int,
    n_trials: int = 5,
    max_iter: int = 100,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Vanilla k-median clustering via k-median++ seeding + single-swap local search.

    Runs `n_trials` independent restarts and returns the best result.

    Parameters
    ----------
    X           : (n, d) float array of data points
    k           : number of clusters
    n_trials    : number of random restarts (best result is returned)
    max_iter    : maximum local-search iterations per trial
    random_seed : optional seed for reproducibility

    Returns
    -------
    centers : (k, d) array of final center coordinates
    labels  : (n,) integer array; labels[i] = index of nearest center for point i
    cost    : total L1 assignment cost (sum of distances to assigned center)

    Notes
    -----
    Distance metric: Manhattan (L1), consistent with the formal problem definition
    in the thesis (metric space with L1 distance).
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_seed)

    best_centers, best_labels, best_cost = None, None, np.inf

    for trial in range(n_trials):
        init_centers = kmedian_plus_plus_seed(X, k, rng)
        centers, labels, cost = _local_search_kmedian(X, k, init_centers, max_iter)
        if cost < best_cost:
            best_centers = centers
            best_labels = labels
            best_cost = cost

    return best_centers, best_labels, best_cost


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Three well-separated blobs
    X = np.vstack([
        rng.normal([0, 0], 0.1, (30, 2)),
        rng.normal([5, 0], 0.1, (30, 2)),
        rng.normal([0, 5], 0.1, (30, 2)),
    ])
    centers, labels, cost = kmedian(X, k=3, n_trials=5, random_seed=42)
    print(f"Cost: {cost:.4f}")
    print(f"Centers:\n{centers}")
    print(f"Cluster sizes: {np.bincount(labels)}")