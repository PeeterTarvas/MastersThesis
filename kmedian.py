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

import matplotlib.pyplot as plt
import pandas as pd
import csv_loader
from coreset import compute_fair_coreset


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


def assignment_cost(X: np.ndarray, centers: np.ndarray, _weights: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """
    Assign each point to its nearest center and return total L1 cost.

    Returns
    -------
    labels : (n,) integer array of center indices
    cost   : sum_i  w_i * d(x_i, assigned center)
    """
    if _weights is None:
        _weights = np.ones(len(X))
    D = pairwise_l1(X, centers)
    labels = np.argmin(D, axis=1)
    min_dists = D[np.arange(len(X)), labels]
    cost = float(np.dot(_weights, min_dists))
    return labels, cost


# ---------------------------------------------------------------------------
# Seeding: k-median++ (D^1 sampling, L1 analog of k-means++)
# ---------------------------------------------------------------------------

def kmedian_plus_plus_seed(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    _weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Probabilistic seeding: choose first center uniformly at random, then
    each subsequent center with probability proportional to its L1 distance
    to the nearest already-chosen center.

    Returns
    -------
    centers : (k, d) array of initial center coordinates (rows of X)
    """
    if _weights is None:
        _weights = np.ones(len(x))

    n = len(x)
    chosen_indices = []

    # First center: uniform random
    probs = _weights / _weights.sum()
    idx = int(rng.choice(n, p=probs))
    chosen_indices.append(idx)

    for _ in range(1, k):
        current_centers = x[chosen_indices]
        d = pairwise_l1(x, current_centers)
        min_dists = d.min(axis=1)
        weighted_dists = _weights * min_dists # w_i * d(x_i, nearest center)
        total = weighted_dists.sum()
        if total == 0:
            probs = np.ones(n) / n  # all remaining points are already at a center; pick uniformly
        else:
            probs = weighted_dists / total
        idx = int(rng.choice(n, p=probs))
        chosen_indices.append(idx)

    return x[chosen_indices].copy()


# ---------------------------------------------------------------------------
# Core: single-swap local search k-median
# ---------------------------------------------------------------------------

def _local_search_kmedian(
    X: np.ndarray,
    k: int,
    _weights: np.ndarray,
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
    labels, cost = assignment_cost(X, centers, _weights)

    # Track which points are currently centers (by index)
    # We work with actual coordinate copies; centers need not be data points
    # but for k-median it is standard to restrict centers to data points.
    center_set = set(map(tuple, centers.tolist()))

    for iteration in range(max_iter):
        best_gain = 0.0
        best_swap = None  # (old_center_idx_in_centers, new_point_idx_in_X)

        for ci in range(k):
            for xi in range(n):
                candidate = X[xi]
                if tuple(candidate.tolist()) in center_set:
                    continue  # already a center

                # Build trial centers with the swap
                trial_centers = centers.copy()
                trial_centers[ci] = candidate
                _, trial_cost = assignment_cost(X, trial_centers, _weights)
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
        labels, cost = assignment_cost(X, centers, _weights)

    return centers, labels, cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kmedian(
    X: np.ndarray,
    k: int,
    _weights: Optional[np.ndarray] = None,
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

    if _weights is not None:
        _weights = np.asarray(_weights, dtype=float)
        assert len(_weights) == len(X), "weights must have same length as X"
        assert np.all(_weights >= 0), "weights must be non-negative"
    else:
        _weights = np.ones(len(X))

    rng = np.random.default_rng(random_seed)

    best_centers, best_labels, best_cost = None, None, np.inf

    for trial in range(n_trials):
        init_centers = kmedian_plus_plus_seed(X, k, rng, _weights)
        centers, labels, cost = _local_search_kmedian(X, k, _weights, init_centers, max_iter)
        if cost < best_cost:
            best_centers = centers
            best_labels = labels
            best_cost = cost

    return best_centers, best_labels, best_cost


if __name__ == "__main__":
    df: pd.DataFrame = csv_loader.load_csv_chunked("us_census_puma_data.csv", csv_loader.LOAD_COLS, csv_loader.LOAD_DTYPES, 100_000, 200_000)

    coreset_df, scaler = compute_fair_coreset(df, n_locations=30000, random_seed=42)

    k = 30
    X = coreset_df[['Lat_Scaled', 'Lon_Scaled']].values
    w = coreset_df['Weight'].values.astype(float)
    centers, labels, cost = kmedian(X, k, w, 10, 100)
    result_df = coreset_df.copy()
    result_df['Cluster'] = labels

    print(f"[Baseline k-median] k={k}, weighted cost = {cost:,.2f}")

    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    scatter = ax.scatter(
        result_df['Longitude'],
        result_df['Latitude'],
        c=result_df['Cluster'],
        cmap='tab20',
        alpha=0.4,
        s=result_df['Weight'] / result_df['Weight'].max() * 10,  # size ∝ weight
        linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.title(f'k-Median Clustering (k={k})', color='black')
    plt.xlabel('Longitude', color='black')
    plt.ylabel('Latitude', color='black')
    plt.tick_params(colors='black')
    plt.tight_layout()
    plt.savefig('kmedian_clusters.png', dpi=150)
    plt.show()




    #rng = np.random.default_rng(0)
    #X = np.vstack([
    #    rng.normal([0, 0], 0.1, (30, 2)),
    #    rng.normal([5, 0], 0.1, (30, 2)),
    #    rng.normal([0, 5], 0.1, (30, 2)),
    #])
    #centers, labels, cost = kmedian(X, k=3, n_trials=5, random_seed=42)
    #print(f"Cost: {cost:.4f}")
    #print(f"Centers:\n{centers}")
    #print(f"Cluster sizes: {np.bincount(labels)}")


    #X_out, weights = compute_fair_coreset(X_norm)
