import numpy as np
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from fast_kmedian import fast_kmedian

import csv_loader
from coreset import compute_fair_coreset

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two points."""
    return float(np.sum(np.abs(a - b)))


def pairwise_l1(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute pairwise L1 distances between all points in X and all centers.
    """
    return np.sum(np.abs(X[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis=2)


def assignment_cost(X: np.ndarray, centers: np.ndarray, _weights: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """
    Assign each point to its nearest center and return total L1 cost.

    """
    if _weights is None:
        _weights = np.ones(len(X))
    D = pairwise_l1(X, centers)
    labels = np.argmin(D, axis=1)
    min_dists = D[np.arange(len(X)), labels]
    cost = float(np.dot(_weights, min_dists))
    return labels, cost

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

def local_search_kmedian(
        X: np.ndarray,
        k: int,
        _weights: np.ndarray,
        init_centers: np.ndarray,
        max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimized single-swap local search for k-median
    """
    n, d = X.shape
    centers = init_centers.copy()

    # Pre-calculate the initial distance matrix (n x k)
    D = pairwise_l1(X, centers)
    labels = np.argmin(D, axis=1)
    min_dists = D[np.arange(n), labels]
    cost = float(np.dot(_weights, min_dists))

    # Fast lookup to prevent swapping a center with itself
    center_set = set(map(tuple, centers.tolist()))

    for iteration in range(max_iter):
        best_gain = 0.0
        best_swap = None

        for ci in range(k):
            # Calculate the distance to the closest center IF center `ci` is removed
            if k == 1:
                min_dists_without_ci = np.full(n, np.inf)
            else:
                D_other = np.delete(D, ci, axis=1)
                min_dists_without_ci = D_other.min(axis=1)

            for xi in range(n):
                candidate = X[xi]
                if tuple(candidate.tolist()) in center_set:
                    continue

                # Instead of computing the full (n x k) matrix, just compute
                # the distance from all points to the *new* candidate center
                dist_to_candidate = np.sum(np.abs(X - candidate), axis=1)

                # The new shortest distance for each point is the minimum of
                # the distance to the new candidate OR the remaining centers
                new_min_dists = np.minimum(min_dists_without_ci, dist_to_candidate)

                trial_cost = float(np.dot(_weights, new_min_dists))
                gain = cost - trial_cost

                if gain > best_gain:
                    best_gain = gain
                    best_swap = (ci, xi)

        if best_swap is None:
            break  # Local optimum reached

        ci, xi = best_swap

        # Update the centers and our tracking sets
        center_set.discard(tuple(centers[ci].tolist()))
        centers[ci] = X[xi].copy()
        center_set.add(tuple(centers[ci].tolist()))

        # Update our running distance matrix and costs
        D[:, ci] = np.sum(np.abs(X - centers[ci]), axis=1)
        labels = np.argmin(D, axis=1)
        min_dists = D[np.arange(n), labels]
        cost = float(np.dot(_weights, min_dists))

    return centers, labels, cost



def kmedian(
    X: np.ndarray,
    k: int,
    _weights: Optional[np.ndarray] = None,
    n_trials: int = 5,
    max_iter: int = 10,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Vanilla k-median clustering via k-median++ seeding + single-swap local search.

    Runs `n_trials` independent restarts and returns the best result.
    """
    X = np.asarray(X, dtype=float)

    if _weights is not None:
        _weights = np.asarray(_weights, dtype=float)
    else:
        _weights = np.ones(len(X))

    rng = np.random.default_rng(random_seed)

    best_centers, best_labels, best_cost = None, None, np.inf

    for trial in range(n_trials):
        init_centers = kmedian_plus_plus_seed(X, k, rng, _weights)
        #centers, labels, cost = fast_kmedian.local_search_kmedian(
        #    X, k,_weights, init_centers, max_iter
        #)
        centers, labels, cost = local_search_kmedian(X, k, _weights, init_centers, max_iter)
        if cost < best_cost:
            best_centers = centers
            best_labels = labels
            best_cost = cost

    return best_centers, best_labels, best_cost


if __name__ == "__main__":
    df: pd.DataFrame = csv_loader.load_csv_chunked("us_census_puma_data.csv",
                                                   csv_loader.LOAD_COLS, csv_loader.LOAD_DTYPES,
                                                   10_0, 10_0)

    coreset_df = compute_fair_coreset(df, n_locations=300, random_seed=42)

    k = 30
    X = coreset_df[['Lat_Scaled', 'Lon_Scaled']].values
    w = coreset_df['Weight'].values.astype(float)
    centers, labels, cost = kmedian(X, k, w)
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
