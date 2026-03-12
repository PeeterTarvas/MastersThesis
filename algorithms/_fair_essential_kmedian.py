import warnings
from typing import Optional

import networkx as nx
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import csv_loader
from coreset import compute_fair_coreset

import numpy as np
import pandas as pd

from kmedian import kmedian, pairwise_l1


def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    """Map arbitrary group labels -> contiguous integers 0..H-1."""
    cats: pd.Categorical = pd.Categorical(group_series)
    cats.codes.astype(dtype=np.int32)
    return cats.codes.astype(dtype=np.int32), list(cats.categories)

def proportional_bounds(
    group_codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive per-group lower / upper bounds from proportional representation
    with ±alpha slack metric.

        l_h = max(0, f_h - alpha)
        u_h = min(1, f_h + alpha)

    where f_h = (total weight of color h) / (total weight).
    """
    total = weights.sum()
    f = np.array([weights[group_codes == h].sum() / total for h in range(n_groups)])
    lower_bound = np.maximum(0.0, f - alpha)
    upper_bound = np.minimum(1.0, f + alpha)
    return lower_bound, upper_bound


def solve_fair_lp(
    x: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    group_codes: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Solve the Fair LP relaxation for *fixed* centers.

    Variables
    ---------
    x_{ij} ∈ [0, 1]  — fractional assignment of point i to center j
    Laid out as a flat vector of length n*k (row-major: x[i*k + j]).

    Objective (Eq. 9)
    -----------------
    min  Σ_{i,j}  w_i · d(i,j) · x_{ij}

    Constraints
    -----------
    (a) Σ_j x_{ij} = 1   ∀i                          (each point fully assigned)
    (b) l_h · Σ_i x_{ij} ≤ Σ_{i∈Col_h} x_{ij}  ∀j,h  (lower fairness)
    (c) Σ_{i∈Col_h} x_{ij} ≤ u_h · Σ_i x_{ij}  ∀j,h  (upper fairness)

    Uses sparse matrices to stay memory-feasible for n up to ~50k.

    Returns
    -------
    x : (n, k) fractional assignment matrix, or None if infeasible.
    """
    n, d = x.shape
    nr_of_centers = len(centers)
    n_vars = n * nr_of_centers
    lower_bound_len = len(lower_bound)

    distances = pairwise_l1(x, centers).astype(np.float64)
    cost_to_center = (distances * weights[:, np.newaxis]).ravel()
    # every
    a_equality_constraint = lil_matrix((n, n_vars), dtype=np.float64)
    for i in range(n):
        a_equality_constraint[i, i * nr_of_centers:(i + 1) * nr_of_centers] = 1.0
    b_equality_constraint = np.ones(n)

    n_ineq = 2 * lower_bound_len * nr_of_centers
    a_upper_bound = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_bound = np.zeros(n_ineq)

    row = 0
    for center_index in range(nr_of_centers):
        for h in range(lower_bound_len):
            in_group = (group_codes == h)
            for i in range(n):
                col = i * nr_of_centers + center_index
                # lower bound row: lower_bound * x_{ij} - (x_{ij} if i∈Col_h)
                a_upper_bound[row, col] = lower_bound[h] - (1.0 if in_group[i] else 0.0)
                # upper bound row: -upper_bound * x_{ij} + (x_{ij} if i∈Col_h)
                a_upper_bound[row + 1, col] = -upper_bound[h] + (1.0 if in_group[i] else 0.0)
            row += 2

    A_eq_csc = a_equality_constraint.tocsc()
    A_ub_csc = a_upper_bound.tocsc()

    result = linprog(
        cost_to_center,
        A_ub=A_ub_csc,
        b_ub=b_bound,
        A_eq=A_eq_csc,
        b_eq=b_equality_constraint,
        bounds=[(0.0, 1.0)] * n_vars,
        method='highs',
        options={'disp': False, 'presolve': True},
    )

    if result.status != 0:
        warnings.warn(f"LP solver returned status {result.status}: {result.message}")
        return None

    return result.x.reshape((n, nr_of_centers))


def min_cost_flow_rounding(
    x_lp: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    n, k = x_lp.shape
    source_node: int = 0
    t = n + k + 1
    point_node  = lambda i: i + 1
    center_node = lambda j: n + j + 1

    min_cost_flow_graph = nx.DiGraph()

    for i in range(n):
        p_node = f"p_{i}"


    return None


def fair_clustering(
        df: pd.DataFrame,
        feature_cols: list,
        protected_group_col: str,
        k_cluster: int,
        alpha: float = 0.1,
        weight_col: Optional[str] = 'Weight',
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        kmedian_trials: int = 3,
        kmedian_max_iter: int = 50,
        random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Algorithm: Essentially Fair k-Median Clustering.

    Parameters
    ----------
    df           : DataFrame. Works with raw points OR a coreset.
    feature_cols : columns used as clustering coordinates (e.g. ['Lat_Scaled','Lon_Scaled'])
    protected_group_col    : column with protected group labels (string or int)
    k_cluster            : number of clusters
    alpha        : proportional-representation slack (ignored if l_h/u_h provided)
    weight_col   : column of point weights.
                   Pass None (or a missing column) to use uniform weights of 1.
    lower_bound, upper_bound     : explicit per-group bounds (length H arrays). If None, derived
                   from alpha and proportional representation.
    kmedian_*    : passed through to kmedian()

    Returns
    -------
    centers : (k, d) final center coordinates
    labels  : (n,) integer cluster assignment per row of df
    cost    : total weighted L1 assignment cost
    """

    x = df[feature_cols].to_numpy(dtype=np.float64)

    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(x), dtype=np.float64)


    group_codes, group_labels = encode_groups_to_int(df[protected_group_col])

    nr_of_groups = len(group_labels)

    print(f"[FairClustering] n={len(x):,}  k={k_cluster}  groups={nr_of_groups}  "
          f"weighted={'yes' if weight_col and weight_col in df.columns else 'no (uniform)'}")

    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = proportional_bounds(group_codes, weights, nr_of_groups, alpha)
        print(f"[FairClustering] Proportional bounds (alpha={alpha}):")
        for h, lbl in enumerate(group_labels):
            print(f"  {lbl}: [{lower_bound[h]:.3f}, {upper_bound[h]:.3f}]")

    total = weights.sum()
    f = np.array([weights[group_codes == h].sum() / total for h in range(nr_of_groups)])
    infeasible_groups = np.where((f < lower_bound - 1e-6) | (f > upper_bound + 1e-6))[0]
    if len(infeasible_groups) > 0:
        for h in infeasible_groups:
            warnings.warn(
                f"Group '{group_labels[h]}' proportion {f[h]:.3f} is outside "
                f"[{upper_bound[h]:.3f}, {lower_bound[h]:.3f}] — LP will be infeasible. "
                "Increase alpha or adjust bounds.")

    print("k-median for centering")
    centers, _, unfair_cost = kmedian(
        x, k_cluster, _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed
    )
    print(f"Unfair k-median cost: {unfair_cost:,.2f}")
    print("Solving Fair LP...")
    x_lp = solve_fair_lp(x, centers, weights, group_codes, lower_bound, upper_bound)

    if x_lp is None:
        warnings.warn("LP failed — returning unfair k-median assignment.")
        d = pairwise_l1(x, centers)
        labels = np.argmin(d, axis=1).astype(np.int32)
        cost = float(np.dot(weights, d[np.arange(len(x)), labels]))
        return centers, labels, cost

    print(f"LP fractional cost: {float(np.dot(weights, (pairwise_l1(x, centers) * x_lp).sum(axis=1))):,.2f}")
    # --- Step 3: MCF rounding ---
    print("Min-cost flow rounding...")
    distances_to_centers = pairwise_l1(x, centers)
    labels = mcf_rounding(x_lp, group_codes, weights, distances_to_centers)
    G.add_edge(S, p_node, capacity=w_int[i], weight=0)

    return None


if __name__ == "__main__":
 df = csv_loader.load_csv_chunked(
        "us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=10_000,
        max_rows=10_000,
    )
 coreset_df = compute_fair_coreset(df, n_locations=300, random_seed=42)
 centers_c, labels_c, cost_c = fair_clustering(
     coreset_df,
     feature_cols=['Lat_Scaled', 'Lon_Scaled'],
     protected_group_col='GROUP_ID',
     k_cluster=10,
     alpha=0.15,
     weight_col='Weight',
 )
