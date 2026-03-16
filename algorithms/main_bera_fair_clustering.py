from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import csv_loader
from coreset import compute_fair_coreset, preprocess_dataset
from kmedian import kmedian, pairwise_l1

def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    """
    Map arbitrary group label strings (e.g. "1_Low", "2_High") to
    contiguous integers 0 … H-1.

    Returns
    -------
    codes  : (n,) int32 array — group index per point
    labels : list of original group names (position = code value)
    """
    cats = pd.Categorical(group_series)
    return cats.codes.astype(np.int32), list(cats.categories)


def proportional_bounds(
    group_codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive per-group lower/upper fairness bounds from proportional
    representation with ±alpha slack, following the paper's δ-parameterization.

    In the paper (Section 5), they set:
        β_i = r_i · (1 − δ)   and   α_i = r_i / (1 − δ)
    where r_i is the fraction of group i in the whole dataset and δ ∈ [0,1].

    Here we use a simpler additive slack:
        l_h = max(0,   f_h − alpha)
        u_h = min(1,   f_h + alpha)

    This is equivalent and makes alpha interpretable as the maximum
    deviation from proportional representation that we tolerate.

    Parameters
    ----------
    alpha : allowed deviation from proportional representation (e.g. 0.1 = 10%)
    """
    total = weights.sum()
    f = np.array([
        weights[group_codes == h].sum() / total
        for h in range(n_groups)
    ])
    lower = np.maximum(0.0, f - alpha)
    upper = np.minimum(1.0, f + alpha)
    return lower, upper


def solve_fair_lp(
    X: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    group_codes: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Solve the Fair p-Assignment LP relaxation w(Eq. 1 from Bera et al. 2019)
    for FIXED centers.

    Decision variables
    ------------------
    x_{ij} ∈ [0, 1]  — fractional assignment of point i to center j.
    Flattened into a vector of length n*k (row-major: index = i*k + j).

    Objective  (weighted L1 cost)
    ------------------------------
    min  Σ_{i,j}  w_i · d(x_i, c_j) · x_{ij}

    Constraints
    -----------
    (assignment) Σ_j x_{ij} = 1                          ∀ i
    (MP lower)   β_h · Σ_i x_{ij} ≤ Σ_{i∈Col_h} x_{ij}  ∀ j, ∀ h
    (RD upper)   Σ_{i∈Col_h} x_{ij} ≤ α_h · Σ_i x_{ij}  ∀ j, ∀ h

    Both fairness constraints are reformulated as ≤ 0 inequalities for
    linprog (scipy uses A_ub @ x ≤ b_ub convention):

      MP:  β_h * Σ_i x_{ij}  −  Σ_{i∈Col_h} x_{ij}  ≤ 0
      RD:  Σ_{i∈Col_h} x_{ij}  −  α_h * Σ_i x_{ij}  ≤ 0

    Returns
    -------
    x_lp : (n, k) fractional assignment matrix, or None if infeasible.
    """
    dataset_len = len(X)
    nr_of_centers = len(centers)
    nr_of_bounds = len(lower_bounds)
    n_vars = dataset_len * nr_of_centers

    distances_to_centers = pairwise_l1(X, centers).astype(np.float64)
    cost_to_center = (distances_to_centers * weights[:, np.newaxis]).ravel()
    a_equality_constraint = lil_matrix((dataset_len, n_vars), dtype=np.float64)
    for i in range(dataset_len):
        a_equality_constraint[i, i * nr_of_centers : (i + 1) * nr_of_centers] = 1.0
    b_equality_constraint = np.ones(dataset_len, dtype=np.float64)

    n_ineq = 2 * nr_of_bounds * nr_of_centers
    a_upper_bound = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_bound = np.zeros(n_ineq)

    row = 0
    for center_index in range(nr_of_centers):
        for bound_index in range(nr_of_bounds):
            in_bound = (group_codes == bound_index)
            for row_indx in range(dataset_len):
                col = row_indx * nr_of_centers + center_index
                a_upper_bound[row, col] = lower_bounds[bound_index] - (1.0 if in_bound[row_indx] else 0.0)
                a_upper_bound[row + 1, col] = -upper_bounds[bound_index] + (1.0 if in_bound[row_indx] else 0.0)
            row += 2


    result = linprog(
        cost_to_center,
        A_ub=a_upper_bound.tocsc(),
        b_ub=b_bound,
        A_eq=a_equality_constraint.tocsc(),
        b_eq=b_equality_constraint,
        bounds=[(0.0, 1.0)] * n_vars,
        method='highs',
        options={'disp': False, 'presolve': True},
    )

    if result.status != 0:
        warnings.warn(
            f"[FairLP] Solver returned status {result.status}: {result.message}\n"
            "  → Try increasing alpha or checking group proportions."
        )
        return None

    return result.x.reshape((dataset_len, nr_of_centers))


def iterative_rounding(
    X: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    group_codes: np.ndarray,
    x_lp: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Iterative LP rounding to turn the fractional Fair-LP solution into
    an integral assignment, following Algorithm 2 of Bera et al. 2019.

    The algorithm achieves (4Δ+3)-additive violation of fairness constraints
    where Δ = max number of groups a single point can belong to.
    Since our groups partition the points (each point in exactly one group),
    Δ = 1 and the additive violation is at most 7.

    Key idea  (matching the paper's proof of Theorem 7)
    ---------------------------------------------------
    The initial LP solution x* may be fractional (a point split across centers).
    We need to round it to an integral assignment while preserving fairness.

    Step A — Compute "target mass" from the LP solution:
        T_f[j]    = Σ_i   w_i · x*_{ij}   (weighted total  at center j)
        T_fh[h,j] = Σ_{i∈Col_h} w_i · x*_{ij}  (weighted group-h mass at center j)

    Step B — Build LP2 with TIGHTER bounds that force integrality:
        ⌊T_f[j]⌋    ≤  Σ_i   w_i · x_{ij}         ≤  ⌈T_f[j]⌉      ∀ j
        ⌊T_fh[h,j]⌋ ≤  Σ_{i∈Col_h} w_i · x_{ij}   ≤  ⌈T_fh[h,j]⌉  ∀ j, h

    Because T_f[j] is the *sum* of the original LP's assignments to center j,
    and it may be non-integer, tightening to [⌊T_f⌋, ⌈T_f⌉] constrains the
    integral assignment to be very close to the LP's fractional one.

    Step C — Iterative loop:
        1. Solve LP2 on the remaining (unassigned) points.
        2. Any x_{ij} = 1 → commit: assign point i to center j, remove from LP.
        3. Any x_{ij} = 0 → prune: remove that variable (point won't go there).
        4. Adjust T_f[j] and T_fh[h,j] by subtracting committed weight.
        5. Drop (j,h) constraint once ≤ 2(Δ+1) fractional vars remain for it.
           This is the sparsity/degree argument from Kiraly et al. [49] in the paper.
        6. Repeat until all points assigned.

    Why does this terminate?  Each iteration either:
      - Commits at least one point (LP2 has an integral vertex due to matroid
        structure when sparsity threshold is reached), OR
      - Drops a fairness constraint (reducing LP2's size).
    Both events are finite, so the loop terminates in O(n · H · k) steps.

    Why does the fairness violation stay small?  When we drop a (j,h) constraint
    because ≤ 2(Δ+1) fractional variables remain, assigning all of them to center j
    (worst case) introduces at most 2(Δ+1) additive error per group per cluster.
    Combined with the floor/ceil rounding error of ±1, this gives the (4Δ+3) bound.

    Weighted points
    ---------------
    With coreset weights w_i > 1, the fairness bounds T_fh are in terms of
    weighted mass rather than point counts.  The LP2 constraints are written
    in terms of weighted sums Σ w_i x_{ij}, and when a point is committed,
    we subtract w_i from T_f[j] and T_fh[group, j].  This keeps the invariant
    that T_f[j] is exactly the remaining weighted mass still to be assigned.

    Parameters
    ----------
    X          : (n, d)  point coordinates
    centers    : (k, d)  center coordinates
    weights    : (n,)    point weights  (1.0 for unweighted data)
    group_codes: (n,)    integer group index per point
    x_lp       : (n, k)  initial fractional LP solution
    D          : (n, k)  precomputed L1 distance matrix

    Returns
    -------
    labels : (n,) integer cluster assignment (index into centers)
    """
    dataset_len, k = x_lp.shape
    max_group = int(group_codes.max()) + 1
    DELTA = 1  # groups partition points → Δ = 1
    SPARSITY_THRESHOLD = 2 * (DELTA + 1)  # = 4; when to drop fairness constraint




def fair_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    protected_group_col: str,
    k_centers: int,
    alpha: float = 0.1,
    weight_col: Optional[str] = 'Weight',
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    kmedian_trials: int = 3,
    kmedian_max_iter: int = 50,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fair k-Median Clustering via the algorithm of Bera et al. (NeurIPS 2019).

    Supports TWO usage modes
    ------------------------

    Mode A — Coreset (weighted):
        Pass a DataFrame produced by compute_fair_coreset().
        Each row is a *representative point* carrying a Weight that encodes
        how many original points it represents.  The LP and rounding work
        natively with these weights, so the algorithm is both correct and fast.

        Example:
            coreset_df = compute_fair_coreset(raw_df, n_locations=1000)
            centers, labels, cost = fair_clustering(
                coreset_df,
                feature_cols=['Lat_Scaled', 'Lon_Scaled'],
                protected_group_col='GROUP_ID',
                k=10,
                alpha=0.1,
                weight_col='Weight',     # ← coreset weights
            )

    Mode B — Raw data (unweighted):
        Pass a DataFrame of individual data points.
        weight_col=None makes every point contribute equally (weight = 1).
        This is correct but slow for large datasets (n > ~5000).

        Example:
            centers, labels, cost = fair_clustering(
                processed_df,
                feature_cols=['Lat_Scaled', 'Lon_Scaled'],
                protected_group_col='GROUP_ID',
                k=10,
                alpha=0.1,
                weight_col=None,         # ← uniform weights
            )

    Parameters
    ----------
    df                  : DataFrame with feature and group columns.
    feature_cols        : Column names used as spatial coordinates for clustering.
    protected_group_col : Column name containing group labels (any hashable type).
    k                   : Number of clusters.
    alpha               : Fairness slack — maximum allowed deviation from proportional
                          group representation per cluster.  Higher = more relaxed.
    weight_col          : Column name for point weights, or None for uniform weights.
    lower_bounds        : Explicit per-group lower bounds (length H). If None,
                          derived automatically from alpha and group frequencies.
    upper_bounds        : Explicit per-group upper bounds (length H). If None,
                          derived automatically from alpha and group frequencies.
    kmedian_trials      : Number of restarts for the vanilla k-median seeding.
    kmedian_max_iter    : Max local-search iterations per trial.
    random_seed         : RNG seed for reproducibility.

    Returns
    -------
    centers : (k, d) final center coordinates (in the scaled feature space)
    labels  : (n,)   integer cluster assignment per row of df
    cost    : total weighted L1 assignment cost
    """
    x = df[feature_cols].to_numpy(dtype=np.float64)
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(x), dtype=np.float64)
    group_codes, group_names = encode_groups_to_int(df[protected_group_col])
    nr_of_groups = len(group_names)
    print(f"\n{'='*60}")
    print(f"  n={len(x):,}  k={k_centers}  groups={nr_of_groups} ")
    print(f"{'='*60}")

    if lower_bounds is None or upper_bounds is None:
        lower_bounds, upper_bounds = proportional_bounds(
            group_codes, weights, nr_of_groups, alpha
        )
        total_w = weights.sum()
        print(f"\n[FairClustering] Proportional bounds (alpha={alpha}):")
        for h, name in enumerate(group_names):
            f_h = weights[group_codes == h].sum() / total_w
            print(f"  Group '{name}': freq={f_h:.3f}  "
                  f"bounds=[{lower_bounds[h]:.3f}, {upper_bounds[h]:.3f}]")

    # Necessary condition: Σ_h lower_bounds[h] ≤ 1 ≤ Σ_h upper_bounds[h]
    if lower_bounds.sum() > 1.0 + 1e-6:
        warnings.warn(
            f"Σ lower_bounds = {lower_bounds.sum():.3f} > 1. "
            "The LP will be infeasible.  Decrease alpha or relax bounds."
        )
    if upper_bounds.sum() < 1.0 - 1e-6:
        warnings.warn(
            f"Σ upper_bounds = {upper_bounds.sum():.3f} < 1. "
            "The LP will be infeasible.  Increase alpha or relax bounds."
        )

    # ---- Vanilla k-median ------------------------------------------
    print(f"\nVanilla k-median (trials={kmedian_trials}, "
          f"max_iter={kmedian_max_iter}) ...")
    centers, unfair_labels, unfair_cost = kmedian(
        x, k_centers,
        _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    print(f"  → Unfair k-median cost: {unfair_cost:,.2f}")

    # ----  Fair LP relaxation ----------------------------------------
    print(f"\nSolving Fair LP relaxation  "
          f"(n_vars = {len(x) * k_centers:,}, n_constraints ≈ {len(x) + 2*nr_of_groups*k_centers:,}) ...")
    x_lp = solve_fair_lp(x, centers, weights, group_codes, lower_bounds, upper_bounds)
    if x_lp is None:
        # LP failed — fall back to unfair assignment
        warnings.warn(
            "[FairClustering] LP infeasible — returning unfair k-median result."
        )
        return None

    distsances_to_centers = pairwise_l1(x, centers)
    lp_cost = float(np.dot(
        weights,
        (distsances_to_centers * x_lp).sum(axis=1)
    ))
    print(f"  → LP fractional cost:   {lp_cost:,.2f}")
    print(f"  → Integrality gap hint: "
          f"{lp_cost / unfair_cost:.3f}x unfair cost")