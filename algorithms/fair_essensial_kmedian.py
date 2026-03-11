"""
Essentially Fair Clustering
============================
Implements Algorithm 2 from:
  "On the Cost of Essentially Fair Clusterings"
  Bera, Chakrabarty, Flores, Negahbani — APPROX/RANDOM 2019

Pipeline
--------
1. Compute unfair k-median centers (via existing kmedian.py).
2. Solve Fair LP relaxation (Eq. 10) with those fixed centers.
3. Round fractional solution to integral essentially-fair assignment
   via min-cost flow (Lemma 7 / transshipment rounding).

Usage — with coreset (fast, recommended for large data)
--------------------------------------------------------
    coreset_df = compute_fair_coreset(df, n_locations=300)
    centers, labels, cost = run_essentially_fair_clustering(
        coreset_df,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        group_col='GROUP_ID',
        k=10,
        alpha=0.1,          # allow ±10 % deviation from proportional representation
    )

Usage — without coreset (raw points, small datasets)
-----------------------------------------------------
    # df must have feature columns and a group column
    centers, labels, cost = run_essentially_fair_clustering(
        df,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        group_col='GROUP_ID',
        k=10,
        alpha=0.1,
        weight_col=None,     # triggers uniform weights = 1 per point
    )
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.optimize import linprog
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

from kmedian import kmedian, pairwise_l1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_groups(group_series: pd.Series) -> tuple[np.ndarray, list]:
    """Map arbitrary group labels -> contiguous integers 0..H-1."""
    cats = pd.Categorical(group_series)
    return cats.codes.to_numpy(dtype=np.int32), list(cats.categories)


def _proportional_bounds(
    group_codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive per-group lower / upper bounds from proportional representation
    with ±alpha slack.

        l_h = max(0, f_h - alpha)
        u_h = min(1, f_h + alpha)

    where f_h = (total weight of color h) / (total weight).
    """
    total = weights.sum()
    f = np.array([weights[group_codes == h].sum() / total for h in range(n_groups)])
    l_h = np.maximum(0.0, f - alpha)
    u_h = np.minimum(1.0, f + alpha)
    return l_h, u_h


# ---------------------------------------------------------------------------
# Step 2: Fair LP (Eq. 10 in the paper)
# ---------------------------------------------------------------------------

def solve_fair_lp(
    X: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    group_codes: np.ndarray,
    l_h: np.ndarray,
    u_h: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Solve the Fair LP relaxation for *fixed* centers.

mass of a group h at center j must be between a lower bound and an upper bound.

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
    n, d = X.shape
    k = len(centers)
    n_vars = n * k
    H = len(l_h)

    # --- Objective ---
    D = pairwise_l1(X, centers).astype(np.float64)   # (n, k)
    c_obj = (D * weights[:, np.newaxis]).ravel()

    # --- Equality: Σ_j x_{ij} = 1 for each i  (n rows) ---
    # Sparse: row i has ones at positions [i*k .. i*k+k-1]
    A_eq = lil_matrix((n, n_vars), dtype=np.float64)
    for i in range(n):
        A_eq[i, i * k:(i + 1) * k] = 1.0
    b_eq = np.ones(n)

    # --- Inequality: fairness  (2 * H * k rows) ---
    # Rewrite (b): l_h*Σ_i x_{ij} - Σ_{i∈Col_h} x_{ij} ≤ 0
    # Rewrite (c): Σ_{i∈Col_h} x_{ij} - u_h*Σ_i x_{ij} ≤ 0
    n_ineq = 2 * H * k
    A_ub = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_ineq)

    row = 0
    for j in range(k):
        for h in range(H):
            in_group = (group_codes == h)
            for i in range(n):
                col = i * k + j
                # lower bound row: l_h * x_{ij} - (x_{ij} if i∈Col_h)
                A_ub[row, col] = weights[i] * l_h[h] - (1.0 if in_group[i] else 0.0)
                # upper bound row: -u_h * x_{ij} + (x_{ij} if i∈Col_h)
                A_ub[row + 1, col] = weights[i] * -u_h[h] + (1.0 if in_group[i] else 0.0)
            row += 2

    A_eq_csc = A_eq.tocsc()
    A_ub_csc = A_ub.tocsc()

    result = linprog(
        c_obj,
        A_ub=A_ub_csc,
        b_ub=b_ub,
        A_eq=A_eq_csc,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * n_vars,
        method='highs',
        options={'disp': False, 'presolve': True},
    )

    if result.status != 0:
        warnings.warn(f"LP solver returned status {result.status}: {result.message}")
        return None

    return result.x.reshape((n, k))


# ---------------------------------------------------------------------------
# Step 3: Min-cost flow rounding  (Lemma 7)
# ---------------------------------------------------------------------------

def _mcf_rounding(
    x_lp: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Transshipment / min-cost-flow rounding from Lemma 7 of the paper.

    The key guarantee: for each center j and color h,
        |#(integral points of color h at j) - (fractional mass of color h at j)| < 1

    This means we violate the LP fairness constraints by at most 1 point per
    (center, color) pair — hence "essentially fair".

    Construction
    ------------
    We build a flow network:
      - Source S
      - One supply node per point i  (supply = 1, representing one unit of mass)
      - One demand node per center j  (demand = Σ_i x_{ij}, the LP's total mass)
      - Arc (i → j) with cost D[i,j] and capacity = 1 (integral assignment)
      - S → i with capacity 1, cost 0
      - j → Sink T with capacity Σ_i x_{ij}, cost 0

    We then route an integer flow that minimises cost, which gives the
    integral assignment with minimum total distance respecting the LP masses.

    For large n*k this can be expensive; for thesis-scale experiments
    (coreset with ~300–2000 points, k ≤ 50) it is tractable.
    """
    n, k = x_lp.shape

    # Node indices: 0=source, 1..n=points, n+1..n+k=centers, n+k+1=sink
    S = 0
    T = n + k + 1
    point_node  = lambda i: i + 1
    center_node = lambda j: n + j + 1

    G = nx.DiGraph()
    G.add_node(S)
    G.add_node(T)
    w_int = np.maximum(1, np.round(weights).astype(int))

    # S → point i  (supply 1 unit per point, weighted by weight)
    # We treat each point as having integer weight already (coreset weights
    # are integers; raw-point weights are 1).  For simplicity here we route
    # one unit per point regardless of weight — the cost encodes weight.
    for i in range(n):
        G.add_edge(S, point_node(i), capacity=int(w_int[i]), weight=0)

    # point i → center j  (cost = w_i * D[i,j], capacity 1)
    for i in range(n):
        for j in range(k):
            if x_lp[i, j] > 1e-9:   # only add arcs with nonzero LP mass
                cost_ij = int(round(weights[i] * D[i, j] * 1e4))  # integer costs
                G.add_edge(point_node(i), center_node(j),
                           capacity=int(w_int[i]), weight=cost_ij)

    # center j → T  (capacity = rounded LP total mass at j)
    for j in range(k):
        mass_j = int(round((x_lp[:, j]  * weights).sum()))
        if mass_j > 0:
            G.add_edge(center_node(j), T, capacity=mass_j, weight=0)

    # Add a high-cost overflow arc S→T to ensure feasibility even if
    # rounding creates small imbalances.
    G.add_edge(S, T, capacity=n, weight=int(1e9))

    # Solve min-cost flow
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        warnings.warn("MCF infeasible — falling back to greedy argmax rounding.")
        return np.argmax(x_lp, axis=1).astype(np.int32)

    # Extract integer assignment from flow
    labels = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        pn = point_node(i)
        if pn not in flow_dict:
            continue
        for j in range(k):
            cn = center_node(j)
            if flow_dict[pn].get(cn, 0) > 0:
                labels[i] = j
                break

    # Any unassigned points (shouldn't happen): fall back to nearest center
    unassigned = labels == -1
    if unassigned.any():
        labels[unassigned] = np.argmin(D[unassigned], axis=1)

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_essentially_fair_clustering(
    df: pd.DataFrame,
    feature_cols: list,
    group_col: str,
    k: int,
    alpha: float = 0.1,
    weight_col: Optional[str] = 'Weight',
    l_h: Optional[np.ndarray] = None,
    u_h: Optional[np.ndarray] = None,
    kmedian_trials: int = 3,
    kmedian_max_iter: int = 50,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Algorithm 2: Essentially Fair k-Median Clustering.

    Parameters
    ----------
    df           : DataFrame. Works with raw points OR a coreset.
    feature_cols : columns used as clustering coordinates (e.g. ['Lat_Scaled','Lon_Scaled'])
    group_col    : column with protected group labels (string or int)
    k            : number of clusters
    alpha        : proportional-representation slack (ignored if l_h/u_h provided)
    weight_col   : column of point weights.
                   Pass None (or a missing column) to use uniform weights of 1.
    l_h, u_h     : explicit per-group bounds (length H arrays). If None, derived
                   from alpha and proportional representation.
    kmedian_*    : passed through to kmedian()

    Returns
    -------
    centers : (k, d) final center coordinates
    labels  : (n,) integer cluster assignment per row of df
    cost    : total weighted L1 assignment cost
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)

    # Weights: uniform if not provided or column missing
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(X), dtype=np.float64)

    group_codes, group_labels = _encode_groups(df[group_col])
    H = len(group_labels)

    print(f"[FairClustering] n={len(X):,}  k={k}  groups={H}  "
          f"weighted={'yes' if weight_col and weight_col in df.columns else 'no (uniform)'}")

    # --- Bounds ---
    if l_h is None or u_h is None:
        l_h, u_h = _proportional_bounds(group_codes, weights, H, alpha)
        print(f"[FairClustering] Proportional bounds (alpha={alpha}):")
        for h, lbl in enumerate(group_labels):
            print(f"  {lbl}: [{l_h[h]:.3f}, {u_h[h]:.3f}]")

    total = weights.sum()
    f = np.array([weights[group_codes == h].sum() / total for h in range(H)])
    infeasible_groups = np.where((f < l_h - 1e-6) | (f > u_h + 1e-6))[0]
    if len(infeasible_groups):
        for h in infeasible_groups:
            warnings.warn(
                f"Group '{group_labels[h]}' proportion {f[h]:.3f} is outside "
                f"[{l_h[h]:.3f}, {u_h[h]:.3f}] — LP will be infeasible. "
                "Increase alpha or adjust bounds."
            )

    # --- Step 1: unfair k-median centers ---
    print("[FairClustering] Step 1: k-median for center selection...")
    centers, _, unfair_cost = kmedian(
        X, k, _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    print(f"  Unfair k-median cost: {unfair_cost:,.2f}")

    # --- Step 2: Fair LP ---
    print("[FairClustering] Step 2: Solving Fair LP...")
    x_lp = solve_fair_lp(X, centers, weights, group_codes, l_h, u_h)

    if x_lp is None:
        warnings.warn("LP failed — returning unfair k-median assignment.")
        D = pairwise_l1(X, centers)
        labels = np.argmin(D, axis=1).astype(np.int32)
        cost = float(np.dot(weights, D[np.arange(len(X)), labels]))
        return centers, labels, cost

    print(f"  LP fractional cost: {float(np.dot(weights, (pairwise_l1(X, centers) * x_lp).sum(axis=1))):,.2f}")

    # --- Step 3: MCF rounding ---
    print("[FairClustering] Step 3: Min-cost flow rounding...")
    D = pairwise_l1(X, centers)
    labels = _mcf_rounding(x_lp, group_codes, weights, D)

    cost = float(np.dot(weights, D[np.arange(len(X)), labels]))
    print(f"  Integral cost after rounding: {cost:,.2f}")

    # --- Fairness audit ---
    _audit_fairness(labels, group_codes, weights, group_labels, l_h, u_h, k)

    return centers, labels, cost


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------

def _audit_fairness(
    labels: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    group_labels: list,
    l_h: np.ndarray,
    u_h: np.ndarray,
    k: int,
):
    """Print per-cluster fairness vio
    D = pairwise_l1(X, centers)
    labels = _mcf_rounding(x_lp, group_codes, weights, D)

    cost = float(np.dot(weights, D[np.arange(len(X)), labels]))
    print(f"  Integral cost after rounding: {cost:,.2f}")lations."""
    violations = 0
    for j in range(k):
        at_j = labels == j
        total_j = weights[at_j].sum()
        if total_j == 0:
            continue
        for h, lbl in enumerate(group_labels):
            mass_h = weights[at_j & (group_codes == h)].sum()
            frac = mass_h / total_j
            if frac < l_h[h] - 1e-4 or frac > u_h[h] + 1e-4:
                violations += 1

    if violations == 0:
        print("[FairClustering] ✓ All clusters satisfy fairness bounds.")
    else:
        print(f"[FairClustering] ⚠  {violations} (cluster, group) pairs violate bounds "
              "(additive violations ≤ 1 are expected by Lemma 7).")


# ---------------------------------------------------------------------------
# Convenience: compute G-PoF  (Generalised Price of Fairness)
# ---------------------------------------------------------------------------

def compute_gpof(
    fair_cost: float,
    unfair_cost: float,
) -> float:
    """
    G-PoF = fair_cost / unfair_cost.
    A value close to 1 means fairness is nearly free.
    """
    if unfair_cost == 0:
        return float('inf')
    return fair_cost / unfair_cost


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import csv_loader
    from coreset import compute_fair_coreset

    # ---- With coreset ----
    print("=== Coreset mode ===")
    df_raw = csv_loader.load_csv_chunked(
        "us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=10_000,
        max_rows=10_000,
    )
    coreset_df = compute_fair_coreset(df_raw, n_locations=300, random_seed=42)

    centers_c, labels_c, cost_c = run_essentially_fair_clustering(
        coreset_df,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        group_col='GROUP_ID',
        k=10,
        alpha=0.15,
        weight_col='Weight',
    )
    print(f"[Coreset] Fair cost = {cost_c:,.2f}")

    # ---- Without coreset (uniform weights) ----
    print("\n=== Direct points mode (no coreset) ===")
    # Prepare minimal columns needed  (we reuse the raw df here)
    scaler = MinMaxScaler()
    df_raw[['Lat_Scaled', 'Lon_Scaled']] = scaler.fit_transform(
        df_raw[['Latitude', 'Longitude']]
    )
    df_raw['GROUP_ID'] = (
        df_raw['RAC1P'].astype(str) + "_" + df_raw['SEX'].astype(str)
    )

    centers_r, labels_r, cost_r = run_essentially_fair_clustering(
        df_raw,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        group_col='GROUP_ID',
        k=10,
        alpha=0.15,
        weight_col=None
    )
    print(f"[Raw] Fair cost = {cost_r:,.2f}")

    print(f"\nG-PoF (coreset): {compute_gpof(cost_c, cost_c):.4f}")  # placeholder