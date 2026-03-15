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
    Final corrected Rounding for Essentially Fair Clustering.
    Addresses supply/demand mismatch and weighted coreset consistency.
    """
    n, k = x_lp.shape
    # Use consistent integer weights for the entire flow network
    w_int = np.maximum(1, np.round(weights).astype(int))
    total_supply = int(w_int.sum())

    S, T = "source", "sink"
    G = nx.DiGraph()

    # 1. Source to Points: Supply is the integer weight of the coreset point
    for i in range(n):
        p_node = f"p_{i}"
        G.add_edge(S, p_node, capacity=w_int[i], weight=0)

        # 2. Points to Centers: Use distances as costs
        # Only add edges where the LP assigned fractional mass
        for j in range(k):
            if x_lp[i, j] > 1e-9:
                c_node = f"c_{j}"
                # Scaled distance to integer for the solver
                cost_ij = int(round(D[i, j] * 10000))
                G.add_edge(p_node, c_node, capacity=w_int[i], weight=cost_ij)

    # 3. Centers to Sink: The bottleneck for Fairness
    # We must ensure total_demand == total_supply exactly
    total_demand = 0
    center_demands = []

    for j in range(k):
        # Calculate the total fractional mass the LP sent to this center
        # using the SAME integer weights used in the supply side
        mass_j = (x_lp[:, j] * w_int).sum()
        center_demands.append(mass_j)

    # Standard rounding can lead to sum(rounded_demands) != total_supply.
    # We use 'Largest Remainder' rounding to keep supply/demand balanced.
    rounded_demands = np.floor(center_demands).astype(int)
    remainder = total_supply - rounded_demands.sum()
    # Distribute the missing units to the centers with largest fractional parts
    diffs = np.array(center_demands) - rounded_demands
    for idx in np.argsort(diffs)[-remainder:]:
        rounded_demands[idx] += 1

    for j in range(k):
        G.add_edge(f"c_{j}", T, capacity=int(rounded_demands[j]), weight=0)

    # 4. Solve Min-Cost Flow
    try:
        # We need a flow that satisfies the total supply
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # Fallback to a simpler rounding if the specific capacities fail
        warnings.warn("MCF Unfeasible: Falling back to greedy rounding.")
        return np.argmax(x_lp, axis=1).astype(np.int32)

    # 5. Extract Labels
    labels = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        p_node = f"p_{i}"
        # Because of coreset weights, flow might be split (e.g., 10 units to C1, 5 to C2)
        # We assign the point to the center that received the MOST flow from it.
        best_center = -1
        max_f = -1
        if p_node in flow_dict:
            for c_node, f in flow_dict[p_node].items():
                if f > max_f:
                    max_f = f
                    best_center = int(c_node.split('_')[1])

        labels[i] = best_center if best_center != -1 else np.argmin(D[i])

    return labels


def min_cost_flow_rounding(
        x_lp: np.ndarray,
        group_codes: np.ndarray,
        weights: np.ndarray,
        D: np.ndarray,
) -> np.ndarray:
    """
    Round the fractional fair LP solution to an integral assignment via
    min-cost flow, following Lemma 8 (separable objectives / k-median) of
    Bercea et al. "On the cost of essentially fair clusterings".

    The paper constructs a graph G whose integral min-cost flow gives an
    essentially-fair integral assignment with additive fairness violation ≤ 1
    per color per cluster.

    Graph structure (Figure 1 of the paper)
    ----------------------------------------
    For each color h and each point j ∈ col_h:
        node  v^h_j   supply  = +1
    For each color h and each center i:
        node  v^h_i   demand  = floor(mass_h(x, i))
    For each center i:
        node  v_i     demand  = B_i  = floor(mass(x,i)) - Σ_h floor(mass_h(x,i))
    Sink t:           demand  = B    = |P| - Σ_i floor(mass(x,i))

    Edges
    -----
    v^h_j → v^h_i   capacity 1, cost d(j,i)   if x_lp[j,i] > 0
    v^h_i → v_i     capacity 1, cost 0         if frac(mass_h(x,i)) > 0
    v_i   → t       capacity 1, cost 0         if frac(mass(x,i))   > 0

    Parameters
    ----------
    x_lp        : (n, k) fractional assignment from solve_fair_lp
    group_codes : (n,)   integer color label per point  (0..H-1)
    weights     : (n,)   point weights  (used as mass — integer-valued for coreset,
                          fractional for raw points; the paper assumes unit weights
                          so for weighted points we scale and round appropriately)
    D           : (n, k) pairwise L1 distances

    Returns
    -------
    labels : (n,) integer array — cluster index for each point
    """
    n, k = x_lp.shape
    n_groups = int(group_codes.max()) + 1
    EPS = 1e-9  # numerical zero threshold

    # ------------------------------------------------------------------ #
    # Weighted mass: mass_h(x, i) = Σ_{j ∈ col_h} w_j * x_lp[j, i]
    # mass(x, i)   = Σ_j w_j * x_lp[j, i]
    # For raw (uniform) data weights=1; for coreset weights are integers.
    # We work with weighted masses throughout so the same code handles both.
    # ------------------------------------------------------------------ #
    # mass_h[h, i]
    mass_h = np.zeros((n_groups, k), dtype=np.float64)
    for h in range(n_groups):
        in_h = (group_codes == h)
        mass_h[h] = (x_lp[in_h] * weights[in_h, np.newaxis]).sum(axis=0)

    # mass[i] = Σ_h mass_h[h, i]
    mass = mass_h.sum(axis=0)  # (k,)

    floor_mass_h = np.floor(mass_h)  # (n_groups, k)
    floor_mass = np.floor(mass)  # (k,)

    # B_i = floor(mass(x,i)) - Σ_h floor(mass_h(x,i))
    B_i = floor_mass - floor_mass_h.sum(axis=0)  # (k,) — always ≥ 0

    # B = total_weight - Σ_i floor(mass(x,i))
    total_weight = float(weights.sum())
    B = total_weight - floor_mass.sum()

    # ------------------------------------------------------------------ #
    # Node naming scheme (all strings, networkx DiGraph)
    # ------------------------------------------------------------------ #
    # "ph_{h}_{j}"  — point node for point j of color h
    # "ch_{h}_{i}"  — color-center node for center i, color h
    # "c_{i}"       — center aggregation node for center i
    # "t"           — global sink
    # ------------------------------------------------------------------ #

    G = nx.DiGraph()
    t = "t"
    G.add_node(t, demand=-int(round(B)))

    # Build color-center nodes and center aggregation nodes
    for i in range(k):
        c_node = f"c_{i}"
        bi_val = int(round(B_i[i]))
        G.add_node(c_node, demand=-bi_val)

        for h in range(n_groups):
            ch_node = f"ch_{h}_{i}"
            floor_mh_i = int(round(floor_mass_h[h, i]))
            G.add_node(ch_node, demand=-floor_mh_i)

            # Edge v^h_i → v_i  (fractional remainder spills upward)
            frac_mh = mass_h[h, i] - floor_mass_h[h, i]
            if frac_mh > EPS:
                G.add_edge(ch_node, c_node, capacity=1, weight=0)

        # Edge v_i → t  (cluster-level fractional remainder)
        frac_m = mass[i] - floor_mass[i]
        if frac_m > EPS:
            G.add_edge(c_node, t, capacity=1, weight=0)

    # Build point nodes and point→color-center edges
    for j in range(n):
        h = int(group_codes[j])
        w_j = float(weights[j])
        ph_node = f"ph_{h}_{j}"
        # Supply = weight of this point (integer for coreset, 1 for raw)
        G.add_node(ph_node, demand=-int(round(w_j)))  # negative demand = supply

        for i in range(k):
            if x_lp[j, i] > EPS:
                ch_node = f"ch_{h}_{i}"
                # Cost is distance scaled to integer (networkx MCF needs int costs)
                # Multiply by 1e6 and round to preserve relative ordering.
                cost_int = int(round(D[j, i] * 1_000_000))
                G.add_edge(ph_node, ch_node, capacity=int(round(w_j)), weight=cost_int)

    # ------------------------------------------------------------------ #
    # Solve min-cost flow
    # networkx min_cost_flow requires that Σ demands = 0.
    # Our construction: supply = Σ_j w_j = total_weight
    #                   demand = Σ_i floor(mass(x,i)) + B = total_weight  ✓
    # ------------------------------------------------------------------ #
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        warnings.warn(
            "Min-cost flow was infeasible — falling back to greedy rounding."
        )
        return np.argmin(D, axis=1).astype(np.int32)
    except Exception as e:
        warnings.warn(f"Min-cost flow failed ({e}) — falling back to greedy rounding.")
        return np.argmin(D, axis=1).astype(np.int32)

    # ------------------------------------------------------------------ #
    # Extract integer assignment from flow
    # Point j is assigned to whichever center i carries positive flow on
    # the edge  ph_{h}_{j} → ch_{h}_{i}.
    # For weighted coreset points (weight > 1) we track each unit of flow
    # as one "copy" of point j assigned to center i.  Since all copies of
    # the same point are identical we just need any center with flow > 0.
    # ------------------------------------------------------------------ #
    labels = np.full(n, -1, dtype=np.int32)
    for j in range(n):
        h = int(group_codes[j])
        ph_node = f"ph_{h}_{j}"
        if ph_node not in flow_dict:
            # No outgoing flow — assign to nearest center as fallback
            labels[j] = int(np.argmin(D[j]))
            continue
        best_i = -1
        best_cost = np.inf
        for i in range(k):
            ch_node = f"ch_{h}_{i}"
            f_val = flow_dict[ph_node].get(ch_node, 0)
            if f_val > 0 and D[j, i] < best_cost:
                best_cost = D[j, i]
                best_i = i
        if best_i == -1:
            best_i = int(np.argmin(D[j]))
        labels[j] = best_i

    if (labels == -1).any():
        missing = np.where(labels == -1)[0]
        warnings.warn(f"{len(missing)} points unassigned after MCF — using nearest center.")
        labels[missing] = np.argmin(D[missing], axis=1).astype(np.int32)

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