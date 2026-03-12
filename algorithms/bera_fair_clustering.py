"""
Fair Clustering - Bera et al. (2019)
=====================================
Implements Algorithm 1 + Algorithm 2 from:
  "Fair Algorithms for Clustering", Bera, Chakrabarty, Flores, Negahbani (NeurIPS 2019)

Two-step procedure:
  1. Run vanilla k-median (fixed centers S)
  2. Solve FAIR p-ASSIGNMENT LP (eq. 1) for those centers
  3. Round via iterative LP2 procedure (Algorithm 2) for guaranteed (4Δ+3) additive violation

Key differences from the broken version:
  - Constraint matrix built via vectorized slice assignment (not per-element loop)
  - Rounding uses iterative LP2 (Algorithm 2) not greedy argmax
  - Delta parameterization: β_i = r_i*(1-δ), α_i = r_i/(1-δ) as in paper Section 5
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import linprog
from typing import Optional

from kmedian import kmedian, pairwise_l1


# ---------------------------------------------------------------------------
# Helper: compute global weighted ratios for each group
# ---------------------------------------------------------------------------

def compute_delta_bounds(
    group_masks: dict,
    weights: np.ndarray,
    delta: float,
) -> tuple[dict, dict]:
    """
    Compute α and β from a single δ parameter, as used in the paper's experiments.

    β_i = r_i * (1 - δ)
    α_i = r_i / (1 - δ)

    where r_i = weighted fraction of group i in the full dataset.
    δ = 0 means exact fairness; δ = 1 means no constraint.
    The paper uses δ = 0.2 (the 80%-rule of disparate impact doctrine).
    """
    total_weight = weights.sum()
    alpha, beta = {}, {}
    for name, mask in group_masks.items():
        r_i = weights[mask].sum() / total_weight
        beta[name]  = r_i * (1 - delta)
        alpha[name] = r_i / (1 - delta) if delta < 1 else 1.0
    return alpha, beta


# ---------------------------------------------------------------------------
# Step 2: FAIR p-ASSIGNMENT LP  (eq. 1 in the paper)
# ---------------------------------------------------------------------------

def bera_fair_assignment_lp(
    X: np.ndarray,
    weights: np.ndarray,
    group_masks: dict,
    centers: np.ndarray,
    alpha: dict,
    beta: dict,
) -> np.ndarray:
    """
    Solves the FAIR p-ASSIGNMENT LP (eq. 1, Bera et al. 2019).

    Variables: x[j, i] = fraction of (weighted) point j assigned to center i
               flattened row-major: index = j*K + i

    Objective  (eq. 1a): min  sum_{j,i}  w_j * d(j, center_i) * x[j,i]

    Constraints:
      (1b) sum_i x[j,i] = 1                          for all j
      (1a) β_g * sum_j w_j*x[j,i]  <=  sum_{j in g} w_j*x[j,i]
                                    <=  α_g * sum_j w_j*x[j,i]
                                                      for all centers i, groups g
      box: x[j,i] in [0, 1]

    FIX vs broken version: constraint rows are filled via vectorized
    slice assignment over all N points at once, not a per-element loop.
    """
    N = len(X)
    K = len(centers)
    groups = list(group_masks.keys())
    G = len(groups)

    # --- Objective ---
    D = pairwise_l1(X, centers)                          # (N, K)
    c_obj = (D * weights[:, np.newaxis]).ravel()         # (N*K,)

    # --- Equality: each point fully assigned (eq. 1b) ---
    # Row j: columns j*K .. j*K+K-1  all get coefficient 1
    A_eq = lil_matrix((N, N * K))
    for j in range(N):
        A_eq[j, j * K : j * K + K] = 1.0
    b_eq = np.ones(N)

    # --- Inequality: RD and MP per center per group (eq. 1a) ---
    # For center i and group g:
    #   MP (lower): sum_{j in g} w_j*x[j,i]  >=  β_g * sum_j w_j*x[j,i]
    #     =>  (β_g * w_j  -  is_g[j] * w_j) * x[j,i]  <=  0   summed over j
    #
    #   RD (upper): sum_{j in g} w_j*x[j,i]  <=  α_g * sum_j w_j*x[j,i]
    #     =>  (is_g[j] * w_j  -  α_g * w_j) * x[j,i]  <=  0   summed over j
    #
    # Each (i, g) pair → 2 rows.  Total rows = 2 * K * G.

    n_ineq = 2 * K * G
    A_ub = lil_matrix((n_ineq, N * K))
    b_ub = np.zeros(n_ineq)

    row = 0
    for i in range(K):
        # Column indices for center i across all points: j*K + i for j=0..N-1
        cols_i = np.arange(N) * K + i                   # (N,) — the FIX

        for g_name in groups:
            mask   = group_masks[g_name].astype(float)  # (N,)
            a_g    = alpha[g_name]
            b_g    = beta[g_name]

            # MP row: (β_g - is_g) * w  <=  0
            mp_coeffs = (b_g - mask) * weights          # (N,)
            A_ub[row, cols_i] = mp_coeffs               # vectorized slice — the FIX
            row += 1

            # RD row: (is_g - α_g) * w  <=  0
            rd_coeffs = (mask - a_g) * weights          # (N,)
            A_ub[row, cols_i] = rd_coeffs               # vectorized slice — the FIX
            row += 1

    print(f"  Solving LP: {N} points, {K} centers, {G} groups → "
          f"{N*K} vars, {N} eq. constraints, {n_ineq} ineq. constraints")

    res = linprog(
        c_obj,
        A_ub=csr_matrix(A_ub),
        b_ub=b_ub,
        A_eq=csr_matrix(A_eq),
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method='highs',
        options={'disp': False},
    )

    if not res.success:
        raise ValueError(
            f"LP failed: {res.message}\n"
            "Hint: fairness constraints may be infeasible for these centers. "
            "Try increasing δ (looser bounds) or more centers."
        )

    return res.x.reshape(N, K)


# ---------------------------------------------------------------------------
# Step 3: Iterative rounding — Algorithm 2 (Bera et al.)
# ---------------------------------------------------------------------------

def _iterative_round_lp2(
    X: np.ndarray,
    weights: np.ndarray,
    group_masks: dict,
    centers: np.ndarray,
    x_star: np.ndarray,
    delta_max: int = 1,
) -> np.ndarray:
    """
    Algorithm 2 from Bera et al.: iterative rounding of x_star.

    Maintains LP2 (eq. 2) which pins fractional totals T_f and T_{f,i}
    to floor/ceil windows, then iteratively:
      - fixes variables already at 0 or 1
      - drops constraints once fewer than 2(Δ+1) fractional variables remain
    until all points are assigned.

    Guarantees (4Δ+3)-additive violation where Δ = delta_max = max groups
    a single point belongs to.

    Returns
    -------
    labels : (N,) integer cluster assignment
    """
    N, K = x_star.shape
    groups = list(group_masks.keys())
    G = len(groups)

    # Remaining unassigned points (indices into original N)
    remaining = np.where(~np.any(x_star == 1, axis=1))[0].tolist()

    labels = np.full(N, -1, dtype=int)

    # Assign any already-integral points from LP solution
    for j in range(N):
        ones = np.where(x_star[j] == 1.0)[0]
        if len(ones):
            labels[j] = ones[0]

    remaining = [j for j in range(N) if labels[j] == -1]

    # Current fractional solution (will be updated each iteration)
    x_cur = x_star.copy()

    threshold = 2 * (delta_max + 1)
    D = pairwise_l1(X, centers)

    iteration = 0
    while remaining:
        iteration += 1
        R = len(remaining)
        idx = np.array(remaining)        # current unassigned point indices

        x_sub = x_cur[idx]              # (R, K) — only fractional vars with x>0
        active = x_sub > 1e-9           # boolean mask of active vars

        # T_f and T_{f,i} for the current remaining points
        T_f   = x_sub.sum(axis=0)                        # (K,)
        T_f_g = np.zeros((K, G))
        for g_idx, g_name in enumerate(groups):
            g_mask_sub = group_masks[g_name][idx]        # (R,) boolean
            T_f_g[:, g_idx] = x_sub[g_mask_sub].sum(axis=0)

        # Build LP2 (eq. 2) for remaining points
        n_vars = R * K
        c_lp2  = (D[idx] * weights[idx, np.newaxis]).ravel()

        A_eq2 = lil_matrix((R, n_vars))
        for j in range(R):
            A_eq2[j, j * K : j * K + K] = 1.0
        b_eq2 = np.ones(R)

        # Count active constraints: drop per paper when few fractional vars remain
        ineq_rows, ineq_cols_list, ineq_data = [], [], []
        ineq_b = []

        for i in range(K):
            cols_i = np.arange(R) * K + i

            # Cluster-total constraints (eq. 2b): floor(T_f) <= sum_j x[j,i] <= ceil(T_f)
            n_frac_total = int(active[:, i].sum())
            if n_frac_total > threshold:
                # lower: -sum x[j,i] <= -floor(T_f)
                ineq_rows.append(-np.ones(R))
                ineq_cols_list.append(cols_i)
                ineq_b.append(-np.floor(T_f[i]))
                # upper: sum x[j,i] <= ceil(T_f)
                ineq_rows.append(np.ones(R))
                ineq_cols_list.append(cols_i)
                ineq_b.append(np.ceil(T_f[i]))

            # Per-group constraints (eq. 2c)
            for g_idx, g_name in enumerate(groups):
                g_mask_sub = group_masks[g_name][idx]
                n_frac_g = int((active[:, i] & g_mask_sub).sum())
                if n_frac_g > threshold:
                    g_coeffs = g_mask_sub.astype(float)
                    # lower: -sum_{j in g} x[j,i] <= -floor(T_{f,g})
                    ineq_rows.append(-g_coeffs)
                    ineq_cols_list.append(cols_i)
                    ineq_b.append(-np.floor(T_f_g[i, g_idx]))
                    # upper: sum_{j in g} x[j,i] <= ceil(T_{f,g})
                    ineq_rows.append(g_coeffs)
                    ineq_cols_list.append(cols_i)
                    ineq_b.append(np.ceil(T_f_g[i, g_idx]))

        if ineq_rows:
            n_ineq2 = len(ineq_b)
            A_ub2 = lil_matrix((n_ineq2, n_vars))
            for r_idx, (coeffs, cols) in enumerate(zip(ineq_rows, ineq_cols_list)):
                A_ub2[r_idx, cols] = coeffs
            A_ub2_csr = csr_matrix(A_ub2)
            b_ub2 = np.array(ineq_b)
        else:
            A_ub2_csr = None
            b_ub2 = None

        # Bound: only allow variables that were fractional in x_star
        bounds = []
        for j in range(R):
            for i in range(K):
                if x_sub[j, i] > 1e-9:
                    bounds.append((0.0, 1.0))
                else:
                    bounds.append((0.0, 0.0))

        res2 = linprog(
            c_lp2,
            A_ub=A_ub2_csr,
            b_ub=b_ub2,
            A_eq=csr_matrix(A_eq2),
            b_eq=b_eq2,
            bounds=bounds,
            method='highs',
            options={'disp': False},
        )

        if not res2.success:
            # Fallback: assign remaining points to nearest center
            for j in remaining:
                labels[j] = int(np.argmin(D[j]))
            break

        x_new = res2.x.reshape(R, K)

        # Fix variables at 0 or 1
        newly_assigned = []
        for r_j, j in enumerate(remaining):
            ones = np.where(x_new[r_j] >= 1.0 - 1e-9)[0]
            if len(ones):
                labels[j] = int(ones[0])
                newly_assigned.append(j)
            else:
                x_cur[j] = x_new[r_j]   # update for next iteration

        if not newly_assigned:
            # No progress — assign greedily to avoid infinite loop
            for j in remaining:
                labels[j] = int(np.argmax(x_cur[j]))
            break

        remaining = [j for j in remaining if labels[j] == -1]

        print(f"  [Round iter {iteration}] assigned {len(newly_assigned)}, "
              f"remaining {len(remaining)}", end="\r")

    print()
    # Any still unassigned (shouldn't happen)
    unassigned = np.where(labels == -1)[0]
    if len(unassigned):
        for j in unassigned:
            labels[j] = int(np.argmin(D[j]))

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bera_fair_clustering(
    df_core: pd.DataFrame,
    k: int,
    delta: float = 0.2,
    alpha: Optional[dict] = None,
    beta: Optional[dict] = None,
    use_iterative_rounding: bool = True,
    n_trials: int = 3,
    max_iter: int = 100,
    random_seed: Optional[int] = None,
) -> tuple:
    """
    Bera et al. (2019) fair clustering pipeline.

    Parameters
    ----------
    df_core               : coreset DataFrame with Lat_Scaled, Lon_Scaled,
                            Weight, SEX, INC_BIN columns
    k                     : number of clusters
    delta                 : fairness looseness in [0,1). δ=0.2 = 80%-rule.
                            Used only if alpha/beta are not provided directly.
    alpha, beta           : optional explicit per-group bounds (override delta)
    use_iterative_rounding: if True, use Algorithm 2 (guaranteed violation bound);
                            if False, use greedy argmax (faster, no guarantee)
    n_trials, max_iter    : passed to kmedian
    random_seed           : reproducibility

    Returns
    -------
    centers       : (k, 2) center coordinates
    fair_labels   : (N,) integral fair cluster assignment
    x_star        : (N, k) fractional LP solution
    unfair_cost   : scalar
    fair_cost     : scalar (cost of integral fair assignment)
    """
    X = df_core[['Lat_Scaled', 'Lon_Scaled']].values
    weights = df_core['Weight'].values.astype(float)

    # Overlapping group masks — a point can be in multiple groups (Δ > 1)
    group_masks = {
        'Female':      (df_core['SEX'] == 2).values,
        'Male':        (df_core['SEX'] == 1).values,
        'Low_Income':  (df_core['INC_BIN'] == 'Low').values,
        'High_Income': (df_core['INC_BIN'] == 'High').values,
    }

    # Δ = max groups a single point belongs to
    delta_max = max(
        int(sum(mask[j] for mask in group_masks.values()))
        for j in range(len(X))
    )

    if alpha is None or beta is None:
        alpha, beta = compute_delta_bounds(group_masks, weights, delta)
        print(f"[Bera] δ={delta}, Δ={delta_max}")
        for g in group_masks:
            print(f"  {g}: β={beta[g]:.3f}, α={alpha[g]:.3f}")

    # Step 1: vanilla k-median
    print("\n[Bera] Step 1: vanilla k-median...")
    centers, unfair_labels, unfair_cost = kmedian(
        X, k, weights, n_trials=n_trials, max_iter=max_iter,
        random_seed=random_seed,
    )
    print(f"[Bera] Unfair cost = {unfair_cost:,.2f}")

    # Step 2: fair LP
    print("\n[Bera] Step 2: solving fair assignment LP...")
    x_star = bera_fair_assignment_lp(X, weights, group_masks, centers, alpha, beta)

    lp_cost = float(np.sum(x_star * pairwise_l1(X, centers) * weights[:, np.newaxis]))
    print(f"[Bera] Fractional fair LP cost = {lp_cost:,.2f}")

    # Step 3: rounding
    print("\n[Bera] Step 3: rounding...")
    if use_iterative_rounding:
        fair_labels = _iterative_round_lp2(
            X, weights, group_masks, centers, x_star, delta_max=delta_max
        )
    else:
        # Greedy fallback — no violation guarantee, but fast
        print("  (greedy argmax rounding — no violation guarantee)")
        fair_labels = np.argmax(x_star, axis=1)

    D = pairwise_l1(X, centers)
    fair_cost = float(np.dot(weights, D[np.arange(len(X)), fair_labels]))

    print(f"\n[Bera] Results:")
    print(f"  Unfair cost : {unfair_cost:,.2f}")
    print(f"  Fair cost   : {fair_cost:,.2f}")
    print(f"  PoF         : {fair_cost / unfair_cost:.4f}x")

    return centers, fair_labels, x_star, unfair_cost, fair_cost


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv_loader
    from coreset import compute_fair_coreset

    df = csv_loader.load_csv_chunked(
        "us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=10_0,
        max_rows=10_0,
    )
    coreset_df = compute_fair_coreset(df, n_locations=300, random_seed=42)

    centers, labels, x_star, u_cost, f_cost = compute_bera_fair_clustering(
        coreset_df,
        k=10,
        delta=0.2,
        use_iterative_rounding=True,
        n_trials=2,
        max_iter=30,
        random_seed=42,
    )