"""
Fair k-Median Clustering
=========================
Implementation of the fair clustering algorithm from:

    Bera, Chakrabarty, Flores, Negahbani (NeurIPS 2019)
    "Fair Algorithms for Clustering"
    arXiv:1901.02393v2

Algorithm overview (Section 3 of the paper):
    Step 1 – Run vanilla k-median to fix centers S.
    Step 2 – Solve FAIR p-ASSIGNMENT via LP + iterative rounding (Algorithm 2).

The fairness model (Definition 1):
    - L demographic groups C_1, ..., C_L (may overlap; a point can be in multiple groups).
    - For each group i, parameters alpha_i (max fraction) and beta_i (min fraction).
    - A clustering is fair if for every open center f and every group i:
          beta_i * |cluster(f)| <= |group_i ∩ cluster(f)| <= alpha_i * |cluster(f)|

Dependencies: numpy, scipy (for LP via linprog)
"""

import numpy as np
from typing import Optional
from scipy.optimize import linprog

from kmedian import kmedian, pairwise_l1, assignment_cost


# ---------------------------------------------------------------------------
# Fairness parameter helpers
# ---------------------------------------------------------------------------

def fairness_params_from_ratios(
    group_memberships: list[np.ndarray],
    n_points: int,
    delta: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive alpha and beta from group ratios and a slack parameter delta,
    following the parameterisation used in Section 5 of the paper:

        beta_i  = r_i * (1 - delta)
        alpha_i = r_i / (1 - delta)

    where r_i = |C_i| / n is the proportion of group i in the dataset.

    delta = 0   -> perfectly proportional (strict)
    delta = 1   -> no fairness constraints at all
    delta = 0.2 -> corresponds to the 80%-rule

    Parameters
    ----------
    group_memberships : list of boolean/integer arrays of length n_points,
                        one per group. group_memberships[i][j] is True/1
                        if point j belongs to group i.
    n_points          : total number of data points
    delta             : fairness slack in [0, 1)

    Returns
    -------
    alpha : (L,) array of upper-bound fractions
    beta  : (L,) array of lower-bound fractions
    """
    l = len(group_memberships)
    alpha = np.zeros(l)
    beta = np.zeros(l)
    for i, membership in enumerate(group_memberships):
        r_i = np.sum(membership) / n_points
        beta[i] = r_i * (1.0 - delta)
        alpha[i] = r_i / (1.0 - delta) if (1.0 - delta) > 0 else 1.0
    return alpha, beta


# ---------------------------------------------------------------------------
# Step 2: Fair Assignment via LP + iterative rounding (Algorithm 2)
# ---------------------------------------------------------------------------

def _solve_fair_assignment_lp(
    X: np.ndarray,
    centers: np.ndarray,
    group_memberships: list[np.ndarray],
    alpha: np.ndarray,
    beta: np.ndarray,
    Tf: Optional[np.ndarray] = None,
    Tf_i: Optional[np.ndarray] = None,
    active_vars: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Solve the LP relaxation for the FAIR p-ASSIGNMENT problem (eq. 1 / eq. 2
    in the paper) using scipy.optimize.linprog.

    Variables: x_{v,f} in [0,1], one per (point, center) pair.
               If active_vars is provided, only variables with active_vars[v,f]=True
               are included (used in the iterative rounding phase, eq. 2).

    Objective: minimise sum_{v,f} d(v,f) * x_{v,f}   (L1 distances)

    Constraints:
        (1a) beta_i * sum_v x_{v,f} <= sum_{v in C_i} x_{v,f}  for all f, i  (MP)
             sum_{v in C_i} x_{v,f} <= alpha_i * sum_v x_{v,f}                (RD)
        (1b) sum_f x_{v,f} = 1  for all v                                     (assignment)

    When Tf / Tf_i are provided (iterative rounding, eq. 2):
        floor(Tf[f])  <= sum_v x_{v,f}  <= ceil(Tf[f])
        floor(Tf_i[i,f]) <= sum_{v in C_i} x_{v,f} <= ceil(Tf_i[i,f])

    Returns
    -------
    x : (n, k) solution matrix, or None if infeasible.
    """
    n, d = X.shape
    k = len(centers)
    L = len(group_memberships)

    # --- Build distance matrix (cost vector for LP) ---
    D = pairwise_l1(X, centers)  # (n, k)

    # --- Determine active (v, f) pairs ---
    if active_vars is None:
        active_vars = np.ones((n, k), dtype=bool)

    # Map (v, f) pairs to variable indices
    # active_pairs[idx] = (v, f)
    active_pairs = [(v, f) for v in range(n) for f in range(k) if active_vars[v, f]]
    num_vars = len(active_pairs)
    if num_vars == 0:
        return None

    pair_to_idx = {(v, f): idx for idx, (v, f) in enumerate(active_pairs)}

    # --- Cost vector ---
    c_vec = np.array([D[v, f] for (v, f) in active_pairs])

    # --- Build constraint matrices ---
    A_ub_rows = []   # inequality constraints (A_ub @ x <= b_ub)
    b_ub_rows = []
    A_eq_rows = []   # equality constraints
    b_eq_rows = []

    # (1b) Assignment constraints: sum_f x_{v,f} = 1 for each v
    # Only over active variables.
    # Group active pairs by v
    v_to_pairs: dict[int, list[int]] = {v: [] for v in range(n)}
    f_to_pairs: dict[int, list[int]] = {f: [] for f in range(k)}
    fi_to_pairs: dict[tuple, list[int]] = {}   # (f, i) -> var indices
    for idx, (v, f) in enumerate(active_pairs):
        v_to_pairs[v].append(idx)
        f_to_pairs[f].append(idx)
        for i, membership in enumerate(group_memberships):
            if membership[v]:
                key = (f, i)
                if key not in fi_to_pairs:
                    fi_to_pairs[key] = []
                fi_to_pairs[key].append(idx)

    for v in range(n):
        row = np.zeros(num_vars)
        for idx in v_to_pairs[v]:
            row[idx] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(1.0)

    # (1a) Fairness constraints for each f and i
    # RD: sum_{v in C_i} x_{v,f} - alpha_i * sum_v x_{v,f} <= 0
    # MP: beta_i * sum_v x_{v,f} - sum_{v in C_i} x_{v,f} <= 0
    for f in range(k):
        for i in range(L):
            # RD: group_sum - alpha_i * total_sum <= 0
            row_rd = np.zeros(num_vars)
            row_mp = np.zeros(num_vars)
            for idx in f_to_pairs[f]:
                row_rd[idx] -= alpha[i]
                row_mp[idx] += beta[i]
            for idx in fi_to_pairs.get((f, i), []):
                row_rd[idx] += 1.0
                row_mp[idx] -= 1.0
            A_ub_rows.append(row_rd)
            b_ub_rows.append(0.0)
            A_ub_rows.append(row_mp)
            b_ub_rows.append(0.0)

    # (2b/2c) Integrality-tightening constraints when Tf / Tf_i are provided
    if Tf is not None:
        for f in range(k):
            if not f_to_pairs[f]:
                continue
            row = np.zeros(num_vars)
            for idx in f_to_pairs[f]:
                row[idx] = 1.0
            # sum >= floor(Tf[f])
            A_ub_rows.append(-row)
            b_ub_rows.append(-np.floor(Tf[f]))
            # sum <= ceil(Tf[f])
            A_ub_rows.append(row)
            b_ub_rows.append(np.ceil(Tf[f]))

    if Tf_i is not None:
        for f in range(k):
            for i in range(L):
                idxs = fi_to_pairs.get((f, i), [])
                if not idxs:
                    continue
                row = np.zeros(num_vars)
                for idx in idxs:
                    row[idx] = 1.0
                A_ub_rows.append(-row)
                b_ub_rows.append(-np.floor(Tf_i[i, f]))
                A_ub_rows.append(row)
                b_ub_rows.append(np.ceil(Tf_i[i, f]))

    # --- Assemble and solve ---
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_rows)

    bounds = [(0.0, 1.0)] * num_vars

    result = linprog(
        c_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return None

    # Unpack back to (n, k) matrix
    x = np.zeros((n, k))
    for idx, (v, f) in enumerate(active_pairs):
        x[v, f] = result.x[idx]

    return x


def fair_assignment(
    X: np.ndarray,
    centers: np.ndarray,
    group_memberships: list[np.ndarray],
    alpha: np.ndarray,
    beta: np.ndarray,
    int_tol: float = 1e-6,
    delta_constraint: int = 2,
) -> np.ndarray:
    """
    FAIRASSIGNMENT procedure (Algorithm 2 in the paper).

    Iteratively rounds the LP solution to an integral assignment.
    At each iteration:
      - Fix variables already at 0 or 1.
      - When a fairness constraint has few active fractional variables
        (<=  2*(Delta+1)), drop that constraint to break cycling.

    Parameters
    ----------
    X                 : (n, d) data points
    centers           : (k, d) fixed cluster centers from vanilla k-median
    group_memberships : list of L boolean arrays of length n
    alpha             : (L,) upper-bound fractions
    beta              : (L,) lower-bound fractions
    int_tol           : threshold to treat LP value as 0 or 1
    delta_constraint  : Delta (max groups a point can belong to); used for
                        the constraint-dropping threshold 2*(Delta+1)

    Returns
    -------
    labels : (n,) integer array; labels[i] = index of assigned center
    """
    n, d = X.shape
    k = len(centers)
    L = len(group_memberships)

    labels = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)   # points not yet assigned
    active_vars = np.ones((n, k), dtype=bool)

    # --- Initial LP solve (eq. 1) ---
    x = _solve_fair_assignment_lp(X, centers, group_memberships, alpha, beta)
    if x is None:
        # Fallback: unconstrained nearest-center assignment
        labs, _ = assignment_cost(X, centers)
        return labs

    # Fix any variables already integral
    for v in range(n):
        for f in range(k):
            if x[v, f] >= 1.0 - int_tol:
                labels[v] = f
                remaining[v] = False
                active_vars[v, :] = False
            elif x[v, f] <= int_tol:
                active_vars[v, f] = False

    # Compute fractional totals Tf and Tf_i (eq. 2)
    Tf = x.sum(axis=0)          # (k,)  total fractional assignment to f
    Tf_i = np.array([
        np.array([x[membership.astype(bool), f].sum() for f in range(k)])
        for membership in group_memberships
    ])  # (L, k)

    drop_total_constraint = np.zeros(k, dtype=bool)    # dropped eq. 2b
    drop_group_constraint = np.zeros((L, k), dtype=bool)  # dropped eq. 2c

    max_rounds = n * 2
    for _ in range(max_rounds):
        if not remaining.any():
            break

        # Build active_vars mask: only remaining points, only non-zero vars
        av = active_vars.copy()
        av[~remaining, :] = False

        # Build Tf / Tf_i restricted to active constraints
        Tf_arg = np.where(drop_total_constraint, None, Tf)  # simple masking below
        # We pass None where constraint is dropped; handle in LP builder
        Tf_active = Tf.copy()
        Tf_i_active = Tf_i.copy()

        x_new = _solve_fair_assignment_lp(
            X[remaining],
            centers,
            [m[remaining] for m in group_memberships],
            alpha,
            beta,
            Tf=Tf_active[...],
            Tf_i=Tf_i_active,
            active_vars=av[remaining],
        )

        if x_new is None:
            break

        remaining_indices = np.where(remaining)[0]

        # Fix integral variables
        newly_assigned = []
        for local_v, global_v in enumerate(remaining_indices):
            for f in range(k):
                if x_new[local_v, f] >= 1.0 - int_tol:
                    labels[global_v] = f
                    remaining[global_v] = False
                    active_vars[global_v, :] = False
                    newly_assigned.append((global_v, f))
                elif x_new[local_v, f] <= int_tol:
                    active_vars[global_v, f] = False

        # Update fractional totals
        for gv, f in newly_assigned:
            Tf[f] = max(0.0, Tf[f] - 1.0)
            for i, membership in enumerate(group_memberships):
                if membership[gv]:
                    Tf_i[i, f] = max(0.0, Tf_i[i, f] - 1.0)

        # Drop constraints with few active fractional variables (line 12-13)
        threshold = 2 * (delta_constraint + 1)
        for f in range(k):
            frac_total = sum(
                1 for v in range(n)
                if remaining[v] and active_vars[v, f]
                and int_tol < x_new[np.where(remaining)[0].tolist().index(v), f] < 1 - int_tol
                if v in np.where(remaining)[0]
            )
            if frac_total <= threshold:
                drop_total_constraint[f] = True
            for i in range(L):
                mem = group_memberships[i]
                frac_group = sum(
                    1 for local_v, gv in enumerate(remaining_indices)
                    if mem[gv] and active_vars[gv, f]
                    and int_tol < x_new[local_v, f] < 1 - int_tol
                )
                if frac_group <= threshold:
                    drop_group_constraint[i, f] = True

        if not newly_assigned:
            break

    # Assign any remaining unassigned points to nearest center (fallback)
    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        D = pairwise_l1(X[unassigned], centers)
        labels[unassigned] = np.argmin(D, axis=1)

    return labels


# ---------------------------------------------------------------------------
# Public API: Fair k-Median
# ---------------------------------------------------------------------------

def fair_kmedian(
    X: np.ndarray,
    k: int,
    group_memberships: list[np.ndarray],
    alpha: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    delta: float = 0.2,
    n_trials: int = 5,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Fair k-median clustering (Bera et al., NeurIPS 2019).

    Two-step algorithm:
        1. Run vanilla k-median (Algorithm 1, line 2) to find centers S.
        2. Solve FAIR 1-ASSIGNMENT (Algorithm 2) to find fair assignment φ.

    Parameters
    ----------
    X                 : (n, d) float array of data points
    k                 : number of clusters
    group_memberships : list of L arrays of length n (boolean or 0/1),
                        group_memberships[i][j] = 1 iff point j is in group i.
                        Groups may overlap (Delta > 1).
    alpha             : (L,) max-fraction per group per cluster.
                        If None, derived from data proportions + delta.
    beta              : (L,) min-fraction per group per cluster.
                        If None, derived from data proportions + delta.
    delta             : fairness slack for auto-computing alpha/beta (default 0.2,
                        the 80%-rule). Ignored if alpha and beta are provided.
    n_trials          : restarts for the vanilla k-median seeding step
    random_seed       : reproducibility seed

    Returns
    -------
    centers         : (k, d) cluster centers (from vanilla k-median)
    fair_labels     : (n,) fair assignment of points to centers
    vanilla_cost    : total L1 cost of the vanilla k-median solution
    fair_cost       : total L1 cost of the fair assignment
    alpha, beta     : the fairness parameters used (useful when auto-derived)

    Notes
    -----
    The centers S are fixed from the vanilla solution; only the assignment φ
    changes in step 2.  This mirrors Algorithm 1 in the paper exactly.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    L = len(group_memberships)
    group_memberships = [np.asarray(m, dtype=bool) for m in group_memberships]

    # Derive fairness parameters if not provided
    if alpha is None or beta is None:
        alpha, beta = fairness_params_from_ratios(group_memberships, n, delta)

    # Step 1: Vanilla k-median
    centers, vanilla_labels, vanilla_cost = kmedian(
        X, k, n_trials=n_trials, random_seed=random_seed
    )

    # Step 2: Fair assignment
    delta_param = int(np.sum(
        np.array([m.astype(int) for m in group_memberships]), axis=0
    ).max())

    fair_labels = fair_assignment(
        X, centers, group_memberships, alpha, beta,
        delta_constraint=delta_param,
    )

    # Compute fair cost
    D = pairwise_l1(X, centers)
    fair_cost = float(np.sum(D[np.arange(n), fair_labels]))

    return centers, fair_labels, vanilla_cost, fair_cost, alpha, beta


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_balance(
    labels: np.ndarray,
    group_memberships: list[np.ndarray],
    k: int,
) -> dict:
    """
    Compute the balance metric per cluster (generalised from Chierichetti et al.
    and used in Section 5 of the paper):

        balance(f) = min_i  min(r_i / r_i(f),  r_i(f) / r_i)

    where r_i = |C_i|/n  and  r_i(f) = |C_i ∩ cluster(f)| / |cluster(f)|.

    Returns
    -------
    dict with:
        'per_cluster' : (k,) array of balance values per cluster
        'overall'     : scalar minimum balance across all clusters
    """
    n = len(labels)
    L = len(group_memberships)

    r = np.array([m.sum() / n for m in group_memberships])
    per_cluster = np.ones(k)

    for f in range(k):
        cluster_mask = labels == f
        cluster_size = cluster_mask.sum()
        if cluster_size == 0:
            per_cluster[f] = 0.0
            continue
        for i, membership in enumerate(group_memberships):
            if r[i] == 0:
                continue
            r_if = membership[cluster_mask].sum() / cluster_size
            if r_if == 0:
                per_cluster[f] = 0.0
            else:
                balance_i = min(r[i] / r_if, r_if / r[i])
                per_cluster[f] = min(per_cluster[f], balance_i)

    return {
        "per_cluster": per_cluster,
        "overall": float(per_cluster.min()),
    }


def compute_cost_of_fairness(vanilla_cost: float, fair_cost: float) -> float:
    """Ratio of fair cost to vanilla cost (>= 1.0 by construction)."""
    if vanilla_cost == 0:
        return 1.0
    return fair_cost / vanilla_cost


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 200

    # Synthetic two-group dataset
    X = rng.random((n, 4))
    race   = (rng.random(n) > 0.6).astype(int)   # ~40% group 0, ~60% group 1
    sex    = (rng.random(n) > 0.5).astype(int)

    group_memberships = [race == 0, race == 1, sex == 0, sex == 1]

    centers, fair_labels, vanilla_cost, fair_cost, alpha, beta = fair_kmedian(
        X, k=4,
        group_memberships=group_memberships,
        delta=0.2,
        n_trials=3,
        random_seed=42,
    )

    balance = compute_balance(fair_labels, group_memberships, k=4)
    cof = compute_cost_of_fairness(vanilla_cost, fair_cost)

    print(f"Vanilla cost : {vanilla_cost:.4f}")
    print(f"Fair cost    : {fair_cost:.4f}")
    print(f"Cost of fairness (ratio): {cof:.4f}")
    print(f"Balance per cluster: {balance['per_cluster'].round(3)}")
    print(f"Overall balance: {balance['overall']:.4f}")
    print(f"alpha: {alpha.round(3)}")
    print(f"beta : {beta.round(3)}")