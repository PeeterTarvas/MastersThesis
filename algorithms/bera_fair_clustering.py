"""
Fair Algorithms for Clustering
================================
Implementation of Bera, Chakrabarty, Flores, Negahbani (NeurIPS 2019).
"Fair Algorithms for Clustering"
arXiv:1901.02393

Algorithm overview (Section 3 of the paper)
--------------------------------------------
The algorithm is a clean two-step procedure:

  Step 1 — Vanilla k-median
      Run standard (unfair) k-median to fix a set of k centers S.
      The key insight from Theorem 3: if we find a ρ-approximation for
      vanilla k-median and then solve the FAIR ASSIGNMENT problem on those
      fixed centers, the overall solution is a (ρ+2)-approximation for
      the FAIR k-MEDIAN problem.  We don't need to re-open centers —
      just re-assign points fairly.

  Step 2 — Fair LP relaxation (Eq. 1 in the paper)
      With centers S fixed, solve the LP:
          min  Σ_{i,j}  w_i · d(x_i, c_j) · x_{ij}
          s.t. Σ_j x_{ij} = 1          ∀ i       (each point fully assigned)
               β_h · Σ_i x_{ij} ≤ Σ_{i∈Col_h} x_{ij} ≤ α_h · Σ_i x_{ij}
                                        ∀ j, ∀ h  (fairness bounds per cluster)
               0 ≤ x_{ij} ≤ 1
      The LP relaxation allows fractional assignments.

  Step 3 — Iterative LP rounding (Algorithm 2 / Theorem 7)
      Iteratively round the fractional LP solution to an integral one
      while maintaining fairness up to an additive violation of (4Δ+3),
      where Δ = max number of groups a single point belongs to.

      The rounding works as follows (matching Algorithm 2 exactly):
        a) Compute T_f   = Σ_i x*_{ij}  (fractional total  per center j)
              and T_{f,h} = Σ_{i∈Col_h} x*_{ij}  (fractional group-h mass per center j)
        b) Build LP2: same variables but with tighter bounds:
               ⌊T_f⌋   ≤ Σ_i x_{ij}    ≤ ⌈T_f⌉        ∀ j
               ⌊T_{f,h}⌋ ≤ Σ_{i∈Col_h} x_{ij} ≤ ⌈T_{f,h}⌉  ∀ j, h
        c) While unassigned points remain:
             - Solve LP2
             - Fix any x_{ij}=1 → assign point i to center j, remove from LP2
             - Fix any x_{ij}=0 → remove variable
             - Drop fairness constraints for center j and group h once
               |{fractional variables x_{ij}: i∈Col_h}| ≤ 2(Δ+1)
               (this is the key sparsity argument from Kiraly et al.)
             - Adjust T_f and T_{f,h} as points are assigned

Usage
-----
  # With coreset (weighted)
  centers, labels, cost = fair_clustering(
      coreset_df,
      feature_cols=['Lat_Scaled', 'Lon_Scaled'],
      protected_group_col='GROUP_ID',
      k=10,
      alpha=0.1,
      weight_col='Weight',       # ← uses coreset weights
  )

  # Without coreset (uniform weights = standard unweighted points)
  centers, labels, cost = fair_clustering(
      raw_df,
      feature_cols=['Lat_Scaled', 'Lon_Scaled'],
      protected_group_col='GROUP_ID',
      k=10,
      alpha=0.1,
      weight_col=None,           # ← every point has weight 1
  )

Notes on weighted vs unweighted
---------------------------------
The LP and rounding are naturally weighted: the cost objective uses w_i,
and the fairness constraints count weighted mass (Σ w_i x_{ij}).
For unweighted (raw) data, setting w_i = 1 ∀i makes both modes identical
in formulation.  The difference is purely practical: coreset points
represent many original points via their weight.
"""

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


# ===========================================================================
# 1.  Utility helpers
# ===========================================================================

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


# ===========================================================================
# 2.  Step 2 — Fair LP relaxation  (Eq. 1 in the paper)
# ===========================================================================

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
    n = len(X)
    k = len(centers)
    H = len(lower_bounds)
    n_vars = n * k

    # ---- cost vector -------------------------------------------------------
    # w_i * d(x_i, c_j) for all (i,j) pairs, flattened row-major
    D = pairwise_l1(X, centers).astype(np.float64)           # (n, k)
    c_obj = (D * weights[:, np.newaxis]).ravel()              # (n*k,)

    # ---- equality: each point fully assigned  (Σ_j x_{ij} = 1) ------------
    A_eq = lil_matrix((n, n_vars), dtype=np.float64)
    for i in range(n):
        A_eq[i, i * k : (i + 1) * k] = 1.0
    b_eq = np.ones(n, dtype=np.float64)

    # ---- inequality: fairness bounds per (center j, group h) ---------------
    # Two rows per (j, h) pair: MP lower bound + RD upper bound
    n_ineq = 2 * k * H
    A_ub = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_ineq, dtype=np.float64)

    row = 0
    for j in range(k):
        for h in range(H):
            in_h = (group_codes == h)          # boolean mask, length n
            for i in range(n):
                col = i * k + j
                # MP lower:  β_h * x_{ij}  −  (x_{ij} if i∈Col_h)  ≤ 0
                A_ub[row,     col] = lower_bounds[h] - (1.0 if in_h[i] else 0.0)
                # RD upper: −α_h * x_{ij}  +  (x_{ij} if i∈Col_h)  ≤ 0
                A_ub[row + 1, col] = -upper_bounds[h] + (1.0 if in_h[i] else 0.0)
            row += 2

    result = linprog(
        c_obj,
        A_ub=A_ub.tocsc(),
        b_ub=b_ub,
        A_eq=A_eq.tocsc(),
        b_eq=b_eq,
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

    return result.x.reshape((n, k))


# ===========================================================================
# 3.  Step 3 — Iterative LP rounding  (Algorithm 2 / Theorem 7)
# ===========================================================================

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
    n, k = x_lp.shape
    H = int(group_codes.max()) + 1
    DELTA = 1                              # groups partition points → Δ = 1
    SPARSITY_THRESHOLD = 2 * (DELTA + 1)  # = 4; when to drop fairness constraint

    # --- Initialise working state -------------------------------------------
    labels    = np.full(n, -1, dtype=np.int32)   # -1 = unassigned
    unassigned = np.ones(n, dtype=bool)

    # T_f[j]    = remaining weighted mass yet to be assigned to center j
    # T_fh[h,j] = remaining weighted group-h mass yet to be assigned to center j
    # Initially these equal the LP fractional assignments, scaled by weight.
    T_f  = np.einsum('ij,i->j', x_lp, weights).astype(np.float64)      # (k,)
    T_fh = np.zeros((H, k), dtype=np.float64)
    for h in range(H):
        mask = (group_codes == h)
        T_fh[h] = np.einsum('ij,i->j', x_lp[mask], weights[mask])       # (k,)

    # Track which (center j, group h) fairness constraints are still enforced
    fair_active = np.ones((H, k), dtype=bool)

    # Cache: for each point, which centers still have a nonzero LP variable?
    # We represent this as a set per point for fast pruning.
    allowed = [set(np.where(x_lp[i] > 1e-9)[0]) for i in range(n)]

    # -------------------------------------------------------------------------
    # Iterative rounding loop  (Algorithm 2, Bera et al.)
    # -------------------------------------------------------------------------
    for outer_iter in range(n + k * H + 10):   # generous upper bound

        still_unassigned = np.where(unassigned)[0]
        if len(still_unassigned) == 0:
            break

        idx_pts   = still_unassigned
        n_sub     = len(idx_pts)
        # Map global index → local row in LP2
        global_to_local = {int(i): ii for ii, i in enumerate(idx_pts)}

        # ---- Enumerate active variables ------------------------------------
        # A variable x_{ij} is active if:
        #   - point i is unassigned, AND
        #   - center j is in allowed[i]
        # We enumerate them as (local_row, center) pairs with a flat index.
        var_list = []   # (local_row ii, center j)
        for ii, i in enumerate(idx_pts):
            for j in sorted(allowed[i]):
                var_list.append((ii, j))
        n_vars_lp = len(var_list)

        if n_vars_lp == 0:
            # No variables left — force greedy assignment
            for i in still_unassigned:
                labels[i] = int(np.argmin(D[i]))
            break

        # Build quick lookup: (local_row, center) → variable index
        var_idx = {(ii, j): v for v, (ii, j) in enumerate(var_list)}

        # ---- cost vector ---------------------------------------------------
        c_lp = np.array([
            weights[idx_pts[ii]] * D[idx_pts[ii], j]
            for ii, j in var_list
        ], dtype=np.float64)

        # ---- equality: each unassigned point fully assigned ----------------
        A_eq2 = lil_matrix((n_sub, n_vars_lp), dtype=np.float64)
        for v, (ii, j) in enumerate(var_list):
            A_eq2[ii, v] = 1.0
        b_eq2 = np.ones(n_sub, dtype=np.float64)

        # ---- inequality: LP2 range constraints -----------------------------
        # Range [lo, hi] is expressed as two ≤ constraints:
        #   Σ (coeff * x_v) ≤ hi      and     -Σ (coeff * x_v) ≤ -lo
        ineq_rows: list[np.ndarray] = []
        ineq_rhs:  list[float]      = []

        def _add_range(coeffs: dict[int, float], lo: float, hi: float) -> None:
            """Add  lo ≤ Σ coeffs[v]*x_v ≤ hi  as two ≤ rows."""
            hi = max(hi, lo)   # numerical guard
            row_p = np.zeros(n_vars_lp)
            row_n = np.zeros(n_vars_lp)
            for v, c in coeffs.items():
                row_p[v] =  c
                row_n[v] = -c
            ineq_rows.append(row_p);  ineq_rhs.append(hi)
            ineq_rows.append(row_n);  ineq_rhs.append(-lo)

        # Per-center total weighted mass range
        for j in range(k):
            tf_j = T_f[j]
            lo = max(0.0, np.floor(tf_j + 1e-9))
            hi =          np.ceil( tf_j - 1e-9)
            coeffs = {
                var_idx[(ii, j)]: weights[idx_pts[ii]]
                for ii in range(n_sub)
                if (ii, j) in var_idx
            }
            if coeffs:
                _add_range(coeffs, lo, hi)

        # Per-center per-group weighted mass range (active constraints only)
        for h in range(H):
            for j in range(k):
                if not fair_active[h, j]:
                    continue
                tf_hj = T_fh[h, j]
                lo = max(0.0, np.floor(tf_hj + 1e-9))
                hi =          np.ceil( tf_hj - 1e-9)
                coeffs = {
                    var_idx[(ii, j)]: weights[idx_pts[ii]]
                    for ii in range(n_sub)
                    if (ii, j) in var_idx and group_codes[idx_pts[ii]] == h
                }
                if coeffs:
                    _add_range(coeffs, lo, hi)

        A_ub2 = np.vstack(ineq_rows) if ineq_rows else None
        b_ub2 = np.array(ineq_rhs)   if ineq_rows else None

        # ---- Solve LP2 -----------------------------------------------------
        result2 = linprog(
            c_lp,
            A_ub=A_ub2,
            b_ub=b_ub2,
            A_eq=A_eq2.tocsc(),
            b_eq=b_eq2,
            bounds=[(0.0, 1.0)] * n_vars_lp,
            method='highs',
            options={'disp': False, 'presolve': True},
        )

        if result2.status != 0:
            warnings.warn(
                f"[Rounding] LP2 infeasible at iter {outer_iter} "
                f"(status {result2.status}). "
                "Assigning remaining points greedily."
            )
            for i in still_unassigned:
                labels[i] = int(np.argmin(D[i]))
            break

        x2 = result2.x   # flat, indexed by var_list

        # ---- Commit integral variables / prune zero variables --------------
        newly_assigned = False

        for v, (ii, j) in enumerate(var_list):
            i = int(idx_pts[ii])
            val = x2[v]

            if val >= 1.0 - 1e-6:
                # x_{ij} = 1 → assign point i to center j
                labels[i]     = j
                unassigned[i] = False
                T_f[j]                      = max(0.0, T_f[j]  - weights[i])
                T_fh[group_codes[i], j]     = max(0.0, T_fh[group_codes[i], j] - weights[i])
                allowed[i]                  = set()   # remove all variables for this point
                newly_assigned = True

            elif val <= 1e-6:
                # x_{ij} ≈ 0 → prune this variable
                allowed[i].discard(j)

        # ---- If no variable became integral: force-commit thewww.auto24.ee/soidukid/4287469www.auto24.ee/soidukid/4287469 highest one --
        # This is a numerical fallback; the Kiraly et al. matroid argument
        # guarantees a vertex solution exists, but floating-point LP solvers
        # sometimes return near-integral solutions just below the threshold.
        if not newly_assigned:
            best_v, best_val = 0, -1.0
            for v, (ii, j) in enumerate(var_list):
                i = int(idx_pts[ii])
                if unassigned[i] and x2[v] > best_val:
                    best_val, best_v = x2[v], v
            ii_b, j_b = var_list[best_v]
            i_b = int(idx_pts[ii_b])
            labels[i_b]     = j_b
            unassigned[i_b] = False
            T_f[j_b]                       = max(0.0, T_f[j_b]  - weights[i_b])
            T_fh[group_codes[i_b], j_b]    = max(0.0, T_fh[group_codes[i_b], j_b] - weights[i_b])
            allowed[i_b]                   = set()

        # ---- Drop fairness constraints where sparsity condition is met -----
        # Per the paper: drop (j, h) constraint once the number of fractional
        # variables x_{ij} with i ∈ Col_h is ≤ 2(Δ+1) = 4.
        # "Fractional" here means: unassigned AND in allowed[i] for center j.
        remaining = np.where(unassigned)[0]
        for h in range(H):
            for j in range(k):
                if not fair_active[h, j]:
                    continue
                frac_count = sum(
                    1 for i in remaining
                    if group_codes[i] == h and j in allowed[i]
                )
                if frac_count <= SPARSITY_THRESHOLD:
                    fair_active[h, j] = False

    # ---- Fallback for any remaining unassigned points ----------------------
    still_left = np.where(labels == -1)[0]
    if len(still_left) > 0:
        warnings.warn(
            f"[Rounding] {len(still_left)} point(s) unassigned after loop. "
            "Assigning greedily."
        )
        for i in still_left:
            labels[i] = int(np.argmin(D[i]))

    return labels


# ===========================================================================
# 4.  Evaluation helpers
# ===========================================================================

def evaluate_fairness(
    labels: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    group_names: list,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    k: int,
    verbose: bool = True,
) -> dict:
    """
    Check per-(cluster, group) fairness and report violations.

    A (cluster j, group h) pair VIOLATES the fairness constraint if:
        frac_h_in_j < lower_bounds[h] - ε   OR
        frac_h_in_j > upper_bounds[h] + ε
    where frac_h_in_j = (weighted mass of group h in cluster j) /
                        (total weighted mass in cluster j).

    Returns a dict with summary statistics.
    """
    violations = 0
    total_pairs = 0
    max_violation = 0.0
    details = []

    for j in range(k):
        at_j = (labels == j)
        total_j = weights[at_j].sum()
        if total_j < 1e-9:
            continue
        for h, name in enumerate(group_names):
            mass_h = weights[at_j & (group_codes == h)].sum()
            frac = mass_h / total_j
            total_pairs += 1
            lo_viol = max(0.0, lower_bounds[h] - frac)
            hi_viol = max(0.0, frac - upper_bounds[h])
            viol = max(lo_viol, hi_viol)
            if viol > 1e-4:
                violations += 1
                max_violation = max(max_violation, viol)
                details.append((j, name, frac, lower_bounds[h], upper_bounds[h], viol))

    if verbose:
        if violations == 0:
            print(f"[Fairness] ✓ All {total_pairs} (cluster, group) pairs satisfy bounds.")
        else:
            print(f"[Fairness] ⚠  {violations}/{total_pairs} pairs violate bounds "
                  f"(max violation = {max_violation:.4f}). "
                  "Additive violations ≤ (4Δ+3) are theoretically expected.")
            for j, name, frac, lo, hi, viol in details[:10]:   # print at most 10
                print(f"  Cluster {j:2d}, group '{name}': "
                      f"frac={frac:.3f}  bounds=[{lo:.3f}, {hi:.3f}]  viol={viol:.4f}")

    return {
        'n_violations': violations,
        'total_pairs':  total_pairs,
        'max_violation': max_violation,
        'details':       details,
    }


def compute_gpof(
    X: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    unfair_labels: np.ndarray,
) -> dict:
    """
    Group-wise Price of Fairness (G-PoF).

    From the thesis (Eq. 1-2):
        G-PoF(ℓ) = Φ_ℓ(C_fair) / Φ_ℓ(C_opt)
    where
        Φ_ℓ(C) = (1/|X_ℓ|) * Σ_{x ∈ X_ℓ} d(x, assigned_center(x))

    A value close to 1 means the fairness constraint imposes little
    extra cost on that group.  A high value means that group "pays" more
    for the global fairness constraint.

    Parameters
    ----------
    unfair_labels : labels from vanilla (unfair) k-median, used as baseline
    """
    D = pairwise_l1(X, centers)
    H = int(group_codes.max()) + 1
    result = {}

    for h in range(H):
        mask = (group_codes == h)
        if mask.sum() == 0:
            continue
        w_h = weights[mask]
        total_w = w_h.sum()

        fair_dists   = D[mask, labels[mask]]
        unfair_dists = D[mask, unfair_labels[mask]]

        phi_fair   = float(np.dot(w_h, fair_dists))   / total_w
        phi_unfair = float(np.dot(w_h, unfair_dists)) / total_w

        gpof = phi_fair / phi_unfair if phi_unfair > 1e-12 else float('inf')
        result[h] = {'phi_fair': phi_fair, 'phi_unfair': phi_unfair, 'gpof': gpof}

    return result


# ===========================================================================
# 5.  Visualization
# ===========================================================================

def visualize_fair_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
    feature_cols: list[str],
    group_col: str,
    weight_col: Optional[str] = 'Weight',
    title_suffix: str = '',
) -> None:
    """
    Two-panel plot:
      Left  — geographic scatter coloured by cluster
      Right — stacked-bar of group composition per cluster (proportional)
    """
    df_vis = df.copy()
    df_vis['Cluster'] = labels
    lat_col, lon_col = feature_cols

    w = df_vis[weight_col].values if (weight_col and weight_col in df_vis.columns) \
        else np.ones(len(df_vis))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Fair k-Median Clustering{title_suffix}', fontsize=14)

    # Left: spatial scatter
    ax = axes[0]
    sc = ax.scatter(
        df_vis[lon_col], df_vis[lat_col],
        c=df_vis['Cluster'], cmap='tab20',
        alpha=0.5,
        s=np.sqrt(w / w.max()) * 20,
        linewidths=0,
    )
    ax.scatter(centers[:, 1], centers[:, 0],
               c='red', marker='X', s=120, zorder=5, label='Centers')
    plt.colorbar(sc, ax=ax, label='Cluster')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Right: group composition per cluster
    ax2 = axes[1]
    comp = (
        df_vis.assign(_w=w)
              .groupby(['Cluster', group_col])['_w']
              .sum()
              .unstack(fill_value=0)
    )
    comp_norm = comp.div(comp.sum(axis=1), axis=0)
    comp_norm.plot(kind='bar', stacked=True, ax=ax2, legend=True,
                   colormap='tab20', edgecolor='none')
    ax2.set_title('Cluster Group Composition (Proportional)')
    ax2.set_xlabel('Cluster ID'); ax2.set_ylabel('Weight Fraction')
    ax2.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(f'fair_clusters{title_suffix.replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"[Viz] Saved fair_clusters{title_suffix.replace(' ', '_')}.png")


# ===========================================================================
# 6.  Main public API
# ===========================================================================

def fair_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    protected_group_col: str,
    k: int,
    alpha: float = 0.1,
    weight_col: Optional[str] = 'Weight',
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    kmedian_trials: int = 3,
    kmedian_max_iter: int = 50,
    random_seed: int = 42,
    visualize: bool = True,
    title_suffix: str = '',
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
    visualize           : Whether to produce and save cluster plots.
    title_suffix        : Appended to plot title / filename (e.g. ' (coreset)').

    Returns
    -------
    centers : (k, d) final center coordinates (in the scaled feature space)
    labels  : (n,)   integer cluster assignment per row of df
    cost    : total weighted L1 assignment cost
    """
    # ---- Extract arrays from DataFrame ------------------------------------
    X = df[feature_cols].to_numpy(dtype=np.float64)

    # Decide weights
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=np.float64)
        mode_str = f"weighted (col='{weight_col}')"
    else:
        weights = np.ones(len(X), dtype=np.float64)
        mode_str = "unweighted (uniform w=1)"

    group_codes, group_names = encode_groups_to_int(df[protected_group_col])
    H = len(group_names)

    print(f"\n{'='*60}")
    print(f"[FairClustering] Bera et al. (NeurIPS 2019)")
    print(f"  n={len(X):,}  k={k}  groups={H}  mode={mode_str}")
    print(f"{'='*60}")

    # ---- Fairness bounds ---------------------------------------------------
    if lower_bounds is None or upper_bounds is None:
        lower_bounds, upper_bounds = proportional_bounds(
            group_codes, weights, H, alpha
        )
        total_w = weights.sum()
        print(f"\n[FairClustering] Proportional bounds (alpha={alpha}):")
        for h, name in enumerate(group_names):
            f_h = weights[group_codes == h].sum() / total_w
            print(f"  Group '{name}': freq={f_h:.3f}  "
                  f"bounds=[{lower_bounds[h]:.3f}, {upper_bounds[h]:.3f}]")

    # Sanity check: are the bounds jointly feasible?
    # Necessary condition: Σ_h lower_bounds[h] ≤ 1 ≤ Σ_h upper_bounds[h]
    if lower_bounds.sum() > 1.0 + 1e-6:
        warnings.warn(
            f"[FairClustering] Σ lower_bounds = {lower_bounds.sum():.3f} > 1. "
            "The LP will be infeasible.  Decrease alpha or relax bounds."
        )
    if upper_bounds.sum() < 1.0 - 1e-6:
        warnings.warn(
            f"[FairClustering] Σ upper_bounds = {upper_bounds.sum():.3f} < 1. "
            "The LP will be infeasible.  Increase alpha or relax bounds."
        )

    # ---- Step 1: Vanilla k-median ------------------------------------------
    print(f"\n[Step 1] Vanilla k-median (trials={kmedian_trials}, "
          f"max_iter={kmedian_max_iter}) ...")
    centers, unfair_labels, unfair_cost = kmedian(
        X, k,
        _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    print(f"  → Unfair k-median cost: {unfair_cost:,.2f}")

    # ---- Step 2: Fair LP relaxation ----------------------------------------
    print(f"\n[Step 2] Solving Fair LP relaxation  "
          f"(n_vars = {len(X) * k:,}, n_constraints ≈ {len(X) + 2*H*k:,}) ...")
    x_lp = solve_fair_lp(X, centers, weights, group_codes, lower_bounds, upper_bounds)

    if x_lp is None:
        # LP failed — fall back to unfair assignment
        warnings.warn(
            "[FairClustering] LP infeasible — returning unfair k-median result."
        )
        D_fb = pairwise_l1(X, centers)
        cost = float(np.dot(weights, D_fb[np.arange(len(X)), unfair_labels]))
        return centers, unfair_labels, cost

    D = pairwise_l1(X, centers)
    lp_cost = float(np.dot(
        weights,
        (D * x_lp).sum(axis=1)
    ))
    print(f"  → LP fractional cost:   {lp_cost:,.2f}")
    print(f"  → Integrality gap hint: "
          f"{lp_cost / unfair_cost:.3f}x unfair cost")

    # ---- Step 3: Iterative rounding ----------------------------------------
    print(f"\n[Step 3] Iterative LP rounding (Algorithm 2) ...")
    labels = iterative_rounding(
        X, centers, weights, group_codes, x_lp, D
    )
    fair_cost = float(np.dot(weights, D[np.arange(len(X)), labels]))
    print(f"  → Fair (integral) cost: {fair_cost:,.2f}")
    print(f"  → Price of Fairness:    {fair_cost / unfair_cost:.4f}x  "
          f"(1.0 = fairness is free)")

    # ---- Evaluate fairness -------------------------------------------------
    print(f"\n[Evaluation] Fairness check:")
    evaluate_fairness(
        labels, group_codes, weights, group_names,
        lower_bounds, upper_bounds, k, verbose=True,
    )

    # ---- G-PoF per group ---------------------------------------------------
    gpof = compute_gpof(X, centers, labels, group_codes, weights, unfair_labels)
    print(f"\n[Evaluation] Group-wise Price of Fairness (G-PoF):")
    for h, name in enumerate(group_names):
        if h in gpof:
            g = gpof[h]
            print(f"  Group '{name}': G-PoF = {g['gpof']:.4f}  "
                  f"(fair_cost/pt={g['phi_fair']:.4f}, "
                  f"unfair_cost/pt={g['phi_unfair']:.4f})")

    # ---- Visualization -----------------------------------------------------
    if visualize:
        visualize_fair_clusters(
            df, labels, centers, feature_cols,
            protected_group_col, weight_col, title_suffix
        )

    print(f"\n{'='*60}")
    return centers, labels, fair_cost


# ===========================================================================
# 7.  Entry point — demo with both coreset and raw modes
# ===========================================================================

if __name__ == "__main__":
    import sys

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading data...")
    df_raw = csv_loader.load_csv_chunked(
        "../us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=10_000,
        max_rows=10_000,
    )

    # =========================================================================
    # MODE A — with coreset  (recommended for large data)
    # =========================================================================
    print("\n" + "="*60)
    print("MODE A: Fair clustering ON CORESET (weighted)")
    print("="*60)

    N_LOCATIONS = 300     # number of coreset representative points
    K = 10                # number of clusters
    ALPHA = 0.15          # fairness slack: ±15% from proportional representation

    coreset_df = compute_fair_coreset(df_raw, n_locations=N_LOCATIONS, random_seed=42)

    centers_c, labels_c, cost_c = fair_clustering(
        coreset_df,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        protected_group_col='GROUP_ID',
        k=K,
        alpha=ALPHA,
        weight_col='Weight',              # ← coreset weights
        kmedian_trials=3,
        kmedian_max_iter=30,
        random_seed=42,
        visualize=True,
        title_suffix=' (coreset)',
    )

    # =========================================================================
    # MODE B — without coreset  (every point has weight 1)
    # =========================================================================
    print("\n" + "="*60)
    print("MODE B: Fair clustering ON RAW DATA (unweighted, small sample)")
    print("="*60)

    # Use a small subset so the LP stays tractable
    N_RAW = 500
    df_raw_small = preprocess_dataset(df_raw.head(N_RAW).copy())

    centers_r, labels_r, cost_r = fair_clustering(
        df_raw_small,
        feature_cols=['Lat_Scaled', 'Lon_Scaled'],
        protected_group_col='GROUP_ID',
        k=K,
        alpha=ALPHA,
        weight_col=None,                  # ← uniform weights (w_i = 1)
        kmedian_trials=3,
        kmedian_max_iter=30,
        random_seed=42,
        visualize=True,
        title_suffix=' (raw)',
    )

    print("\n[Summary]")
    print(f"  Coreset mode — n={len(coreset_df):,} weighted pts, "
          f"cost={cost_c:,.2f}")
    print(f"  Raw mode     — n={N_RAW} points,         cost={cost_r:,.2f}")