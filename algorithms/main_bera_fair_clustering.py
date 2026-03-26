from __future__ import annotations

import warnings
from typing import Optional, Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from scipy.optimize import linprog
from scipy.sparse import kron, eye, lil_matrix
import time
from evaluate import plot_execution_times, make_result, audit_fairness_proportional, evaluate, \
    plot_spatial_clusters, plot_cluster_pof, plot_pof_comparison, plot_group_pof, plot_cost_breakdown
from kmedian import kmedian, pairwise_l1
import csv_loader


def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    cats = pd.Categorical(group_series)
    return cats.codes.astype(np.int32), list(cats.categories)


def proportional_bounds(
    group_codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    dataset_len = len(X)
    nr_of_centers = len(centers)
    nr_of_groups = len(lower_bounds)
    n_vars = dataset_len * nr_of_centers

    distances_to_centers = pairwise_l1(X, centers).astype(np.float64)
    cost_to_center = (distances_to_centers * weights[:, np.newaxis]).ravel()

    a_equality = kron(eye(dataset_len), np.ones((1, nr_of_centers)), format='csr')
    b_equality = np.ones(dataset_len, dtype=np.float64)

    n_ineq = 2 * nr_of_groups * nr_of_centers
    a_upper_bound = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_bound = np.zeros(n_ineq)

    row = 0
    for center_index in range(nr_of_centers):
        for group_index in range(nr_of_groups):
            in_bound = (group_codes == group_index)
            cols = np.arange(dataset_len) * nr_of_centers + center_index

            a_upper_bound[row, cols] = lower_bounds[group_index] - in_bound
            a_upper_bound[row + 1, cols] = in_bound - upper_bounds[group_index]
            row += 2

    result = linprog(
        cost_to_center,
        A_ub=a_upper_bound.tocsc(),
        b_ub=b_bound,
        A_eq=a_equality.tocsc(),
        b_eq=b_equality,
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
    weights: np.ndarray,
    group_codes: np.ndarray,
    x_lp: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    dataset_len, nr_of_centers = x_lp.shape
    max_group_code = int(group_codes.max()) + 1
    DELTA = 1
    SPARSITY_THRESHOLD = 2 * (DELTA + 1)

    labels = np.full(dataset_len, -1, dtype=np.int32)
    unassigned = np.ones(dataset_len, dtype=bool)

    weighted_mass = np.einsum('ij,i->j', x_lp, weights).astype(np.float64)
    group_weighted_mass = np.array([
        np.einsum('ij,i->j', x_lp[group_codes == group_code], weights[group_codes == group_code])
        for group_code in range(max_group_code)
    ])

    fair_active = np.ones((max_group_code, nr_of_centers), dtype=bool)
    # cache for each point, which centers still have a nonzero LP variable?
    allowed = [set(np.where(x_lp[i] > 1e-9)[0]) for i in range(dataset_len)]

    iteration_amount = (dataset_len + nr_of_centers) * max_group_code + 10
    for iter in range(iteration_amount):
        still_unassigned = np.where(unassigned)[0]
        if len(still_unassigned) == 0:
            break
        nr_unassigned = len(still_unassigned)

        # enumerate active variables
        # variable x_{ij} is active if:
        #   - point i is unassigned
        #   - center j is in allowed[i]
        # enumerate them as (local_row, center) pairs with a flat index
        var_list = [(idx, j) for idx, unassigned in enumerate(still_unassigned) for j in sorted(allowed[unassigned])]
        nr_vars_lp = len(var_list)

        if nr_vars_lp == 0: # No variables left — force greedy assignment
            for i in still_unassigned:
                labels[i] = int(np.argmin(D[i]))
            break

        quick_lookup_idx = {(unassigned_point_enum, j): enum for enum, (unassigned_point_enum, j) in enumerate(var_list)}
        cost_vector_lp = np.array([
            weights[still_unassigned[unassigned_point_enum]] * D[still_unassigned[unassigned_point_enum], j]
            for unassigned_point_enum, j in var_list
        ], dtype=np.float64)

        a_equality = lil_matrix((nr_unassigned, nr_vars_lp), dtype=np.float64)
        for enum, (unassigned_point_enum, j) in enumerate(var_list):
            a_equality[unassigned_point_enum, enum] = 1.0
        b_equality = np.ones(nr_unassigned, dtype=np.float64)

        ineq_rows, ineq_rhs = [], []

        def _add_range(coeffs: dict[int, float], low: float, high: float) -> None:
            row_p, row_n = np.zeros(nr_vars_lp), np.zeros(nr_vars_lp)
            high = max(high, low)
            for value, coefficient in coeffs.items():
                row_p[value], row_n[value] = coefficient, -coefficient
            ineq_rows.extend([row_p, row_n])
            ineq_rhs.extend([high, -low])

        # Per-group, per-center weighted mass range
        for group_code_idx in range(max_group_code):
            for col in range(nr_of_centers):
                if not fair_active[group_code_idx, col]:
                    continue
                group_weighted_mass_col = group_weighted_mass[group_code_idx, col]
                low = max(0.0, np.floor(group_weighted_mass_col + 1e-9))
                high = np.ceil(group_weighted_mass_col - 1e-9)
                coeffs = {
                    quick_lookup_idx[(nr_unassigned_idx, col)]: weights[still_unassigned[nr_unassigned_idx]]
                    for nr_unassigned_idx in range(nr_unassigned) if group_codes[still_unassigned[nr_unassigned_idx]] == group_code_idx
                                                                     and (nr_unassigned_idx, col) in quick_lookup_idx
                }
                if coeffs:
                    _add_range(coeffs, low, high)

        a_upperbound = np.vstack(ineq_rows) if ineq_rows else None
        b_upperbound = np.array(ineq_rhs) if ineq_rows else None


        result = linprog(
            cost_vector_lp,
            A_ub=a_upperbound,
            b_ub=b_upperbound,
            A_eq=a_equality.tocsc(),
            b_eq=b_equality,
            bounds=[(0.0, 1.0)] * nr_vars_lp,
            method='highs',
            options={'disp': False, 'presolve': True},
        )

        if result.status != 0:
            print(
                f"[Rounding] LP2 infeasible at iter {iter} "
                f"(status {result.status}). "
                "Assigning remaining points greedily."
            )
            for i in still_unassigned:
                labels[i] = int(np.argmin(D[i]))
            break

        _result = result.x
        best_v, best_val = 0, -1
        newly_assigned = False
        # commit integral variables / prune zero variables
        for enum, (unassigned_point_enum, j) in enumerate(var_list):
            i = int(still_unassigned[unassigned_point_enum])
            val = _result[enum]
            if unassigned[i] and val > best_val:
                best_val, best_v = val, enum
            if val >= 1.0 - 1e-6:
                # x_{ij} = 1 → assign point i to center j
                labels[i] = j
                unassigned[i] = False
                weighted_mass[j] = max(0.0, weighted_mass[j] - weights[i])
                group_weighted_mass[group_codes[i], j] = max(0.0, group_weighted_mass[group_codes[i], j] - weights[i])
                allowed[i] = set()  # remove all variables for this point
                newly_assigned = True
            elif val <= 1e-6:
                # x_{ij} ≈ 0 → prune this variable
                allowed[i].discard(j)

        # if no variable became integral: force-commit the highest one
        # numerical fallback
        # guarantees a vertex solution exists, but floating-point LP solvers
        # sometimes return near-integral solutions just below the threshold.
        if not newly_assigned:
            idx_b, j_b = var_list[best_v]
            i_b = int(still_unassigned[idx_b])
            labels[i_b] = j_b
            unassigned[i_b] = False
            weighted_mass[j_b] = max(0.0, weighted_mass[j_b] - weights[i_b])
            group_weighted_mass[group_codes[i_b], j_b] = max(0.0, group_weighted_mass[group_codes[i_b], j_b] - weights[i_b])
            allowed[i_b] = set()

        # drop fairness constraints where sparsity condition is met
        # drop (j, h) constraint once the number of fractional
        for h in range(max_group_code):
            for j in range(nr_of_centers):
                if fair_active[h, j]:
                    frac_count = sum(1 for i in np.where(unassigned)[0] if group_codes[i] == h and j in allowed[i])
                    if frac_count <= SPARSITY_THRESHOLD:
                        fair_active[h, j] = False
        # Fallback for any remaining unassigned points
        for i in np.where(labels == -1)[0]:
            labels[i] = int(np.argmin(D[i]))

    return labels

def audit_fairness(
    labels: np.ndarray,
    group_codes: np.ndarray,
    weights: np.ndarray,
    group_labels: list,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    k: int,
):
    violations = 0
    for j in range(k):
        at_j = labels == j
        total_j = weights[at_j].sum()
        if total_j == 0:
            continue
        for h, lbl in enumerate(group_labels):
            mass_h = weights[at_j & (group_codes == h)].sum()
            frac = mass_h / total_j
            if frac < lower_bounds[h] - 1e-4 or frac > upper_bounds[h] + 1e-4:
                violations += 1

    if violations == 0:
        print("[FairClustering] ✓ All clusters satisfy fairness bounds.")
    else:
        print(f"[FairClustering] ⚠  {violations} (cluster, group) pairs violate bounds "
              "(additive violations ≤ 1 are expected by Lemma 7).")


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
) -> tuple[
         ndarray, ndarray, float, ndarray, float, dict[Any, Any], Any, ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
             Any, dtype[Any]] | Any, ndarray, list, ndarray, ndarray] | None:
    timing = {}
    t_start = time.perf_counter()

    t_start_prep = time.perf_counter()
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
    timing['Data Preparation'] = time.perf_counter() - t_start_prep

    t_start_kmedian = time.perf_counter()
    print(f"\nVanilla k-median (trials={kmedian_trials}, "
          f"max_iter={kmedian_max_iter}) ...")
    unfair_centers, unfair_labels, unfair_cost = kmedian(
        x, k_centers,
        _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    timing['Vanilla K-Median'] = time.perf_counter() - t_start_kmedian
    print(f"  → Unfair k-median cost: {unfair_cost:,.2f}")

    t_start_lp = time.perf_counter()
    print(f"\nSolving Fair LP relaxation  "
          f"(n_vars = {len(x) * k_centers:,}, n_constraints ≈ {len(x) + 2*nr_of_groups*k_centers:,}) ...")
    x_lp = solve_fair_lp(x, unfair_centers, weights, group_codes, lower_bounds, upper_bounds)
    timing['Solve Initial LP'] = time.perf_counter() - t_start_lp

    t_start_rounding = time.perf_counter()
    distsances_to_centers = pairwise_l1(x, unfair_centers)
    lp_cost = float(np.dot(
        weights,
        (distsances_to_centers * x_lp).sum(axis=1)
    ))
    print(f"  → LP fractional cost:   {lp_cost:,.2f}")
    print(f"  → Integrality gap hint: "
          f"{lp_cost / unfair_cost:.3f}x unfair cost")
    fair_labels = iterative_rounding(
         weights, group_codes, x_lp, distsances_to_centers
    )
    timing['Iterative Rounding'] = time.perf_counter() - t_start_rounding

    t_start_cost = time.perf_counter()
    fair_cost = float(np.dot(weights, distsances_to_centers[np.arange(len(x)), fair_labels]))
    print(f"  → Fair (integral) cost: {fair_cost:,.2f}")

    audit_fairness(fair_labels, group_codes, weights, group_names,
                   lower_bounds, upper_bounds, k_centers)

    print(f"  → Price of Fairness:    {fair_cost / unfair_cost:.4f}x  "
          f"(1.0 = fairness is free)")
    timing['Cost Calculation'] = time.perf_counter() - t_start_cost

    timing['Total Time'] = time.perf_counter() - t_start

    return unfair_centers, unfair_labels, unfair_cost, fair_labels, fair_cost, timing, x, weights, group_codes, group_names, lower_bounds, upper_bounds



if __name__ == "__main__":
    df = csv_loader.load_csv_chunked(
        "../us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=50_000,
        max_rows=50_000,
    )
    ##coreset_df = compute_fair_coreset(df_raw, n_locations=300, random_seed=42)

    ##centers_c, labels_c, cost_c = fair_clustering(
    ##    coreset_df,
    ##    feature_cols=['Lat_Scaled', 'Lon_Scaled'],
    ##    group_col='GROUP_ID',
    ##    k=10,
    ##    alpha=0.15,
    ##    weight_col='Weight',
    ##)
    ##print(f"[Coreset] Fair cost = {cost_c:,.2f}")

    preprocessed_df = csv_loader.preprocess_dataset(df)

    ### ---- Without coreset (uniform weights) ----
    ##print("\n=== Direct points mode (no coreset) ===")
    ### Prepare minimal columns needed  (we reuse the raw df here)
    ##scaler = MinMaxScaler()
    ##df[['Lat_Scaled', 'Lon_Scaled']] = scaler.fit_transform(
    ##    df[['Latitude', 'Longitude']]
    ##)
    ##df['GROUP_ID'] = (
    ##    df['RAC1P'].astype(str) + "_" + df['SEX'].astype(str)
    ##)

    FEATURE_COLS  = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K             = 4
    ALPHA         = 0.02

    (unfair_centers, unfair_labels, unfair_cost,
     fair_labels, fair_cost, timing, x, weights,
     group_codes, group_names, lower_bounds,
     upper_bounds) = fair_clustering(
        preprocessed_df,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_centers=K,
        alpha=ALPHA,
        weight_col=None
    )

    fair_result = make_result(
        algorithm="bera",
        centers=unfair_centers,
        labels=fair_labels,
        fair_cost=fair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing,
    )

    unfair_result = make_result(
        algorithm="kmedian-unfair-baseline",
        centers=unfair_centers,
        labels=unfair_labels,
        fair_cost=unfair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
    )

    summary = evaluate(fair_result, unfair_result=unfair_result)

    #udit_fairness_proportional(fair_result, lower_bounds, upper_bounds)

    plot_execution_times(fair_result, timing, title="Bera et al. — Run Time")
    #plot_spatial_clusters(preprocessed_df, fair_result,
    #                      feature_cols=FEATURE_COLS, group_col=PROTECTED_COL,
    #                      weight_col=None)
    #plot_cluster_pof(fair_result, [summary])
    #plot_pof_comparison(fair_result, [summary])
    #plot_group_pof(fair_result, [summary])
    #plot_cost_breakdown(fair_result, [summary])