import warnings
from typing import Optional, Any

import networkx as nx
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, kron, eye
import csv_loader

import numpy as np
import pandas as pd
import time

from evaluate import make_result, evaluate, audit_fairness_proportional, plot_execution_times, plot_spatial_clusters, \
    plot_cluster_pof, plot_pof_comparison, plot_group_pof, plot_cost_breakdown
from kmedian import kmedian, pairwise_l1


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
    total_points, d = x.shape
    nr_of_centers = len(centers)
    n_vars = total_points * nr_of_centers
    lower_bound_len = len(lower_bound)

    distances = pairwise_l1(x, centers).astype(np.float64)
    cost_to_center = (distances * weights[:, np.newaxis]).ravel()

    a_equality_constraint = kron(eye(total_points), np.ones((1, nr_of_centers)), format='csr')
    b_equality_constraint = np.ones(total_points)

    n_ineq = 2 * lower_bound_len * nr_of_centers
    a_upper_bound = lil_matrix((n_ineq, n_vars), dtype=np.float64)
    b_bound = np.zeros(n_ineq)

    row = 0
    for center_index in range(nr_of_centers):
        for h in range(lower_bound_len):
            in_group = (group_codes == h)
            cols = np.arange(total_points) * nr_of_centers + center_index
            # lower bound row: lower_bound * x_{ij} - (x_{ij} if i∈Col_h)
            a_upper_bound[row, cols] = lower_bound[h] - in_group
            # upper bound row: -upper_bound * x_{ij} + (x_{ij} if i∈Col_h)
            a_upper_bound[row + 1, cols] = in_group - upper_bound[h]
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
        method='highs-ds',
        options={'disp': False, 'presolve': True},
    )

    if result.status != 0:
        warnings.warn(f"LP solver returned status {result.status}: {result.message}")
        return None

    return result.x.reshape((total_points, nr_of_centers))


def min_cost_flow_rounding(
        x_lp: np.ndarray,
        group_codes: np.ndarray,
        weights: np.ndarray,
        D: np.ndarray,
) -> np.ndarray:
    n, k = x_lp.shape
    nr_groups = int(group_codes.max()) + 1

    # calculate weighted mass per group per center
    mass_group = np.zeros((nr_groups, k))
    for group_nr in range(nr_groups):
        mask = group_codes == group_nr
        mass_group[group_nr] = (x_lp[mask] * weights[mask, np.newaxis]).sum(axis=0)

    # total weighted mass assigned to centre i
    mass_total = (x_lp * weights[:, np.newaxis]).sum(axis=0)

    # integer components and remainders
    floor_mass_group = np.floor(mass_group + 1e-9).astype(int)
    floor_mass_total = np.floor(mass_total + 1e-9).astype(int)

    frac_group = mass_group - floor_mass_group
    frac_total = mass_total - floor_mass_total

    B_i = floor_mass_total - floor_mass_group.sum(axis=0)
    B = int(round(weights.sum())) - floor_mass_total.sum()

    # scale numbers up because min_cost_flow works slower with floats
    COST_SCALE = 100_000
    G = nx.DiGraph()


    # layer 1: each point node supplies their weight units
    for point in range(n):
        G.add_node(f"p_{point}", demand=-int(round(weights[point])))

    # layer 2: color_center nodes - each takes lower mass_h units
    for cluster in range(k):
        for group_nr in range(nr_groups):
            G.add_node(f"vh_{group_nr}_{cluster}", demand=int(floor_mass_group[group_nr, cluster]))

    # layer 3: center aggregator nodes - absorb B_i units
    for center in range(k):
        G.add_node(f"v_{center}", demand=int(round(B_i[center])))

    # Layer 4: sink — absorbs B units
    G.add_node("t", demand=int(round(B)))

    for point in range(n):
        group_code = group_codes[point]
        for center in range(k):
            if x_lp[point, center] > 1e-12:
                G.add_edge(f"p_{point}", f"vh_{group_code}_{center}",
                           capacity=1,
                           weight=int(round(D[point, center] * COST_SCALE)))
    for center in range(k):
        for group_nr in range(nr_groups):
            if frac_group[group_nr, center] > 1e-9:
                G.add_edge(f"vh_{group_nr}_{center}", f"v_{center}", capacity=1, weight=0)

    for center in range(k):
        if frac_total[center] > 1e-9:
            G.add_edge(f"v_{center}", "t", capacity=1, weight=0)

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

    labels = np.zeros(n, dtype=np.int32)
    for point in range(n):
        point_code = group_codes[point]
        best_flow, assigned = 0, -1
        for center in range(k):
            f_val = flow_dict.get(f"p_{point}", {}).get(f"vh_{point_code}_{center}", 0)
            if f_val > best_flow:
                best_flow, assigned = f_val, center

        labels[point] = assigned if assigned != -1 else np.argmin(D[point])

    return labels


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
) -> tuple[ndarray, Any, float] | tuple[
    ndarray, ndarray, float, ndarray, float, dict[Any, Any], Any, ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
        Any, dtype[Any]] | Any, ndarray, list, Any, Any]:
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

    print(f"[FairClustering] n={len(x):,}  k={k_cluster}  groups={nr_of_groups}  "
          f"weighted={'yes' if weight_col and weight_col in df.columns else 'no (uniform)'}")

    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = proportional_bounds(group_codes, weights, nr_of_groups, alpha)
        print(f"[FairClustering] Proportional bounds (alpha={alpha}):")
        for h, lbl in enumerate(group_names):
            print(f"  {lbl}: [{lower_bound[h]:.3f}, {upper_bound[h]:.3f}]")

    total = weights.sum()
    f = np.array([weights[group_codes == h].sum() / total for h in range(nr_of_groups)])
    infeasible_groups = np.where((f < lower_bound - 1e-6) | (f > upper_bound + 1e-6))[0]
    if len(infeasible_groups) > 0:
        for h in infeasible_groups:
            warnings.warn(
                f"Group '{group_names[h]}' proportion {f[h]:.3f} is outside "
                f"[{upper_bound[h]:.3f}, {lower_bound[h]:.3f}] — LP will be infeasible. "
                "Increase alpha or adjust bounds.")
    timing['Data Preparation'] = time.perf_counter() - t_start_prep

    t_start_kmedian = time.perf_counter()
    print("k-median for centering")
    centers, unfair_labels, unfair_cost = kmedian(
        x, k_cluster, _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed
    )
    timing['Vanilla K-Median'] = time.perf_counter() - t_start_kmedian

    t_start_lp = time.perf_counter()
    print(f"Unfair k-median cost: {unfair_cost:,.2f}")
    print("Solving Fair LP...")
    x_lp = solve_fair_lp(x, centers, weights, group_codes, lower_bound, upper_bound)
    timing['Solve Initial LP'] = time.perf_counter() - t_start_lp

    print(f"LP fractional cost: {float(np.dot(weights, (pairwise_l1(x, centers) * x_lp).sum(axis=1))):,.2f}")
    # --- Step 3: MCF rounding ---
    print("Min-cost flow rounding...")
    t_start_rounding = time.perf_counter()
    distances_to_centers = pairwise_l1(x, centers)
    labels = min_cost_flow_rounding(x_lp, group_codes, weights, distances_to_centers)
    timing['MCF Rounding'] = time.perf_counter() - t_start_rounding

    t_start_cost = time.perf_counter()
    cost = float(np.dot(weights, distances_to_centers[np.arange(len(x)), labels]))
    timing['Cost Calculation'] = time.perf_counter() - t_start_cost

    timing['Total Time'] = time.perf_counter() - t_start

    return centers, unfair_labels, unfair_cost, labels, cost, timing, x, weights, group_codes, group_names, lower_bound, upper_bound


if __name__ == "__main__":
    df = csv_loader.load_csv_chunked(
        "../us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        max_rows=10_000,
    )
    ##coreset_df = compute_fair_coreset(df, n_locations=3000, random_seed=42)
    df = csv_loader.preprocess_dataset(df)
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 5
    ALPHA = 0.02

    (centers, unfair_labels, unfair_cost, labels, cost, timing,
     x, weights, group_codes, group_names, lower_bounds, upper_bounds) = fair_clustering(
        df,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_cluster=K,
        alpha=ALPHA,
        weight_col='Weight',
    )

    fair_result = make_result(
        algorithm="bercea",
        centers=centers,
        labels=labels,
        fair_cost=cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing,
    )

    unfair_result = make_result(
        algorithm="kmedian-unfair-baseline",
        centers=centers,
        labels=unfair_labels,
        fair_cost=unfair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
    )

    summary = evaluate(fair_result, unfair_result=unfair_result)

    audit_fairness_proportional(fair_result, lower_bounds, upper_bounds)

    plot_execution_times(fair_result ,timing, title="Essential k-Median — Run Time")
    plot_spatial_clusters(df, fair_result, unfair_result,
                          feature_cols=FEATURE_COLS)
    plot_cluster_pof(fair_result, [summary])
    plot_pof_comparison(fair_result, [summary])
    plot_group_pof(fair_result, [summary])
    plot_cost_breakdown(fair_result, [summary])