"""

1. Run a vanilla (unfair) k-median to obtain integral centers S.
2. Solve the fair-clustering LP relaxation restricted to those centers
   (the assignment LP from inequality (10) in the paper) to obtain a
   fractional fair assignment x_LP.
3. Round x_LP to an integer assignment via a min-cost flow (Lemma 8 /
   Figure 1 of the paper). The resulting integer solution is essentially
   fair: for every center i and color h, |mass_h(integer) - mass_h(x_LP)|
   is at most 1.

The paper's Lemma 7 first solves the LP over the full location set L
and then consolidates fractional mass onto the nearest opened center,
yielding a fractional fair solution supported on the integral centers.
This implementation skips the consolidation step by solving the LP
directly over the kmedian centers. The resulting LP cost is at least
as large as the consolidated cost, so the essentially-fair guarantee is
preserved, but the cost bound 2 * c_LP + c_bar from the paper does not
transfer verbatim.


"""


import warnings
from typing import Optional, Any

import networkx as nx
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, kron, eye
from fair_clustering import csv_loader

import numpy as np
import pandas as pd
import time

from fair_clustering.evaluate import make_result, evaluate, audit_fairness_proportional, plot_execution_times, \
    plot_spatial_clusters, \
    plot_cluster_pof, plot_pof_comparison, plot_group_pof, plot_cost_breakdown
from fair_clustering.kmedian import kmedian, pairwise_l1


def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    """
    Convert a categorical protected-attribute column to integer codes.
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
    Compute relaxed lower and upper bound for each color

    Proportional relaxation here: each cluster's fraction of color h must lie in
    [max(0, f_h - alpha), min(1, f_h + alpha)] where f_h is the
    weighted global proportion of color h.


    :param group_codes:  Integer codes of length n giving the color of each
            point.
    :param weights: not used, legacy, is just array of [1,1,1,1,1...]
    :param n_groups: Number of distinct colors
    :param alpha: additive slack in [0, 1]
    :return: pair lower_bound, upper_bound, each an array of length
        n_groups giving the lower and upper allowed proportion per color
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
    Solve the fair-assignment LP for fixed centers.

    x_{ij} in [0, 1] for each (point i, center j). The LP
    encodes the inequalities restricted to the given centers:
        sum_j x_{ij} == 1  for all i
        l_h * sum_i w_i x_{ij} = sum_{i in C_h} x_{ij} <= u_h * sum_i x_{ij} for all (j, h)

    The objective minimises sum_{i,j} d(i, j) x_{ij} (weighted L1
    cost), the LP-relaxed k-median objective on the fixed center set.


    The fairness inequalities are encoded in matrix form by stacking
    two rows per (center, color) pair. For point i, center j and
    color h the coefficient of x_{ij} is l_h - 1[i in C_h]
    on the lower-bound row and 1[i in C_h] - u_h on the upper-bound row, which expand to the sums above.


    :param x: array of datapoints
    :param centers: array of centers with the (fixed) centers from the
            unfair k-median.
    :param weights: currenly legacy, only array of [1, 1, 1-..]
    :param group_codes: array of integer color codes of length n
    :param lower_bound: per-color lower fraction bounds
    :param upper_bound: per-color upper fraction bounds
    :return:
    """
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

    # highs dual simplex tends to be the most reliable for this LP shape
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
    """
    Round the fractional fair LP solution to an integer assignment.

    Constructs a four-layer min-cost flow network whose integer optimum is an integer assignment
    that respects the floored per-(color, center) masses of x_lp, losing at most one unit of fairness mass per (color, cluster) pair.

    Network layers (with NetworkX node-demand convention,
    negative = supply):
        p_i : one node per point, demand -w_i (supplies w_i units)
        vh_{h,j} : one node per (color h, center j), demand floor(mass_h(x_lp, j))
        v_j : one node per center, demand B_j = floor(mass(x_lp, j)) - sum_h floor(mass_h(x_lp, j))
        t : sink, demand B = sum_i w_i - sum_j floor(mass(x_lp, j))

    The vh -> v -> t tail layers absorb the fractional remainders so
    that total supply matches total demand and a feasible integer flow
    exists whenever the LP solution did. Each unit of flow on edge
    (p_i, vh_{c(i), j}) represents assigning one mass unit of point
    i to center j; we recover labels by taking the center carrying the
    most flow out of each point.

    Edge costs are scaled by 100_000 and rounded to
    integers because NetworkX's min_cost_flow is much faster on
    integer weights.

    :param x_lp: array of fractional assignments from solve_fair_lp
    :param group_codes: array of integer color codes
    :param weights: not used in reality, array if 1
    :param D: L1 distance matrix from points to centers, used as
            edge cost
    :return: labels of length n with cluster ids
    """
    n, k = x_lp.shape
    nr_groups = int(group_codes.max()) + 1

    # calculate weighted mass per group per center
    # mass_h(x_lp, j) = sum_{i in color h} w_i * x_lp[i, j]
    mass_group = np.zeros((nr_groups, k))
    for group_nr in range(nr_groups):
        mask = group_codes == group_nr
        mass_group[group_nr] = (x_lp[mask] * weights[mask, np.newaxis]).sum(axis=0)

    # total weighted mass assigned to centre i
    # mass(x_lp, j) = total weighted mass assigned to center j
    mass_total = (x_lp * weights[:, np.newaxis]).sum(axis=0)

    # integer (floor) and fractional (remainder) parts
    floor_mass_group = np.floor(mass_group + 1e-9).astype(int)
    floor_mass_total = np.floor(mass_total + 1e-9).astype(int)

    frac_group = mass_group - floor_mass_group
    frac_total = mass_total - floor_mass_total

    # B_i: fractional remainder at center j across all colors combined
    B_i = floor_mass_total - floor_mass_group.sum(axis=0)
    # B: total fractional remainder, absorbed by the sink t
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

    # layer 4: sink — absorbs B units
    G.add_node("t", demand=int(round(B)))

    # edges p -> vh: only where the LP put nonzero mass, cost is L1 distance
    for point in range(n):
        group_code = group_codes[point]
        for center in range(k):
            if x_lp[point, center] > 1e-12:
                G.add_edge(f"p_{point}", f"vh_{group_code}_{center}",
                           capacity=1,
                           weight=int(round(D[point, center] * COST_SCALE)))
    # edges vh -> v carry the fractional color-mass remainder
    for center in range(k):
        for group_nr in range(nr_groups):
            if frac_group[group_nr, center] > 1e-9:
                G.add_edge(f"vh_{group_nr}_{center}", f"v_{center}", capacity=1, weight=0)
    # edges v -> t carry the per-center fractional remainder to the sink
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

    # each point i is assigned to the center whose
    # vh_{c(i), j} node received the largest amount of flow from p_i
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
    """
    Steps:
        1. Encode features, and protected-group labels.
        2. Derive per-color [l_h, u_h] bounds.
        3. Run vanilla weighted k-median to obtain centers and an unfair
           baseline cost.
        4. Solve the fair-assignment LP on those centers.
        5. Round to an integer fair assignment via min-cost flow.
        6. Compute the resulting fair cost.


    :param df: data
    :param feature_cols: names of numeric x and y columns for points
    :param protected_group_col: name of the column holding the protected
            attribute, treated as disjoint categorical groups.
    :param k_cluster: number of clusters k
    :param alpha: additive slack used to derive the lower and upper bounds
    :param weight_col:
    :params lower_bound, upper_bound: Optional pre-computed per-color
            bounds; if either is None, both are recomputed from
            alpha with proportional_bounds
    :param kmedian_trials, kmedian_max_iter, random_seed:  Forwarded to the
            kmedian solver.

    :return: (centers, unfair_labels, unfair_cost, labels, cost, timing,
        x, weights, group_codes, group_names, lower_bound, upper_bound).
        Timing is a dict mapping pipeline stage names to runtime
        seconds.
    """

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

    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = proportional_bounds(group_codes, weights, nr_of_groups, alpha)

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
    centers, unfair_labels, unfair_cost = kmedian(
        x, k_cluster, _weights=weights,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed
    )
    timing['Vanilla K-Median'] = time.perf_counter() - t_start_kmedian

    t_start_lp = time.perf_counter()
    x_lp = solve_fair_lp(x, centers, weights, group_codes, lower_bound, upper_bound)
    timing['Solve Initial LP'] = time.perf_counter() - t_start_lp

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
    K = 10
    ALPHA = 0.05

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