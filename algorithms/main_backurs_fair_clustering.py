from __future__ import annotations

import time
import warnings
from typing import Optional, Any

import numpy as np
import pandas as pd

import csv_loader
from evaluate import (
    make_result, evaluate, audit_fairness_proportional,
    plot_execution_times, plot_spatial_clusters,
    plot_cluster_pof, plot_pof_comparison, plot_group_pof, plot_cost_breakdown,
)
from kmedian import kmedian, pairwise_l1

def is_balanced(len_left: int, len_right: int, left: int, right: int) -> bool:
    if len_left == 0 and len_right == 0:
        return True
    if len_left == 0 or len_right == 0:
        return False
    return len_left * right <= len_right * left and len_right * right <= len_left * left


def validate_fairlets(fairlets: list[list[int]], colours: np.ndarray,
                      r: int, b: int, n_total: int) -> None:
    """Check that fairlets cover every point exactly once and are balanced."""
    seen = set()
    duplicates = 0
    balance_violations = 0

    for fairlet in fairlets:
        for pid in fairlet:
            if pid in seen:
                duplicates += 1
            seen.add(pid)
        if len(fairlet) > 1:
            nr = sum(1 for i in fairlet if colours[i] == 0)
            nb = len(fairlet) - nr
            if not is_balanced(nr, nb, r, b):
                balance_violations += 1

    if duplicates > 0:
        warnings.warn(f"{duplicates} duplicate point(s) in fairlets!")
    if len(seen) != n_total:
        warnings.warn(f"Fairlets cover {len(seen)}/{n_total} points "
                      f"({n_total - len(seen)} missing).")
    if balance_violations > 0:
        warnings.warn(f"{balance_violations}/{len(fairlets)} fairlets "
                      f"violate ({r},{b})-balance.")
    else:
        print(f"[Backurs] All {len(fairlets)} fairlets satisfy "
              f"({r},{b})-balance.")

def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    cats = pd.Categorical(group_series)
    return cats.codes.astype(np.int32), list(cats.categories)


class HSTNode:
    __slots__ = ['level', 'children', 'leaf_indices', 'edge_weight']

    def __init__(self, level: int, edge_weight: float = 0.0):
        self.level = level
        self.children: list[HSTNode] = []
        self.leaf_indices: list[int] = []  # point indices ONLY at leaves

def build_hst(
    x: np.ndarray,
    gamma: int = 2,
    random_seed: int = 42,
) -> HSTNode:
    """
    construct gamma-HST embedding of X using randomly shifted grids, points are then stored in leaf nodes
    """
    rng = np.random.default_rng(random_seed)
    n_points, n_dims = x.shape

    # bounding cube
    span = (x.max(axis=0) - x.min(axis=0)).max()
    cube_side  = 4 * span


    shift = rng.uniform(-span, span, size=n_dims)
    x_shifted = x + shift

    origin = x_shifted.min(axis=0) - span
    x_norm = x_shifted - origin

    max_depth = int(np.ceil(np.log(max(n_points, 4)) / np.log(gamma))) + 3

    def _build(point_idx: list[int], depth: int, cell_side: float) -> HSTNode:
        edge_w = cell_side * np.sqrt(n_dims)
        node = HSTNode(level=depth)

        if len(point_idx) <= 1 or depth >= max_depth:
            node.leaf_indices = list(point_idx)
            return node

        child_side = cell_side / gamma

        if child_side < 1e-15:
            node.leaf_indices = list(point_idx)
            return node

        coords = x_norm[point_idx]
        grid_idx = np.floor(coords / child_side).astype(np.int64)
        grid_idx = np.clip(grid_idx, 0, gamma - 1)


        buckets: dict[tuple, list[int]] = {}
        for local_point_idx, global_point_idx in enumerate(point_idx):
            cell_key = tuple(grid_idx[local_point_idx])
            buckets.setdefault(cell_key, []).append(global_point_idx)

        if len(buckets) <= 1:
            if  depth < max_depth:
                child = _build(point_idx, depth + 1, child_side)
                node.children = [child]
            else:
                node.leaf_indices = list(point_idx)
            return node

        for cell_points in buckets.values():
            node.children.append(_build(cell_points, depth + 1, child_side))
        return node

    return _build(list(range(n_points)), 0, cube_side)

def collect_leaf_points(node: HSTNode) -> list[int]:
    """Return every point index stored in the leaves below *node*."""
    if not node.children:
        return list(node.leaf_indices)
    result: list[int] = []
    for child in node.children:
        result.extend(collect_leaf_points(child))
    return result


def compute_excess(n_red: int, n_blue: int,
                   r: int, b: int) -> tuple[int, int]:
    """
    Return (remove_red, remove_blue) — the minimum number of points
    to discard from one colour so that the remainder is (r,b)-balanced.

    Example with (r=1, b=1):
        (5 red, 3 blue)  →  remove 2 red  →  (3, 3) balanced.
    """
    if n_red == 0 and n_blue == 0:
        return 0, 0
    if n_red >= n_blue:
        max_allowed_red = (n_blue * r) // b
        return max(0, n_red - max_allowed_red), 0
    else:
        max_allowed_blue = (n_red * r) // b
        return 0, max(0, n_blue - max_allowed_blue)


def leftover_fairlet_size(n_red: int, n_blue: int,
                          r: int, b: int) -> tuple[int, int]:
    """
    After packing as many full (r+b)-sized fairlets as possible from a
    balanced set, return (leftover_red, leftover_blue).
    """
    if n_red == 0 and n_blue == 0:
        return 0, 0
    if n_red >= n_blue:
        full_fairlets = n_blue // b
        left_red = n_red - full_fairlets * r
        left_blue = n_blue - full_fairlets * b
    else:
        full_fairlets = n_red // b
        left_blue = n_blue - full_fairlets * r
        left_red = n_red - full_fairlets * b
    return max(0, left_red), max(0, left_blue)


def borrowable_dominant(is_red_dominant: bool, n_red: int, n_blue: int,
                        r: int, b: int) -> int:
    """
    How many points of the dominant colour can we borrow from a child
    while keeping that child (r,b)-balanced?
    """
    if is_red_dominant:
        if n_red <= n_blue:
            return 0
        min_red = -(-n_blue * b // r)
        return max(0, n_red - min_red)
    else:
        if n_blue <= n_red:
            return 0
        min_blue = -(-n_red * b // r)
        return max(0, n_blue - min_blue)


def compute_heavy_point_counts(
        child_colour_counts: list[tuple[int, int]],
        r: int, b: int,
) -> list[tuple[int, int]]:
    """
    For every child of an internal HST node, decide how many red and blue
    points to pull up as "heavy" points so that:
      (a) each child's remaining points are (r,b)-balanced, and
      (b) the collected heavy points are themselves (r,b)-balanced
          (so we can form fairlets from them).

    Three stages:

      Stage 1 — Mandatory excess.
          Remove the minimum necessary from each child to make it balanced.

      Stage 2 — Borrow dominant-colour points.
          If the total heavy set is imbalanced, borrow extra points of the
          dominant colour from children that have a surplus.

      Stage 3 — Pull up non-saturated fairlet remainders.
          If still imbalanced, pull up entire leftover fairlets from
          children (each leftover is smaller than r+b).
    """
    n_children = len(child_colour_counts)
    if n_children == 0:
        return []

    removals = [compute_excess(nr, nb, r, b)
                for nr, nb in child_colour_counts]

    total_remove_red = sum(rr for rr, _ in removals)
    total_remove_blue = sum(rb for _, rb in removals)

    if is_balanced(total_remove_red, total_remove_blue, r, b):
        return removals

    red_is_dominant = total_remove_red >= total_remove_blue

    for i, (child_red, child_blue) in enumerate(child_colour_counts):
        remove_red, remove_blue = removals[i]
        remaining_red = child_red - remove_red
        remaining_blue = child_blue - remove_blue

        if red_is_dominant:
            can_borrow = borrowable_dominant(True, remaining_red,
                                             remaining_blue, r, b)
            need = (total_remove_red
                    - (total_remove_blue * r // b if total_remove_blue > 0
                       else 0))
            take = min(can_borrow, max(0, need))
            removals[i] = (remove_red + take, remove_blue)
        else:
            can_borrow = borrowable_dominant(False, remaining_red,
                                             remaining_blue, r, b)
            need = (total_remove_blue
                    - (total_remove_red * r // b if total_remove_red > 0
                       else 0))
            take = min(can_borrow, max(0, need))
            removals[i] = (remove_red, remove_blue + take)

        total_remove_red = sum(rr for rr, _ in removals)
        total_remove_blue = sum(rb for _, rb in removals)
        if is_balanced(total_remove_red, total_remove_blue, r, b):
            return removals

    for i, (child_red, child_blue) in enumerate(child_colour_counts):
        remove_red, remove_blue = removals[i]
        remaining_red = child_red - remove_red
        remaining_blue = child_blue - remove_blue
        if remaining_red + remaining_blue == 0:
            continue

        left_red, left_blue = leftover_fairlet_size(
            remaining_red, remaining_blue, r, b)
        if left_red + left_blue == 0:
            continue

        removals[i] = (remove_red + left_red, remove_blue + left_blue)
        total_remove_red = sum(rr for rr, _ in removals)
        total_remove_blue = sum(rb for _, rb in removals)
        if is_balanced(total_remove_red, total_remove_blue, r, b):
            return removals

    return removals


def pack_into_fairlets(red_ids: list[int], blue_ids: list[int],
                       red_balance: int, blue_balance: int) -> list[list[int]]:
    """
    Partition a collection of red and blue point indices into (r,b)-fairlets.

    Each fairlet has at most r+b points:  r from the majority colour and
    b from the minority colour.  Any remainder that cannot form a balanced
    fairlet is split into singletons (a single point is trivially fair).
    """
    n_red, n_blue = len(red_ids), len(blue_ids)
    if n_red == 0 and n_blue == 0:
        return []

    if n_red >= n_blue:
        majority, minority = list(red_ids), list(blue_ids)
    else:
        majority, minority = list(blue_ids), list(red_ids)

    fairlets: list[list[int]] = []
    mi, ni = 0, 0  # cursors into majority / minority

    while mi + red_balance<= len(majority) and ni + blue_balance <= len(minority):
        fairlet = majority[mi:mi + red_balance] + minority[ni:ni + blue_balance]
        fairlets.append(fairlet)
        mi += red_balance
        ni += blue_balance

    #leftover points
    leftover_maj = majority[mi:]
    leftover_min = minority[ni:]

    if leftover_maj or leftover_min:
        left_r = len(leftover_maj) if n_red >= n_blue else len(leftover_min)
        left_b = len(leftover_min) if n_red >= n_blue else len(leftover_maj)

        if is_balanced(left_r, left_b, red_balance, blue_balance) and left_r + left_b > 0:
            fairlets.append(leftover_maj + leftover_min)
        else:
            # can't form a balanced group — split into harmless singletons
            for idx in leftover_maj:
                fairlets.append([idx])
            for idx in leftover_min:
                fairlets.append([idx])

    return fairlets

def fairlet_decomposition(root: HSTNode, colours: np.ndarray,
                          r: int, b: int) -> list[list[int]]:
    """
    Walk the HST top-down.  At every internal node:
      1. Decide how many points of each colour to pull from each child
         (MinHeavyPoints).
      2. Actually remove those points from the available pool and form
         fairlets from them.
      3. Recurse into each child with the remaining (now balanced) points.

    We track which points are still "available" (not yet placed in a fairlet)
    using a global set — this prevents any point from appearing twice.
    """
    available_points = set(collect_leaf_points(root))
    all_fairlets: list[list[int]] = []

    def _count_available(node: HSTNode) -> tuple[int, int]:
        "Count available red / blue points in the subtree of *node*."
        if not node.children:
            n_red = sum(1 for i in node.leaf_indices
                        if i in available_points and colours[i] == 0)
            n_blue = sum(1 for i in node.leaf_indices
                         if i in available_points and colours[i] == 1)
            return n_red, n_blue
        total_red, total_blue = 0, 0
        for child in node.children:
            child_red, child_blue = _count_available(child)
            total_red += child_red
            total_blue += child_blue
        return total_red, total_blue

    def _take_points(node: HSTNode, colour: int,
                     how_many: int) -> list[int]:
        "Collect up to how_many available points of colour from subtree."
        if how_many <= 0:
            return []
        taken: list[int] = []
        if not node.children:
            for pid in node.leaf_indices:
                if pid in available_points and colours[pid] == colour:
                    taken.append(pid)
                    if len(taken) >= how_many:
                        return taken
            return taken
        for child in node.children:
            taken.extend(_take_points(child, colour,
                                      how_many - len(taken)))
            if len(taken) >= how_many:
                return taken[:how_many]
        return taken

    def _decompose(node: HSTNode) -> None:
        n_red, n_blue = _count_available(node)
        if n_red + n_blue == 0:
            return

        if not node.children:
            reds = [i for i in node.leaf_indices
                    if i in available_points and colours[i] == 0]
            blues = [i for i in node.leaf_indices
                     if i in available_points and colours[i] == 1]
            for fairlet in pack_into_fairlets(reds, blues, r, b):
                for pid in fairlet:
                    available_points.discard(pid)
                all_fairlets.append(fairlet)
            return

        child_colour_counts = [_count_available(ch) for ch in node.children]
        removals = compute_heavy_point_counts(child_colour_counts, r, b)

        heavy_reds: list[int] = []
        heavy_blues: list[int] = []
        for child_idx, child in enumerate(node.children):
            take_red, take_blue = removals[child_idx]
            if take_red > 0:
                heavy_reds.extend(_take_points(child, 0, take_red))
            if take_blue > 0:
                heavy_blues.extend(_take_points(child, 1, take_blue))

        for pid in heavy_reds:
            available_points.discard(pid)
        for pid in heavy_blues:
            available_points.discard(pid)

        if heavy_reds or heavy_blues:
            all_fairlets.extend(
                pack_into_fairlets(heavy_reds, heavy_blues, r, b))

        for child in node.children:
            _decompose(child)

    _decompose(root)

    left_overs = sorted(available_points)
    if left_overs:
        reds = [i for i in left_overs if colours[i] == 0]
        blues = [i for i in left_overs if colours[i] == 1]
        all_fairlets.extend(pack_into_fairlets(reds, blues, r, b))
        available_points.clear()

    return all_fairlets


def cluster_fairlets(points: np.ndarray, fairlets: list[list[int]],
                     k: int, kmedian_trials: int = 5,
                     kmedian_max_iter: int = 30,
                     random_seed: int = 42
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    1. For each fairlet pick one representative point.
    2. Run k-median on the representatives, weighted by fairlet size.
    3. Every point inherits the cluster of its fairlet's representative.
    """
    n_points = len(points)

    representative_ids = np.array([f[0] for f in fairlets])
    representative_coords = points[representative_ids]
    fairlet_sizes = np.array([float(len(f)) for f in fairlets])

    centers, rep_labels, _ = kmedian(
        representative_coords, k,
        _weights=fairlet_sizes,
        n_trials=kmedian_trials, max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )

    # cluster labels from representatives to all members
    labels = np.full(n_points, -1, dtype=np.int32)
    for fairlet_idx, fairlet in enumerate(fairlets):
        cluster_id = rep_labels[fairlet_idx]
        for pid in fairlet:
            labels[pid] = cluster_id

    leftover_points = np.where(labels == -1)[0]
    if len(leftover_points) > 0:
        distances = pairwise_l1(points[leftover_points], centers)
        labels[leftover_points] = np.argmin(distances, axis=1).astype(np.int32)

    all_distances = pairwise_l1(points, centers)
    total_cost = float(all_distances[np.arange(n_points), labels].sum())

    return centers, labels, total_cost

def fair_clustering(
        df: pd.DataFrame,
        feature_cols: list[str],
        protected_group_col: str,
        k: int,
        red_balance: int = 1,
        blue_balance: int = 1,
        kmedian_trials: int = 5,
        kmedian_max_iter: int = 30,
        random_seed: int = 42,
        gamma: int = 2,
) -> tuple:
    timing = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    x = df[feature_cols].to_numpy(dtype=np.float64)
    group_codes, group_names = encode_groups_to_int(df[protected_group_col])

    colors = group_codes.copy()
    n_red = int((colors == 0).sum())
    n_blue = int((colors == 1).sum())

    print(f"\n[Backurs] n={len(x):,}  k={k}  groups={group_names}")
    print(f"[Backurs] Red ('{group_names[0]}'): {n_red}  "
          f"Blue ('{group_names[1]}'): {n_blue}")
    print(f"[Backurs] Balance: (r={red_balance}, b={blue_balance})  "
          f"→ min balance ≥ {blue_balance/red_balance:.3f}")

    timing["Data Preparation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    unfair_centers, unfair_labels, unfair_cost = kmedian(
        x, k, _weights=None,
        n_trials=kmedian_trials, max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    timing["Vanilla k-Median"] = time.perf_counter() - t0
    print(f"[Backurs] Unfair k-median cost: {unfair_cost:,.2f}")

    t0 = time.perf_counter()
    print("[Backurs] Building HST embedding...")
    hst_root = build_hst(x, gamma=gamma, random_seed=random_seed)
    timing["HST Construction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fairlets = fairlet_decomposition(hst_root, colors,
                                     red_balance, blue_balance)
    timing["Fairlet Decomposition"] = time.perf_counter() - t0

    n_fairlets = len(fairlets)
    covered = sum(len(f) for f in fairlets)
    avg_size = covered / n_fairlets if n_fairlets else 0
    print(f"[Backurs] {n_fairlets} fairlets  "
          f"(avg size {avg_size:.1f}, "
          f"covering {covered}/{len(x)} points)")
    validate_fairlets(fairlets, colors, red_balance, blue_balance, len(x))

    t0 = time.perf_counter()
    fair_centers, fair_labels, fair_cost = cluster_fairlets(
        x, fairlets, k,
        kmedian_trials=kmedian_trials, kmedian_max_iter=kmedian_max_iter,
        random_seed=random_seed,
    )
    timing["Cluster Fairlets"] = time.perf_counter() - t0
    timing["Total"] = time.perf_counter() - t_start

    pof = fair_cost / unfair_cost if unfair_cost > 0 else float("inf")

    print(f"[Backurs] Fair cost: {fair_cost:,.2f}   "
          f"PoF: {pof:.4f}")

    return (fair_centers, fair_labels, fair_cost, timing,
            unfair_centers, unfair_labels, unfair_cost,
            x, group_codes, group_names, df, fairlets)

def audit_cluster_balance(labels: np.ndarray, colours: np.ndarray,
                          k: int, r: int, b: int,
                          group_names: list) -> int:
    """Print per-cluster balance and return the number of violations."""
    required_balance = b / r
    violations = 0
    print(f"\n[Balance Audit] Required >= {required_balance:.3f}")
    for j in range(k):
        mask = labels == j
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue
        n_red = int((mask & (colours == 0)).sum())
        n_blue = cluster_size - n_red
        balance = (min(n_red / n_blue, n_blue / n_red)
                   if n_red > 0 and n_blue > 0 else 0.0)
        if balance < required_balance - 1e-6:
            violations += 1
            print(f"  ⚠ Cluster {j}: {group_names[0]}={n_red}, "
                  f"{group_names[1]}={n_blue}, balance={balance:.3f}")
    if violations == 0:
        print(f"  ✓ All {k} clusters satisfy ({r},{b})-balance.")
    return violations

if __name__ == "__main__":
    df = csv_loader.load_csv_chunked(
        "../us_census_puma_data.csv",
        csv_loader.LOAD_COLS, max_rows=10_000)
    df = csv_loader.preprocess_dataset(df)
    df["BINARY_GROUP"] = df["SEX"].astype(str)

    (fair_centers, fair_labels, fair_cost, timing,
     unfair_centers, unfair_labels, unfair_cost,
     points, group_codes, group_names, _, fairlets) = fair_clustering(
        df,
        feature_cols=["Lat_Scaled", "Lon_Scaled"],
        protected_group_col="BINARY_GROUP",
        k=5, red_balance=1, blue_balance=1, random_seed=42)

    weights = np.ones(len(points))
    fair_result = make_result("backurs", fair_centers, fair_labels,
                              fair_cost, unfair_cost, points, weights,
                              group_codes, group_names, timing)
    unfair_result = make_result("kmedian-unfair", unfair_centers,
                                unfair_labels, unfair_cost, unfair_cost,
                                points, weights, group_codes, group_names)
    evaluate(fair_result, unfair_result=unfair_result)
    audit_cluster_balance(fair_labels, group_codes, 5, 1, 1, group_names)