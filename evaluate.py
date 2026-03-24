import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Distance helper for cost calculation
from kmedian import pairwise_l1


"""
evaluate.py — Shared evaluation, fairness auditing, and visualisation
======================================================================
All three fair-clustering algorithms (Bera, Böhm, Essential k-Median)
return their results in a common format and call the helpers here.

Common result format
--------------------
Every algorithm should produce:

    ClusteringResult(
        centers       : (k, d)  np.ndarray
        labels        : (n,)    np.ndarray  int32
        fair_cost     : float
        unfair_cost   : float
        weights       : (n,)    np.ndarray  (1.0 per point if unweighted)
        group_codes   : (n,)    np.ndarray  int32
        group_names   : list[str]
        timing        : dict[str, float]    step → seconds
        algorithm     : str                 display name
    )

Helper factory: make_result(...)  — see below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score



@dataclass
class ClusteringResult:
    centers:      np.ndarray
    labels:       np.ndarray
    fair_cost:    float
    unfair_cost:  float
    weights:      np.ndarray
    group_codes:  np.ndarray
    group_names:  list
    timing:       dict = field(default_factory=dict)
    algorithm:    str  = "Unknown"

    # ---- convenient derived quantities ------------------------------------
    @property
    def k(self) -> int:
        return len(self.centers)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def pof(self) -> float:
        """Overall Price of Fairness (fair / unfair cost)."""
        return compute_pof(self.fair_cost, self.unfair_cost)


def make_result(
    algorithm:    str,
    centers:      np.ndarray,
    labels:       np.ndarray,
    fair_cost:    float,
    unfair_cost:  float,
    weights:      np.ndarray,
    group_codes:  np.ndarray,
    group_names:  list,
    timing:       Optional[dict] = None,
) -> ClusteringResult:
    return ClusteringResult(
        algorithm=algorithm,
        centers=centers,
        labels=labels,
        fair_cost=fair_cost,
        unfair_cost=unfair_cost,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing or {},
    )



def compute_pof(fair_cost: float, unfair_cost: float) -> float:
    """
    Overall Price of Fairness = fair_cost / unfair_cost.
    A value of 1.0 means fairness is free; higher means more expensive.
    """
    if unfair_cost == 0:
        return float("inf")
    return fair_cost / unfair_cost


def compute_group_costs(result: ClusteringResult) -> dict:
    """
    Weighted L1 cost broken down by demographic group.

    Returns
    -------
    dict  group_name → weighted cost (float)
    """
    costs = {}
    for h, name in enumerate(result.group_names):
        mask = result.group_codes == h
        if not mask.any():
            costs[name] = 0.0
            continue
        idx = result.labels[mask]
        dists = np.sum(np.abs(result.centers[idx] - _placeholder_X(result)[mask]), axis=1)
        costs[name] = float((dists * result.weights[mask]).sum())
    return costs


def compute_group_costs_from_X(
    X: np.ndarray,
    result: ClusteringResult,
) -> dict:
    """
    Weighted L1 cost broken down by demographic group.

    Parameters
    ----------
    X      : (n, d) point coordinates matching result.labels / result.weights
    result : ClusteringResult

    Returns
    -------
    dict  group_name → weighted cost (float)
    """
    n = len(X)
    # distance of every point to its assigned center
    all_weighted_dists = np.zeros(n)
    for j in range(result.k):
        mask = result.labels == j
        if mask.any():
            dists = np.sum(np.abs(X[mask] - result.centers[j]), axis=1)
            all_weighted_dists[mask] = dists * result.weights[mask]

    costs = {}
    for h, name in enumerate(result.group_names):
        gmask = result.group_codes == h
        costs[name] = float(all_weighted_dists[gmask].sum())
    return costs


def compute_gpof(
    fair_group_costs: dict,
    unfair_group_costs: dict,
) -> dict:
    """
    Group-level Price of Fairness.

    G-PoF(g) = fair_cost(g) / unfair_cost(g)

    Returns
    -------
    dict  group_name → G-PoF (float)
    """
    gpof = {}
    for g, fc in fair_group_costs.items():
        uc = unfair_group_costs.get(g, 0.0)
        gpof[g] = fc / uc if uc > 0 else float("inf")
    return gpof





def audit_fairness_proportional(
    result: ClusteringResult,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    verbose: bool = True,
) -> int:
    """
    Check per-(cluster, group) proportional fairness for Bera / Essential k-Median.

    A cluster violates fairness if the weighted fraction of group h is outside
    [lower_bounds[h] - ε, upper_bounds[h] + ε].

    Returns the number of (cluster, group) violations.
    """
    k          = result.k
    violations = 0
    tol        = 1e-4

    if verbose:
        print(f"\n[Fairness Audit — Proportional Bounds] algorithm='{result.algorithm}'")

    for j in range(k):
        at_j    = result.labels == j
        total_j = result.weights[at_j].sum()
        if total_j == 0:
            continue
        for h, name in enumerate(result.group_names):
            mass_h = result.weights[at_j & (result.group_codes == h)].sum()
            frac   = mass_h / total_j
            lo, hi = lower_bounds[h], upper_bounds[h]
            if frac < lo - tol or frac > hi + tol:
                violations += 1
                if verbose:
                    print(f"  ⚠ Cluster {j:2d}  Group '{name}': "
                          f"{frac*100:.1f}% outside [{lo*100:.1f}%, {hi*100:.1f}%]")

    if verbose:
        if violations == 0:
            print("  ✓ All clusters satisfy proportional fairness bounds.")
        else:
            print(f"  → {violations} (cluster, group) pair(s) violate bounds "
                  "(additive violations ≤ 1 are expected by Lemma 7 / Bera et al.).")
    return violations


def audit_fairness_exact_balance(
    result: ClusteringResult,
    verbose: bool = True,
) -> int:
    """
    Check exact equal-group-size balance required by Böhm et al.

    Every cluster must have exactly 1/H of each group (count-based).
    Returns the number of (cluster, group) violations.
    """
    k          = result.k
    H          = result.n_groups
    violations = 0
    tol        = 1e-4

    if verbose:
        print(f"\n[Fairness Audit — Exact Balance] algorithm='{result.algorithm}'")

    for j in range(k):
        at_j    = result.labels == j
        total_j = int(at_j.sum())
        if total_j == 0:
            continue
        expected = 1.0 / H
        for h, name in enumerate(result.group_names):
            count_h = int((at_j & (result.group_codes == h)).sum())
            frac    = count_h / total_j
            if abs(frac - expected) > tol:
                violations += 1
                if verbose:
                    print(f"  ⚠ Cluster {j:2d}  Group '{name}': "
                          f"{count_h} pts ({frac*100:.1f}%) ≠ {expected*100:.1f}%")

    if verbose:
        if violations == 0:
            print("  ✓ All clusters are perfectly balanced across groups.")
        else:
            print(f"  → {violations} uneven group distributions.")
    return violations


# ---------------------------------------------------------------------------
# 4.  Full evaluation summary
# ---------------------------------------------------------------------------

def evaluate(
    X:                  np.ndarray,
    result:             ClusteringResult,
    unfair_group_costs: Optional[dict] = None,
    silhouette:         bool = True,
    verbose:            bool = True,
) -> dict:
    """
    Compute all thesis metrics for one algorithm result.

    Parameters
    ----------
    X                  : (n, d) point coordinates
    result             : ClusteringResult
    unfair_group_costs : pre-computed unfair group costs for G-PoF.
                         If None, G-PoF is not computed.
    silhouette         : whether to compute the Silhouette Score (slow for large n)
    verbose            : print a summary table

    Returns
    -------
    dict with keys:
        Algorithm, Total Cost (Fair), Total Cost (Unfair), PoF,
        Max G-PoF, Avg G-PoF, Silhouette, Group_Costs, Group_PoFs
    """
    fair_group_costs = compute_group_costs_from_X(X, result)
    total_fair       = sum(fair_group_costs.values())

    gpof = {}
    if unfair_group_costs is not None:
        gpof = compute_gpof(fair_group_costs, unfair_group_costs)


    summary = {
        "Algorithm":            result.algorithm,
        "Total Cost (Fair)":    total_fair,
        "Total Cost (Unfair)":  result.unfair_cost,
        "PoF":                  compute_pof(total_fair, result.unfair_cost),
        "Max G-PoF":            max(gpof.values()) if gpof else None,
        "Avg G-PoF":            float(np.mean(list(gpof.values()))) if gpof else None,
        "Group_Costs":          fair_group_costs,
        "Group_PoFs":           gpof,
    }

    if verbose:
        _print_summary(summary)

    return summary


def _print_summary(s: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluation — {s['Algorithm']}")
    print(f"{'='*55}")
    print(f"  Fair cost    : {s['Total Cost (Fair)']:>15,.2f}")
    print(f"  Unfair cost  : {s['Total Cost (Unfair)']:>15,.2f}")
    print(f"  PoF          : {s['PoF']:>15.4f}  (1.0 = fairness is free)")
    if s["Max G-PoF"] is not None:
        print(f"  Max G-PoF    : {s['Max G-PoF']:>15.4f}")
        print(f"  Avg G-PoF    : {s['Avg G-PoF']:>15.4f}")
    print(f"  Silhouette   : {s['Silhouette']:>15.4f}")
    if s["Group_PoFs"]:
        print(f"\n  Per-group G-PoF:")
        for g, v in s["Group_PoFs"].items():
            print(f"    {str(g):20s}: {v:.4f}")
    print(f"{'='*55}\n")


def compare(summaries: list[dict]) -> pd.DataFrame:
    """
    Build a tidy comparison DataFrame from a list of evaluate() outputs.

    Usage::

        df_cmp = compare([eval_bera, eval_boehm, eval_essential])
        print(df_cmp.to_string(index=False))
    """
    rows = []
    for s in summaries:
        rows.append({
            "Algorithm":   s["Algorithm"],
            "Fair Cost":   s["Total Cost (Fair)"],
            "Unfair Cost": s["Total Cost (Unfair)"],
            "PoF":         s["PoF"],
            "Max G-PoF":   s.get("Max G-PoF"),
            "Avg G-PoF":   s.get("Avg G-PoF"),
            "Silhouette":  s["Silhouette"],
        })
    return pd.DataFrame(rows)



_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def plot_execution_times(
    timing: dict,
    title:  str = "Execution Time by Step",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of per-step wall-clock times."""
    steps = list(timing.keys())
    times = list(timing.values())
    max_t = max(times) if times else 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = (_PALETTE * ((len(steps) // len(_PALETTE)) + 1))[:len(steps)]
    bars = ax.bar(steps, times, color=colors)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03 * max_t,
            f"{t:.2f}s",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_spatial_clusters(
    df:           pd.DataFrame,
    result:       ClusteringResult,
    feature_cols: list[str],
    group_col:    str,
    weight_col:   Optional[str] = "Weight",
    save_path:    Optional[str] = None,
) -> None:
    """
    Two-panel plot:
      Left  — scatter of points coloured by cluster, centres marked with ✕
      Right — stacked bar of (normalised) group composition per cluster
    """
    lat_col, lon_col = feature_cols
    df_vis = df.copy()
    df_vis["_Cluster"] = result.labels

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Cluster Visualisation — {result.algorithm}", fontsize=13)

    # --- left: spatial scatter ---
    ax = axes[0]
    sns.scatterplot(
        data=df_vis, x=lon_col, y=lat_col,
        hue="_Cluster", palette="tab10",
        alpha=0.5, s=12, legend=False, ax=ax,
    )
    ax.scatter(
        result.centers[:, 1], result.centers[:, 0],
        c="red", marker="X", s=120, zorder=5, label="Centers",
    )
    ax.set_title("Spatial Distribution")
    ax.set_xlabel("Longitude (scaled)")
    ax.set_ylabel("Latitude (scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- right: group composition ---
    ax = axes[1]
    w_col = weight_col if (weight_col and weight_col in df_vis.columns) else None
    if w_col:
        comp = (
            df_vis.groupby(["_Cluster", group_col])[w_col]
            .sum()
            .unstack(fill_value=0)
        )
    else:
        comp = (
            df_vis.groupby(["_Cluster", group_col])
            .size()
            .unstack(fill_value=0)
        )
    comp_norm = comp.div(comp.sum(axis=1), axis=0)
    comp_norm.plot(kind="bar", stacked=True, ax=ax,
                   colormap="tab10", legend=True)
    ax.set_title("Group Composition per Cluster (normalised)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Fraction")
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pof_comparison(
    summaries:  list[dict],
    save_path:  Optional[str] = None,
) -> None:
    """
    Side-by-side bar chart of PoF and Avg G-PoF for all evaluated algorithms.
    The dashed red line at y=1 marks the unfair baseline.
    """
    df = compare(summaries)

    x      = np.arange(len(df))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, df["PoF"],        width, label="PoF (overall)",  color=_PALETTE[0])
    ax.bar(x + width / 2, df["Avg G-PoF"],  width, label="Avg G-PoF",      color=_PALETTE[1])
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Unfair baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], rotation=10)
    ax.set_ylabel("Price of Fairness ratio")
    ax.set_title("Algorithm Comparison — Price of Fairness")
    ax.legend()
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_group_pof(
    summaries:  list[dict],
    save_path:  Optional[str] = None,
) -> None:
    """
    Grouped bar chart showing the per-group G-PoF for every algorithm.
    Useful for spotting which demographic group bears the fairness cost.
    """
    # Collect all group names
    all_groups = []
    for s in summaries:
        for g in s.get("Group_PoFs", {}):
            if g not in all_groups:
                all_groups.append(g)

    if not all_groups:
        print("[plot_group_pof] No group-level G-PoF data available.")
        return

    n_algs   = len(summaries)
    n_groups = len(all_groups)
    x        = np.arange(n_groups)
    width    = 0.8 / n_algs

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))
    for i, s in enumerate(summaries):
        vals = [s["Group_PoFs"].get(g, 0.0) for g in all_groups]
        offset = (i - n_algs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=s["Algorithm"],
               color=_PALETTE[i % len(_PALETTE)])

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Unfair baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in all_groups], rotation=20, ha="right")
    ax.set_ylabel("G-PoF")
    ax.set_title("Per-group Price of Fairness by Algorithm")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_cost_breakdown(
    summaries:  list[dict],
    save_path:  Optional[str] = None,
) -> None:
    """
    Stacked bar of per-group costs for each algorithm, normalised to the
    unfair total cost so the y-axis reads directly as PoF.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.5
    x = np.arange(len(summaries))

    # Gather all group names (preserving order from first algorithm)
    all_groups = []
    for s in summaries:
        for g in s.get("Group_Costs", {}):
            if g not in all_groups:
                all_groups.append(g)

    bottoms = np.zeros(len(summaries))
    for gi, g in enumerate(all_groups):
        vals = []
        for s in summaries:
            gc = s.get("Group_Costs", {})
            normalised = gc.get(g, 0.0) / s["Total Cost (Unfair)"] if s["Total Cost (Unfair)"] else 0.0
            vals.append(normalised)
        ax.bar(x, vals, bar_width, bottom=bottoms,
               label=str(g), color=_PALETTE[gi % len(_PALETTE)])
        bottoms += np.array(vals)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Unfair baseline (PoF=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([s["Algorithm"] for s in summaries], rotation=10)
    ax.set_ylabel("Normalised cost (÷ unfair total)")
    ax.set_title("Per-group Cost Breakdown by Algorithm")
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()