from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist


@dataclass
class ClusteringResult:
    algorithm: str
    centers: np.ndarray
    labels: np.ndarray
    fair_cost: float
    unfair_cost: float
    X: np.ndarray  # (n, d) — point coordinates (no coreset assumed)
    weights: np.ndarray
    group_codes: np.ndarray
    group_names: list
    timing: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.centers)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def pof(self) -> float:
        return compute_pof(self.fair_cost, self.unfair_cost)


def make_result(
        algorithm: str,
        centers: np.ndarray,
        labels: np.ndarray,
        fair_cost: float,
        unfair_cost: float,
        X: np.ndarray,
        weights: np.ndarray,
        group_codes: np.ndarray,
        group_names: list,
        timing: Optional[dict] = None,
) -> ClusteringResult:
    return ClusteringResult(
        algorithm=algorithm,
        centers=centers,
        labels=labels,
        fair_cost=fair_cost,
        unfair_cost=unfair_cost,
        X=X,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing or {},
    )


def compute_pof(fair_cost: float, unfair_cost: float) -> float:
    if unfair_cost == 0:
        return float("inf")
    return fair_cost / unfair_cost


def compute_group_costs(result: ClusteringResult) -> dict:
    """
    weighted L1 assignment cost broken down by demographic group.
    Uses result.X and result.centers
    """
    X = result.X
    n = len(X)
    weighted_dists = np.zeros(n)

    for j in range(result.k):
        mask = result.labels == j
        if mask.any():
            dists = np.sum(np.abs(X[mask] - result.centers[j]), axis=1)
            weighted_dists[mask] = dists * result.weights[mask]

    return {
        name: float(weighted_dists[result.group_codes == h].sum())
        for h, name in enumerate(result.group_names)
    }


def compute_gpof(
        fair_group_costs: dict,
        unfair_group_costs: dict,
) -> dict:
    """
    Group-level Price of Fairness.

    G-PoF(g) = fair_cost(g) / unfair_cost(g)
    """
    gpof = {}
    for g, fc in fair_group_costs.items():
        uc = unfair_group_costs.get(g, 0.0)
        gpof[g] = fc / uc if uc > 0 else float("inf")
    return gpof


def compute_cluster_costs(result: ClusteringResult) -> dict:
    """
    weighted L1 assignment cost broken down by cluster.
    """
    X = result.X
    costs = {}
    for center in range(result.k):
        mask = result.labels == center
        if not mask.any():
            costs[center] = 0.0
            continue
        dists = np.sum(np.abs(X[mask] - result.centers[center]), axis=1)
        costs[center] = float((dists * result.weights[mask]).sum())
    return costs


def _match_clusters(fair_result: ClusteringResult, unfair_result: ClusteringResult) -> dict:
    """
    Match fair clusteers to unfair clusters by nearest centre (L1).
    This is needed because the two runs may number their clusters differently.
    Simple greedy nearest-centre matching — good enough for PoF
    comparisons where k is small.
    """
    D = cdist(fair_result.centers, unfair_result.centers, metric="cityblock")
    # greedy: assign each fair cluster to its closest unfair cluster
    # (many-to-one is fine for PoF — we just want the right cost baseline)
    return {j: int(np.argmin(D[j])) for j in range(fair_result.k)}


def compute_cluster_pof(
        fair_result: ClusteringResult,
        unfair_result: ClusteringResult,
) -> dict:
    """
    Per-cluster Price of Fairness.
    Clusters are matched by nearest centre between the two runs, then:
        C-PoF(j) = fair_cluster_cost(j) / unfair_cluster_cost(matched_j)
    """
    fair_costs = compute_cluster_costs(fair_result)
    unfair_costs = compute_cluster_costs(unfair_result)
    matching = _match_clusters(fair_result, unfair_result)

    cpof = {}
    for j, uc_j in matching.items():
        fc = fair_costs[j]
        uc = unfair_costs.get(uc_j, 0.0)
        cpof[j] = fc / uc if uc > 0 else float("inf")
    return cpof


def audit_fairness_proportional(
        result: ClusteringResult,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        verbose: bool = True,
) -> int:
    """
    flags any cluster where the weighted fraction of group h falls outside
    [lower_bounds[h] - ε, upper_bounds[h] + ε].
    """
    violations = 0
    tol = 1e-4

    if verbose:
        print(f"\n[Fairness Audit — Proportional] '{result.algorithm}'")

    for j in range(result.k):
        at_j = result.labels == j
        total_j = result.weights[at_j].sum()
        if total_j == 0:
            continue
        for h, name in enumerate(result.group_names):
            mass_h = result.weights[at_j & (result.group_codes == h)].sum()
            frac = mass_h / total_j
            lo, hi = lower_bounds[h], upper_bounds[h]
            if frac < lo - tol or frac > hi + tol:
                violations += 1
                if verbose:
                    print(f"  ⚠ Cluster {j:2d}  Group '{name}': "
                          f"{frac * 100:.1f}% outside [{lo * 100:.1f}%, {hi * 100:.1f}%]")

    if verbose:
        if violations == 0:
            print("  ✓ All clusters satisfy proportional fairness bounds.")
        else:
            print(f"  → {violations} violation(s)  "
                  "(additive violations ≤ 1 are expected per Lemma 7 / Bera et al.)")
    return violations


def audit_fairness_exact_balance(
        result: ClusteringResult,
) -> int:
    """
    every cluster must contain exactly 1/H of each group (count-based).
    returns the total number of (cluster, group) violations.
    """
    H = result.n_groups
    violations = 0
    tol = 1e-4
    expected = 1.0 / H

    print(f"\n[Fairness Audit — Exact Balance] '{result.algorithm}'")

    for j in range(result.k):
        at_j = result.labels == j
        total_j = int(at_j.sum())
        if total_j == 0:
            continue
        for h, name in enumerate(result.group_names):
            count_h = int((at_j & (result.group_codes == h)).sum())
            frac = count_h / total_j
            if abs(frac - expected) > tol:
                violations += 1
                print(f"  ⚠ Cluster {j:2d}  Group '{name}': "
                      f"{count_h} pts ({frac * 100:.1f}%) ≠ {expected * 100:.1f}%")

    if violations == 0:
        print("  ✓ All clusters are perfectly balanced across groups.")
    else:
        print(f"  → {violations} uneven group distribution(s).")
    return violations


def evaluate(
        result: ClusteringResult,
        unfair_result: Optional[ClusteringResult] = None,
        save_csv: bool = True
) -> dict:
    """
    Compute all thesis metrics for one algorithm result.

    Parameters
    ----------
    result             : ClusteringResult from the fair algorithm
    unfair_result      : ClusteringResult from plain k-median (same X).
                         Required for G-PoF.  If None, G-PoF is skipped.

    Returns
    -------
    dict with keys:
        Algorithm, Total Cost (Fair), Total Cost (Unfair), PoF,
        Max G-PoF, Avg G-PoF, Group_Costs, Group_PoFs
    """
    fair_group_costs = compute_group_costs(result)
    fair_cluster_costs = compute_cluster_costs(result)
    total_fair = sum(fair_group_costs.values())

    gpof = {}
    cpof = {}
    unfair_cluster_costs = {}
    unfair_group_costs = {}
    if unfair_result is not None:
        unfair_group_costs = compute_group_costs(unfair_result)
        unfair_cluster_costs = compute_cluster_costs(unfair_result)
        gpof = compute_gpof(fair_group_costs, unfair_group_costs)
        cpof = compute_cluster_pof(result, unfair_result)
    valid_gpof = [v for v in gpof.values() if v != float('inf')] if gpof else []
    valid_cpof = [v for v in cpof.values() if v != float('inf')] if cpof else []
    summary = {
        "Algorithm": result.algorithm,
        "Total Cost (Fair)": total_fair,
        "Total Cost (Unfair)": result.unfair_cost,
        "PoF": compute_pof(total_fair, result.unfair_cost),
        "Max G-PoF": max(valid_gpof),
        "Avg G-PoF": float(np.mean(valid_gpof)),
        "Max C-PoF": max(valid_cpof),
        "Avg C-PoF": float(np.mean(valid_cpof)),
        "Group_Costs (Fair)": fair_group_costs,
        "Group_Costs (Unfair)": unfair_group_costs,
        "Group_PoFs": gpof,
        "Cluster_Costs (Fair)": fair_cluster_costs,
        "Cluster_Costs (Unfair)": unfair_cluster_costs,
        "Cluster_PoFs": cpof,
    }

    _print_summary(summary)

    if save_csv:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)

        flat_summary = {
            "Algorithm": summary["Algorithm"],
            "Total Cost (Fair)": summary["Total Cost (Fair)"],
            "Total Cost (Unfair)": summary["Total Cost (Unfair)"],
            "PoF": summary["PoF"],
            "Max G-PoF": summary["Max G-PoF"],
            "Avg G-PoF": summary["Avg G-PoF"],
            "Max C-PoF": summary["Max C-PoF"],
            "Avg C-PoF": summary["Avg C-PoF"],
        }

        for g, val in summary["Group_PoFs"].items():
            flat_summary[f"G-PoF_{g}"] = val
            flat_summary[f"Fair_Cost_{g}"] = summary["Group_Costs (Fair)"].get(g, None)
            flat_summary[f"Unfair_Cost_{g}"] = summary["Group_Costs (Unfair)"].get(g, None)

        df = pd.DataFrame([flat_summary])
        df.to_csv(save_dir / "evaluation_summary.csv", index=False)

    return summary


def _print_summary(s: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Evaluation — {s['Algorithm']}")
    print(f"{'=' * 55}")
    print(f"  Fair cost    : {s['Total Cost (Fair)']:>15,.2f}")
    print(f"  Unfair cost  : {s['Total Cost (Unfair)']:>15,.2f}")
    print(f"  PoF          : {s['PoF']:>15.4f}  (1.0 = fairness is free)")

    if s["Max G-PoF"] is not None:
        print(f"\n  Group-level G-PoF:")
        print(f"    Max G-PoF  : {s['Max G-PoF']:>10.4f}")
        print(f"    Avg G-PoF  : {s['Avg G-PoF']:>10.4f}")
        for g, v in s["Group_PoFs"].items():
            fair_g = s["Group_Costs (Fair)"].get(g, 0.0)
            unfair_g = s["Group_Costs (Unfair)"].get(g, 0.0)
            print(f"    {str(g):15s}: PoF={v:.4f}  "
                  f"fair={fair_g:,.1f}  unfair={unfair_g:,.1f}")

    if s["Max C-PoF"] is not None:
        print(f"\n  Cluster-level C-PoF:")
        print(f"    Max C-PoF  : {s['Max C-PoF']:>10.4f}")
        print(f"    Avg C-PoF  : {s['Avg C-PoF']:>10.4f}")
        for j, v in s["Cluster_PoFs"].items():
            fair_c = s["Cluster_Costs (Fair)"].get(j, 0.0)
            unfair_c = s["Cluster_Costs (Unfair)"].get(j, 0.0)
            print(f"    Cluster {j:3d}  : PoF={v:.4f}  "
                  f"fair={fair_c:,.1f}  unfair={unfair_c:,.1f}")

    print(f"{'=' * 55}\n")


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
            "Algorithm": s["Algorithm"],
            "Fair Cost": s["Total Cost (Fair)"],
            "Unfair Cost": s["Total Cost (Unfair)"],
            "PoF": s["PoF"],
            "Max G-PoF": s.get("Max G-PoF"),
            "Avg G-PoF": s.get("Avg G-PoF"),
            "Max C-PoF": s.get("Max C-PoF"),
            "Avg C-PoF": s.get("Avg C-PoF"),
        })
    return pd.DataFrame(rows)


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def plot_execution_times(
        result: ClusteringResult,
        timing: dict,
        title: str = "Execution Time by Step",
        save_path: Optional[str] = None
) -> None:
    """Bar chart of per-step times."""

    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "execution_times.png")

    steps = list(timing.keys())
    times = list(timing.values())
    max_t = max(times) if times else 1.0
    colors = (_PALETTE * ((len(steps) // len(_PALETTE)) + 1))[:len(steps)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(steps, times, color=colors)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03 * max_t,
            f"{t:.2f}s", ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_spatial_clusters(
        df: pd.DataFrame,
        result: ClusteringResult,
        feature_cols: list[str],
        group_col: str,
        weight_col: Optional[str] = "Weight",
        save_path: Optional[str] = None,
) -> None:
    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "spatial")

    lat_col, lon_col = feature_cols
    df_vis = df.copy()
    df_vis["_Cluster"] = result.labels

    jitter = 0.008
    df_jit = df_vis.copy()
    df_jit[lon_col] = df_jit[lon_col] + np.random.uniform(-jitter, jitter, len(df_jit))
    df_jit[lat_col] = df_jit[lat_col] + np.random.uniform(-jitter, jitter, len(df_jit))

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df_jit, x=lon_col, y=lat_col, hue="_Cluster", palette="tab10",
        alpha=0.4, s=14, legend="full", ax=ax1,
    )
    ax1.scatter(
        result.centers[:, 1], result.centers[:, 0],
        c="red", marker="X", s=150, zorder=5, label="Centers",
    )
    ax1.set_title(f"By Cluster — {result.algorithm}")
    ax1.set_xlabel("Longitude (scaled)")
    ax1.set_ylabel("Latitude (scaled)")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title="Cluster", fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig1.tight_layout()
    if save_path:
        fig1.savefig(f"{save_path}_cluster.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df_jit, x=lon_col, y=lat_col, hue=group_col, palette="Set2",
        alpha=0.4, s=14, legend="full", ax=ax2,
    )
    ax2.scatter(
        result.centers[:, 1], result.centers[:, 0],
        c="black", marker="X", s=150, zorder=5, label="Centers",
    )
    ax2.set_title(f"By Demographic Group — {result.algorithm}")
    ax2.set_xlabel("Longitude (scaled)")
    ax2.set_ylabel("Latitude (scaled)")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, title=group_col, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    if save_path:
        fig2.savefig(f"{save_path}_group.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Plot 3: Group Composition Stacked Bar ---
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    w_col = weight_col if (weight_col and weight_col in df_vis.columns) else None
    if w_col:
        comp = df_vis.groupby(["_Cluster", group_col])[w_col].sum().unstack(fill_value=0)
    else:
        comp = df_vis.groupby(["_Cluster", group_col]).size().unstack(fill_value=0)

    comp.div(comp.sum(axis=1), axis=0).plot(
        kind="bar", stacked=True, ax=ax3, colormap="Set2", legend=True
    )
    ax3.set_title(f"Group Composition per Cluster — {result.algorithm}")
    ax3.set_xlabel("Cluster ID")
    ax3.set_ylabel("Fraction")
    ax3.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax3.tick_params(axis="x", rotation=0)
    fig3.tight_layout()
    if save_path:
        fig3.savefig(f"{save_path}_composition.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_pof_comparison(
        result: ClusteringResult,
        summaries: list[dict],
        save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side PoF and Avg G-PoF bar chart for all algorithms.
    The dashed red line marks the unfair baseline (PoF = 1).
    """

    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "pof_comparison.png")

    df = compare(summaries)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["PoF"], width, label="PoF (overall)", color=_PALETTE[0])
    ax.bar(x + width / 2, df["Avg G-PoF"], width, label="Avg G-PoF", color=_PALETTE[1])
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
        result: ClusteringResult,
        summaries: list[dict],
        save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart of per-group G-PoF for every algorithm.
    Highlights which demographic group bears the fairness cost.
    """
    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "group_pof.png")
    all_groups: list = []
    for s in summaries:
        for g in s.get("Group_PoFs", {}):
            if g not in all_groups:
                all_groups.append(g)

    if not all_groups:
        print("[plot_group_pof] No group-level G-PoF data — pass unfair_result to evaluate().")
        return

    n_algs = len(summaries)
    x = np.arange(len(all_groups))
    width = 0.8 / n_algs

    fig, ax = plt.subplots(figsize=(max(8, len(all_groups) * 1.5), 5))
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


def plot_cluster_pof(
        result: ClusteringResult,
        summaries: list[dict],
        save_path: Optional[str] = None,
) -> None:
    """
    Per-cluster C-PoF bar chart for each algorithm, with:
      - one group of bars per cluster (x-axis)
      - one bar per algorithm within each group
      - dashed red baseline at PoF = 1
      - fair and unfair cost annotated above each bar

    Only shown for summaries that have Cluster_PoFs data (i.e. unfair_result
    was passed to evaluate()).
    """
    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "cluster_pof.png")
    summaries_with_data = [s for s in summaries if s.get("Cluster_PoFs")]
    if not summaries_with_data:
        print("[plot_cluster_pof] No cluster-level PoF data — "
              "pass unfair_result to evaluate().")
        return

    cluster_ids = sorted(summaries_with_data[0]["Cluster_PoFs"].keys())
    n_clusters = len(cluster_ids)
    n_algs = len(summaries_with_data)
    width = 0.8 / n_algs
    x = np.arange(n_clusters)

    fig, ax = plt.subplots(figsize=(max(10, n_clusters * 0.6), 5))

    for i, s in enumerate(summaries_with_data):
        cpof = s["Cluster_PoFs"]
        fc_map = s["Cluster_Costs (Fair)"]
        uc_map = s["Cluster_Costs (Unfair)"]
        vals = [cpof.get(j, 0.0) for j in cluster_ids]
        offset = (i - n_algs / 2 + 0.5) * width

        bars = ax.bar(x + offset, vals, width,
                      label=s["Algorithm"], color=_PALETTE[i % len(_PALETTE)])

        for bar, j in zip(bars, cluster_ids):
            fc = fc_map.get(j, 0.0)
            uc = uc_map.get(j, 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{fc:,.0f}\n({uc:,.0f})",
                ha="center", va="bottom", fontsize=6, color="dimgray",
                linespacing=1.2,
            )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Unfair baseline (C-PoF = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{j}" for j in cluster_ids], rotation=0)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("C-PoF  (fair cost / unfair cost)")
    ax.set_title("Per-cluster Price of Fairness\n"
                 "(bar annotation: fair cost  /  unfair cost)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_cost_breakdown(
        result: ClusteringResult,
        summaries: list[dict],
        save_path: Optional[str] = None,
) -> None:
    """
    Stacked bar of per-group costs normalised to the unfair total cost,
    so the y-axis reads directly as contribution to the PoF ratio.
    """
    if not save_path:
        save_dir = Path("results") / result.algorithm
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / "cost_breakdown.png")
    all_groups: list = []
    for s in summaries:
        for g in s.get("Group_Costs (Fair)", {}):
            if g not in all_groups:
                all_groups.append(g)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.1
    x = np.arange(len(summaries))
    bottoms = np.zeros(len(summaries))

    for gi, g in enumerate(all_groups):
        vals = []
        for s in summaries:
            gc = s.get("Group_Costs (Fair)", {})
            uc = s["Total Cost (Unfair)"]
            vals.append(gc.get(g, 0.0) / uc if uc else 0.0)
        ax.bar(x, vals, bar_width, bottom=bottoms, label=str(g),
               color=_PALETTE[gi % len(_PALETTE)])
        bottoms += np.array(vals)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Unfair baseline (PoF=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([s["Algorithm"] for s in summaries], rotation=10)
    ax.set_ylabel("Normalised cost (÷ unfair total)")
    ax.set_title("Per-group Cost Breakdown by Algorithm")
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
