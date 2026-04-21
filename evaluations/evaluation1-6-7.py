import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc
from results_encoder import save_summary
from runner import run_trials, build_bera_result, build_bercea_result, build_boehm_result, build_backurs_result


ALG_PALETTE = {
    "Bera": "#4C72B0",
    "Bercea": "#DD8452",
    "Böhm": "#55A868",
    "Backurs": "#AAA868",
}
PALETTE_LIST = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                 "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def plot_eval1_pof_bar(summaries: dict[str, dict]) -> None:
    """Bar chart of PoF ± std per algorithm."""
    labels = list(summaries.keys())
    pof_means = [summaries[a]["All results PoF (mean)"] for a in labels]
    pof_stds = [summaries[a]["All results PoF (std)"] for a in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 5))
    colors = [ALG_PALETTE.get(a, "gray") for a in labels]

    bars = ax.bar(x, pof_means, yerr=pof_stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Unfair baseline — PoF = 1")

    for bar, m, s in zip(bars, pof_means, pof_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f"{m:.4f}\n±{s:.4f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_title(f"Eval 1 — Overall PoF Comparison  (RACE_6, n={N_SIZE}, k={K}, α={ALPHA})",
                 fontsize=12)
    ax.legend(loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    ymax = max(m + s for m, s in zip(pof_means, pof_stds))
    ax.set_ylim(bottom=min(0.95, min(pof_means) - 0.05), top=ymax * 1.12)

    fig.tight_layout()
    fig.savefig("eval1_pof_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved eval1_pof_bar_chart.png")


def print_eval1_table(summaries: dict[str, dict]) -> None:
    header = (
        f"{'Algorithm':<12s}  {'N':>3s}  "
        f"{'Fair Cost':>16s}  {'Unfair Cost':>16s}  "
        f"{'PoF mean':>12s}  {'PoF std':>10s}  "
        f"{'PoF min':>10s}  {'PoF max':>10s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "Algorithm,N_runs,Fair_Cost_mean,Fair_Cost_std,"
        "Unfair_Cost_mean,Unfair_Cost_std,"
        "PoF_mean,PoF_std,PoF_min,PoF_max"
    ]

    for alg, s in summaries.items():
        n = s["number of runs"]
        fc_m = s["All results Fair Cost (mean)"]
        fc_s = s["All results Fair Cost (std)"]
        uc_m = s["All results Unfair Cost (mean)"]
        uc_s = s["All results Unfair Cost (std)"]
        pm = s["All results PoF (mean)"]
        ps = s["All results PoF (std)"]
        pmin = s["All results PoF (min)"]
        pmax = s["All results PoF (max)"]

        print(
            f"{alg:<12s}  {n:>3d}  "
            f"{fc_m:>12,.0f}±{fc_s:>3,.0f}  {uc_m:>12,.0f}±{uc_s:>3,.0f}  "
            f"{pm:>12.4f}  {ps:>10.4f}  "
            f"{pmin:>10.4f}  {pmax:>10.4f}"
        )
        csv_lines.append(
            f"{alg},{n},{fc_m:.2f},{fc_s:.2f},"
            f"{uc_m:.2f},{uc_s:.2f},"
            f"{pm:.6f},{ps:.6f},{pmin:.6f},{pmax:.6f}"
        )

    print(sep)
    csv_path = "./evaluation1-6-7_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  Results saved to {csv_path}")


def plot_eval6_cpof_mean_bar(summaries: dict[str, dict]) -> None:
    """Per-cluster C-PoF bar chart averaged across all runs."""
    algs = [a for a in ["Bera", "Bercea"] if a in summaries]
    if not algs:
        return

    cluster_ids = sorted(summaries[algs[0]]["_all_cpofs"][0].keys())
    n_clusters = len(cluster_ids)
    n_algs = len(algs)
    width = 0.8 / n_algs
    x = np.arange(n_clusters)

    fig, ax = plt.subplots(figsize=(max(10, n_clusters * 1.2), 5.5))

    for i, alg in enumerate(algs):
        all_cpofs = summaries[alg]["_all_cpofs"]

        means, stds = [], []
        for cid in cluster_ids:
            vals = [d.get(cid, float("inf")) for d in all_cpofs]
            valid = [v for v in vals if v != float("inf")]
            means.append(np.mean(valid) if valid else 0.0)
            stds.append(np.std(valid, ddof=1) if len(valid) > 1 else 0.0)

        offset = (i - n_algs / 2 + 0.5) * width
        color = ALG_PALETTE.get(alg, "gray")

        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=alg, color=color, edgecolor="black",
                      linewidth=0.3, zorder=3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.01,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7, color="dimgray")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="C-PoF = 1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{j}" for j in cluster_ids])
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("C-PoF  (fair cost / unfair cost)", fontsize=11)
    ax.set_title(f"Eval 6 — Mean Per-Cluster PoF (across {N_RUNS} runs)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_cpof_mean_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation1-6-7_cpof_mean_bar.png")


def plot_eval6_cpof_pooled_histogram(summaries: dict[str, dict]) -> None:
    """Histogram of pooled C-PoF values across all runs × clusters."""
    algs = [a for a in ["Bera", "Bercea"] if a in summaries]
    if not algs:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for alg in algs:
        pooled = summaries[alg]["Pooled C-PoF values"]
        if not pooled:
            continue
        color = ALG_PALETTE.get(alg, "gray")
        ax.hist(pooled, bins=30, alpha=0.55, color=color, label=alg,
                edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(pooled), color=color, linestyle="--", linewidth=1.5,
                   alpha=0.8)

    ax.axvline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5,
               label="C-PoF = 1")
    ax.set_xlabel("C-PoF", fontsize=11)
    ax.set_ylabel("Count (runs × clusters)", fontsize=11)
    ax.set_title("Eval 6 — Pooled C-PoF Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_cpof_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation1-6-7_histogram.png")


def plot_eval6_cpof_spread_gini(summaries: dict[str, dict]) -> None:
    """Box plots of per-run C-PoF spread and Gini coefficient."""
    algs = [a for a in ["Bera", "Bercea"]]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    spread_data = [summaries[a]["C-PoF spreads"] for a in algs]
    bp1 = ax1.boxplot(spread_data, tick_labels=algs, patch_artist=True)
    for patch, alg in zip(bp1["boxes"], algs):
        patch.set_facecolor(ALG_PALETTE.get(alg, "gray"))
        patch.set_alpha(0.6)
    ax1.set_ylabel("max(C-PoF) − min(C-PoF)", fontsize=10)
    ax1.set_title("Per-Run C-PoF Spread", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    fig, ax2 = plt.subplots(figsize=(12, 5))
    gini_data = [summaries[a]["C-PoF Ginis"] for a in algs]
    bp2 = ax2.boxplot(gini_data, tick_labels=algs, patch_artist=True)
    for patch, alg in zip(bp2["boxes"], algs):
        patch.set_facecolor(ALG_PALETTE.get(alg, "gray"))
        patch.set_alpha(0.6)
    ax2.set_ylabel("Gini coefficient", fontsize=10)
    ax2.set_title("Per-Run C-PoF Gini", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Eval 6 — C-PoF Inequality Across Runs", fontsize=12)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_cpof_spread_gini.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation1-6-7_cpof_spread_gini.png")


def print_eval6_table(summaries: dict[str, dict]) -> None:
    algs = [a for a in ["Bera", "Bercea"] if a in summaries]

    csv_lines = [
        "Algorithm,Cluster,C-PoF_mean,C-PoF_std,"
        "FairCost_mean,FairCost_std,UnfairCost_mean,UnfairCost_std"
    ]

    for alg in algs:
        all_cpofs = summaries[alg]["_all_cpofs"]
        all_fc = summaries[alg]["_all_cluster_fair_costs"]
        all_uc = summaries[alg]["_all_cluster_unfair_costs"]
        cluster_ids = sorted(all_cpofs[0].keys())

        print(f"\n  {alg} — Per-Cluster C-PoF (mean ± std across {len(all_cpofs)} runs)")
        print(f"  {'Cluster':>8s}  {'C-PoF':>16s}  {'Fair Cost':>14s}  {'Unfair Cost':>14s}")
        print(f"  {'-' * 60}")

        for cid in cluster_ids:
            cpof_vals = [d.get(cid, float('inf')) for d in all_cpofs]
            fc_vals = [d.get(cid, 0.0) for d in all_fc]
            uc_vals = [d.get(cid, 0.0) for d in all_uc]
            valid_cpof = [v for v in cpof_vals if v != float('inf')]

            cm = np.mean(valid_cpof) if valid_cpof else float('inf')
            cs = np.std(valid_cpof, ddof=1) if len(valid_cpof) > 1 else 0.0
            fm = np.mean(fc_vals)
            fs = np.std(fc_vals, ddof=1) if len(fc_vals) > 1 else 0.0
            um = np.mean(uc_vals)
            us = np.std(uc_vals, ddof=1) if len(uc_vals) > 1 else 0.0

            print(f"  C{cid:>6d}  {cm:>7.4f}±{cs:.4f}  {fm:>9,.0f}±{fs:>4,.0f}  {um:>9,.0f}±{us:>4,.0f}")
            csv_lines.append(f"{alg},{cid},{cm:.6f},{cs:.6f},{fm:.2f},{fs:.2f},{um:.2f},{us:.2f}")

        spreads = summaries[alg]["C-PoF spreads"]
        ginis = summaries[alg]["C-PoF Ginis"]
        print(f"  Spread: {np.mean(spreads):.4f} ± {np.std(spreads, ddof=1) if len(spreads) > 1 else 0:.4f}")
        print(f"  Gini:   {np.mean(ginis):.4f} ± {np.std(ginis, ddof=1) if len(ginis) > 1 else 0:.4f}")

    csv_path = "./evaluation1-6-7_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")



def plot_eval7_gpof_bar(summaries: dict[str, dict]) -> None:
    """Grouped bar chart of mean G-PoF per group, per algorithm."""
    algs = list(summaries.keys())
    group_names = list(summaries[algs[0]]["G-PoF means"].keys())
    n_groups = len(group_names)
    n_algs = len(algs)
    width = 0.8 / n_algs
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2), 5.5))

    for i, alg in enumerate(algs):
        means = [summaries[alg]["G-PoF means"].get(g, 0.0) for g in group_names]
        stds = [summaries[alg]["G-PoF stds"].get(g, 0.0) for g in group_names]
        offset = (i - n_algs / 2 + 0.5) * width
        color = ALG_PALETTE.get(alg, PALETTE_LIST[i % len(PALETTE_LIST)])

        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=alg, color=color, edgecolor="black", linewidth=0.3,
                      zorder=3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7,
                    color="dimgray")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="G-PoF = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("G-PoF  (group fair cost / group unfair cost)", fontsize=10)
    ax.set_title("Eval 7 — Per-Group PoF (mean ± std across runs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_gpof_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation1-6-7_gpof_bar.png")


def plot_eval7_gpof_spread_gini(summaries: dict[str, dict]) -> None:
    """Box plots of per-run G-PoF equity spread and Gini coefficient."""
    algs = list(summaries.keys())

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    spread_data = [summaries[a]["G-PoF spreads"] for a in algs]
    bp1 = ax1.boxplot(spread_data, tick_labels=algs, patch_artist=True)
    for patch, alg in zip(bp1["boxes"], algs):
        patch.set_facecolor(ALG_PALETTE.get(alg, "gray"))
        patch.set_alpha(0.6)
    ax1.set_ylabel("max(G-PoF) − min(G-PoF)", fontsize=10)
    ax1.set_title("Per-Run G-PoF Equity Spread", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_gpof_spread.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 5))
    gini_data = [summaries[a]["G-PoF Ginis"] for a in algs]
    bp2 = ax2.boxplot(gini_data, tick_labels=algs, patch_artist=True)
    for patch, alg in zip(bp2["boxes"], algs):
        patch.set_facecolor(ALG_PALETTE.get(alg, "gray"))
        patch.set_alpha(0.6)
    ax2.set_ylabel("Gini coefficient", fontsize=10)
    ax2.set_title("Per-Run G-PoF Gini", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("evaluation1-6-7_gpof_gini.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation1-6-7_gpof_spread_gini.png")


def plot_eval7_gpof_per_run_heatmap(summaries: dict[str, dict]) -> None:
    """
    Per-run G-PoF per group as a heatmap for each algorithm.
    Shows how stable group-level fairness costs are across random samples.
    """
    algs = [a for a in ["Bera", "Bercea", "Böhm", "Backurs"] if a in summaries]
    if not algs:
        return

    for idx, alg in enumerate(algs):
        fig, ax = plt.subplots(figsize=(7, 5))
        all_gpofs = summaries[alg]["_all_gpofs"]
        group_names = list(all_gpofs[0].keys())
        n_runs = len(all_gpofs)

        matrix = np.zeros((n_runs, len(group_names)))
        for r, gpof_dict in enumerate(all_gpofs):
            for g_idx, g in enumerate(group_names):
                val = gpof_dict.get(g, float('inf'))
                matrix[r, g_idx] = val if val != float('inf') else np.nan

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest")
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Run #", fontsize=10)
        ax.set_title(f"{alg}", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="G-PoF")

        fig.suptitle(f"Eval 7 — Per-Run G-PoF Heatmap {alg}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"evaluation1-6-7-{alg}_gpof_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved evaluation1-6-7-{alg}_gpof_heatmap.png")


def print_eval7_table(summaries: dict[str, dict]) -> None:
    algs = list(summaries.keys())
    group_names = list(summaries[algs[0]]["G-PoF means"].keys())

    csv_lines = [
        "Algorithm,Group,G-PoF_mean,G-PoF_std,"
        "FairGroupCost_mean,FairGroupCost_std,"
        "UnfairGroupCost_mean,UnfairGroupCost_std"
    ]

    for alg in algs:
        all_gpofs = summaries[alg]["_all_gpofs"]
        all_gc_fair = summaries[alg]["_all_group_costs_fair"]
        all_gc_unfair = summaries[alg]["_all_group_costs_unfair"]

        print(f"\n  {alg} — Per-Group G-PoF (mean ± std across {len(all_gpofs)} runs)")
        print(f"  {'Group':<20s}  {'G-PoF':>16s}  {'Fair Cost':>16s}  {'Unfair Cost':>16s}")
        print(f"  {'-' * 72}")

        for g in group_names:
            gpof_vals = [d.get(g, float('inf')) for d in all_gpofs]
            fc_vals = [d.get(g, 0.0) for d in all_gc_fair]
            uc_vals = [d.get(g, 0.0) for d in all_gc_unfair]
            valid_gpof = [v for v in gpof_vals if v != float('inf')]

            gm = np.mean(valid_gpof) if valid_gpof else float('inf')
            gs = np.std(valid_gpof, ddof=1) if len(valid_gpof) > 1 else 0.0
            fm = np.mean(fc_vals)
            fs = np.std(fc_vals, ddof=1) if len(fc_vals) > 1 else 0.0
            um = np.mean(uc_vals)
            us = np.std(uc_vals, ddof=1) if len(uc_vals) > 1 else 0.0

            print(f"  {str(g):<20s}  {gm:>7.4f}±{gs:.4f}  {fm:>10,.0f}±{fs:>5,.0f}  {um:>10,.0f}±{us:>5,.0f}")
            csv_lines.append(f"{alg},{g},{gm:.6f},{gs:.6f},{fm:.2f},{fs:.2f},{um:.2f},{us:.2f}")

        spreads = summaries[alg]["G-PoF spreads"]
        ginis = summaries[alg]["G-PoF Ginis"]
        print(f"  Equity Spread: {np.mean(spreads):.4f} ± {np.std(spreads, ddof=1) if len(spreads) > 1 else 0:.4f}")
        print(f"  Gini:          {np.mean(ginis):.4f} ± {np.std(ginis, ddof=1) if len(ginis) > 1 else 0:.4f}")

    csv_path = "./evaluation1-6-7_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    N_SIZE = 1000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    GROUP_ID_FEATURES = ["RACE_6"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    ALPHA = 0.05
    N_RUNS = 3

    summaries: dict[str, dict] = {}

    print(f"\n{'#' * 60}")
    print(f"  Running Bera")
    print(f"{'#' * 60}")
    bera_summary = run_trials(
        max_rows=N_SIZE,
        algorithm_fn=bera_fc,
        result_builder=build_bera_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_centers=K,
        alpha=ALPHA,
        weight_col=None,
    )
    summaries["Bera"] = bera_summary
    save_summary(bera_summary, "eval167_bera_summary.json")

    print(f"\n{'#' * 60}")
    print(f"  Running Bercea")
    print(f"{'#' * 60}")
    bercea_summary = run_trials(
        max_rows=N_SIZE,
        algorithm_fn=bercea_fc,
        result_builder=build_bercea_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_cluster=K,
        alpha=ALPHA,
        weight_col=None,
    )
    summaries["Bercea"] = bercea_summary
    save_summary(bercea_summary, "eval167_bercea_summary.json")

    print(f"\n{'#' * 60}")
    print(f"  Running Böhm")
    print(f"{'#' * 60}")
    boehm_summary = run_trials(
        max_rows=N_SIZE,
        algorithm_fn=boehm_fc,
        result_builder=build_boehm_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k=K,
        kmedian_trials=3,
        kmedian_max_iter=30,
    )
    summaries["Böhm"] = boehm_summary
    save_summary(boehm_summary, "eval167_boehm_summary.json")

    print(f"\n{'#' * 60}")
    print(f"  Running Backurs")
    print(f"{'#' * 60}")
    backurs_summary = run_trials(
        max_rows=N_SIZE,
        algorithm_fn=backurs_fc,
        result_builder=build_backurs_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_cluster=K,
        alpha=ALPHA,
    )
    summaries["Backurs"] = backurs_summary
    save_summary(backurs_summary, "eval167_backurs_summary.json")

    print(f"\n{'=' * 60}")
    print("  EVALUATION 1 — Overall PoF Comparison")
    print(f"{'=' * 60}")
    plot_eval1_pof_bar(summaries)
    print_eval1_table(summaries)

    print(f"\n{'=' * 60}")
    print("  EVALUATION 6 — Per-Cluster C-PoF")
    print(f"{'=' * 60}")
    plot_eval6_cpof_mean_bar(summaries)
    plot_eval6_cpof_pooled_histogram(summaries)
    plot_eval6_cpof_spread_gini(summaries)
    print_eval6_table(summaries)

    print(f"\n{'=' * 60}")
    print("  EVALUATION 7 — Per-Group G-PoF")
    print(f"{'=' * 60}")
    plot_eval7_gpof_bar(summaries)
    plot_eval7_gpof_spread_gini(summaries)
    plot_eval7_gpof_per_run_heatmap(summaries)
    print_eval7_table(summaries)