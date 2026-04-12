import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from results_encoder import save_summary

from runner import run_trials, build_bera_result, build_bercea_result, build_boehm_result


def plot_algorithm_pof_comparison(
    summaries: list[dict]
) -> None:
    labels = [s["Algorithm"] for s in summaries]
    pof_means = [s["All results PoF (mean)"] for s in summaries]
    pof_stds = [s["All results PoF (std)"] for s in summaries]

    _PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 5))

    bars = ax.bar(
        x, pof_means, yerr=pof_stds,
        capsize=5, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))],
        edgecolor="black", linewidth=0.6, zorder=3,
    )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Unfair baseline - PoF = 1")

    for bar, mean, std in zip(bars, pof_means, pof_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean}\n±{std}",
            ha="center", va="bottom", fontsize=9, color="dimgray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.legend(loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    ymax = max(m + s for m, s in zip(pof_means, pof_stds))
    ax.set_ylim(bottom=min(0.95, min(pof_means) - 0.05), top=ymax * 1.12)

    fig.tight_layout()

    fig.savefig("evaluation1_pof_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()

    header = (
        f"{'Algorithm'} {'N runs'}  "
        f"{'Fair Cost'}  {'Unfair Cost'}  "
        f"{'PoF mean'}  {'PoF std'}  "
        f"{'PoF min'}  {'PoF max'}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "Algorithm,N_runs,Fair_Cost_mean,Fair_Cost_std,"
        "Unfair_Cost_mean,Unfair_Cost_std,"
        "PoF_mean,PoF_std,PoF_min,PoF_max,Effective_n"
    ]

    for s in summaries:
        alg = s["Algorithm"]
        n_runs = s["number of runs"]
        fc_m = s["All results Fair Cost (mean)"]
        fc_s = s["All results Fair Cost (std)"]
        uc_m = s["All results Unfair Cost (mean)"]
        uc_s = s["All results Unfair Cost (std)"]
        pof_m = s["All results PoF (mean)"]
        pof_s = s["All results PoF (std)"]
        pof_min = s["All results PoF (min)"]
        pof_max = s["All results PoF (max)"]

        print(
            f"{alg} {n_runs}  "
            f"{fc_m}  {uc_m}  "
            f"{pof_m}  {pof_s}  "
            f"{pof_min}  {pof_max}"
        )

        csv_lines.append(
            f"{alg},{n_runs},{fc_m},{fc_s},"
            f"{uc_m},{uc_s},"
            f"{pof_m},{pof_s},{pof_min},{pof_max}"
        )

    print(sep)

if __name__ == "__main__":
    ROW_SIZE = 100_0

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    GROUP_ID_FEATURES = ["RACE_BINARY"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 3
    ALPHA = 0.05
    print("  RUNNING BERCEA ET AL. (Proportional Bounds)")
    print("=" * 60)

    bera_result, bera_summary = run_trials(
        max_rows=ROW_SIZE,
        algorithm_fn=bera_fc,
        result_builder=build_bera_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        # Bera specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_centers=K,
        alpha=ALPHA,
        weight_col=None
    )

    print("  RUNNING BERA ET AL. (Iterative Rounding)")
    print("=" * 60)
    bercea_result, bercea_summary = run_trials(
        max_rows=ROW_SIZE,
        algorithm_fn=bercea_fc,
        result_builder=build_bercea_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        # Bercea specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_cluster=K,
        alpha=ALPHA,
        weight_col=None
    )

    print("\n" + "=" * 60)
    print("  RUNNING BÖHM ET AL. (Weighted Exact Balance)")
    print("=" * 60)
    boehm_result, boehm_summary = run_trials(
        max_rows=ROW_SIZE,
        algorithm_fn=boehm_fc,
        result_builder=build_boehm_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        # Böhm specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k=K,
        kmedian_trials=3,
        kmedian_max_iter=30
    )

    save_summary(bera_summary, 'evaluation1_bera_summary')
    save_summary(bercea_summary, 'evaluation1_bercea_summary')
    save_summary(boehm_summary, 'evaluation1_boehm_summary')


    plot_algorithm_pof_comparison([bera_summary, bercea_summary, boehm_summary])
