import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from fair_clustering.algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from fair_clustering.algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from fair_clustering.algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from fair_clustering.algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc

from fair_clustering.runner import run_trials, build_bera_result, build_bercea_result, build_boehm_result, \
    build_backurs_result
BERA_PHASES = [
    "Data Preparation",
    "Vanilla K-Median",
    "Solve Initial LP",
    "Iterative Rounding",
    "Cost Calculation",
]
BERCEA_PHASES = [
    "Data Preparation",
    "Vanilla K-Median",
    "Solve Initial LP",
    "MCF Rounding",
    "Cost Calculation",
]
BOEHM_PHASES = [
    "Balance Dataset",
    "Vanilla k-Median",
    "Böhm Fair Clustering",
]

BACKURS_PHASES = [
    "Data Preparation",
    "Vanilla K-Median",
    "Base Selection",
    "HST Construction",
    "Fairlet Decomposition",
    "Cluster Fairlets",
]

PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
    "backurs": "#AAA868",
}


def plot_per_algorithm_phases(
    alg_name: str,
    phase_keys: list[str],
    sizes: list[int],
    summaries_by_size: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, phase in enumerate(phase_keys):
        means = []
        stds = []
        for s in summaries_by_size:
            timings_list = s["_timings"]
            vals = [t.get(phase, 0.0) for t in timings_list]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        color = PHASE_PALETTE[idx % len(PHASE_PALETTE)]
        ax.plot(sizes, means, marker="o", label=phase, color=color)
        ax.fill_between(
            sizes,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    total_means = []
    total_stds = []
    for s in summaries_by_size:
        timings_list = s["_timings"]
        vals = [t.get("Total Time", 0.0) for t in timings_list]
        total_means.append(float(np.mean(vals)))
        total_stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
    ax.plot(sizes, total_means, marker="s", linestyle="--", linewidth=2,
            label="Total", color="black")
    ax.fill_between(
        sizes,
        [m - s for m, s in zip(total_means, total_stds)],
        [m + s for m, s in zip(total_means, total_stds)],
        alpha=0.10, color="black",
    )

    ax.set_xlabel("Dataset size (n)", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_title(f"{alg_name} — Per-Phase Scalability", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fname = f"evaluation2_scalability_{alg_name.lower().replace(' ', '_')}_phases.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")


def plot_overall_comparison(
    alg_configs: list[dict],
    sizes: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cfg in alg_configs:
        name = cfg["name"]
        summaries = cfg["summaries_by_size"]
        means, stds = [], []
        for s in summaries:
            timings_list = s["_timings"]
            vals = [t.get("Total Time", 0.0) for t in timings_list]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        color = ALG_PALETTE.get(name.lower().split()[0], "gray")
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                     label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Dataset size (n)", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_title("Total Runtime", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fname = "evaluation2_scalability_overall.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")


def print_scalability_table(
    alg_configs: list[dict],
    sizes: list[int],
) -> None:
    header = f"{'Algorithm':<10s}" + "".join(f"  {'n=' + str(n):>14s}" for n in sizes)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for cfg in alg_configs:
        name = cfg["name"]
        summaries = cfg["summaries_by_size"]
        row = f"{name:<10s}"
        for s in summaries:
            timings_list = s["_timings"]
            vals = [t.get("Total Time", 0.0) for t in timings_list]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            row += f"  {m:>6.1f}±{sd:<5.1f}s"
        print(row)
    print(sep)

    for cfg in alg_configs:
        name = cfg["name"]
        phases = cfg["phases"]
        summaries = cfg["summaries_by_size"]
        print(f"\n{'=' * 60}")
        print(f"  {name} — Per-Phase Average Timing (seconds)")
        print(f"{'=' * 60}")
        phase_header = f"{'Phase':<22s}" + "".join(f"  {'n=' + str(n):>14s}" for n in sizes[:len(summaries)])
        print(phase_header)
        print("-" * len(phase_header))
        for phase in phases + ["Total Time"]:
            row = f"{phase:<22s}"
            for s in summaries:
                timings_list = s["_timings"]
                vals = [t.get(phase, 0.0) for t in timings_list]
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                row += f"  {m:>6.1f}±{sd:<5.1f}s"
            print(row)


if __name__ == "__main__":
    if __name__ == "__main__":
        SIZES_ALL = [1_000, 5_000]
        SIZES_LP = [1_000, 5_000]
        SIZES_BACKURS = [1_000, 5_000]  # extend to [25_000, 50_000, 100_000, ...] as time permits

        FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
        GROUP_ID_FEATURES = ["RACE_BINARY"]
        PROTECTED_COL = "GROUP_ID"
        K = 10
        N_RUNS = 3
        ALPHA = 0.05

        bera_summaries: list[dict] = []
        bercea_summaries: list[dict] = []
        boehm_summaries: list[dict] = []
        backurs_summaries: list[dict] = []

        for n in SIZES_LP:
            print(f"\n{'#' * 60}")
            print(f"  DATASET SIZE n = {n}")
            print(f"{'#' * 60}")

            print(f"\n  RUNNING BERA ET AL. (n={n})")
            print("=" * 60)
            bera_s = run_trials(
                max_rows=n,
                algorithm_fn=bera_fc,
                result_builder=build_bera_result,
                group_id_features=GROUP_ID_FEATURES,
                n_runs=N_RUNS,
                csv_path="../../../us_census_puma_data.csv",
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_centers=K,
                alpha=ALPHA,
                weight_col=None,
            )
            bera_summaries.append(bera_s)

            print(f"\n  RUNNING BERCEA ET AL. (n={n})")
            print("=" * 60)
            bercea_s = run_trials(
                max_rows=n,
                algorithm_fn=bercea_fc,
                result_builder=build_bercea_result,
                group_id_features=GROUP_ID_FEATURES,
                n_runs=N_RUNS,
                csv_path="../../../us_census_puma_data.csv",
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_cluster=K,
                alpha=ALPHA,
                weight_col=None,
            )
            bercea_summaries.append(bercea_s)

            if n in SIZES_ALL:
                print(f"\n  RUNNING BÖHM ET AL. (n={n})")
                print("=" * 60)
                boehm_s = run_trials(
                    max_rows=n,
                    algorithm_fn=boehm_fc,
                    result_builder=build_boehm_result,
                    group_id_features=GROUP_ID_FEATURES,
                    n_runs=N_RUNS,
                    csv_path="../../../us_census_puma_data.csv",
                    feature_cols=FEATURE_COLS,
                    protected_group_col=PROTECTED_COL,
                    k=K,
                    kmedian_trials=3,
                    kmedian_max_iter=30,
                )
                boehm_summaries.append(boehm_s)

        for n in SIZES_BACKURS:
            if any(s.get("_n_size") == n for s in backurs_summaries):
                continue
            print(f"\n  RUNNING BACKURS ET AL. (n={n})")
            print("=" * 60)
            backurs_s = run_trials(
                max_rows=n,
                algorithm_fn=backurs_fc,
                result_builder=build_backurs_result,
                group_id_features=GROUP_ID_FEATURES,
                n_runs=N_RUNS,
                csv_path="../../../us_census_puma_data.csv",
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_cluster=K,
                alpha=ALPHA,
            )
            backurs_summaries.append(backurs_s)

        plot_per_algorithm_phases("Bera", BERA_PHASES,
                                  SIZES_LP, bera_summaries)
        plot_per_algorithm_phases("Bercea", BERCEA_PHASES,
                                  SIZES_LP, bercea_summaries)
        plot_per_algorithm_phases("Böhm", BOEHM_PHASES,
                                  SIZES_ALL, boehm_summaries)
        plot_per_algorithm_phases("Backurs", BACKURS_PHASES,
                                  SIZES_BACKURS, backurs_summaries)

        alg_configs = [
            {"name": "Bera", "phases": BERA_PHASES, "summaries_by_size": bera_summaries},
            {"name": "Bercea", "phases": BERCEA_PHASES, "summaries_by_size": bercea_summaries},
            {"name": "Böhm", "phases": BOEHM_PHASES, "summaries_by_size": boehm_summaries},
            {"name": "Backurs", "phases": BACKURS_PHASES, "summaries_by_size": backurs_summaries},
        ]
        plot_overall_comparison(alg_configs, SIZES_BACKURS)

        print_scalability_table(alg_configs, SIZES_BACKURS)