import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc

from runner import run_trials, build_bera_result, build_bercea_result, build_boehm_result

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

_PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

_ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
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
        color = _PHASE_PALETTE[idx % len(_PHASE_PALETTE)]
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
        color = _ALG_PALETTE.get(name.lower().split()[0], "gray")
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
    header = f"{'Algorithm'}" + "".join(f"{'n='+str(n)}" for n in sizes)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for cfg in alg_configs:
        name = cfg["name"]
        total_key = cfg["total_key"]
        summaries = cfg["summaries_by_size"]
        row = f"{name}"
        for s in summaries:
            timings_list = s["_timings"]
            vals = [t.get(total_key, 0.0) for t in timings_list]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            row += f"{m}±{sd}"
        print(row)
    print(sep)

    for cfg in alg_configs:
        name = cfg["name"]
        phases = cfg["phases"]
        total_key = cfg["total_key"]
        summaries = cfg["summaries_by_size"]
        print(f"\n{'='*60}")
        print(f"  {name} — Per-Phase Average Timing (seconds)")
        print(f"{'='*60}")
        phase_header = f"{'Phase': }" + "".join(f"{'n='+str(n)}" for n in sizes)
        print(phase_header)
        print("-" * len(phase_header))
        for phase in phases + [total_key]:
            row = f"{phase}"
            for s in summaries:
                timings_list = s["_timings"]
                vals = [t.get(phase, 0.0) for t in timings_list]
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                row += f"{m}±{sd}"
            print(row)
        print()


if __name__ == "__main__":
    SIZES = [1_000, 5_000, 10_000]

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    GROUP_ID_FEATURES = ["RACE_BINARY"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 3
    ALPHA = 0.05

    bera_summaries: list[dict] = []
    bercea_summaries: list[dict] = []
    boehm_summaries: list[dict] = []

    for n in SIZES:
        print(f"\n{'#'*60}")
        print(f"  DATASET SIZE n = {n}")
        print(f"{'#'*60}")

        print(f"\n  RUNNING BERA ET AL. (n={n})")
        print("=" * 60)
        _, bera_s = run_trials(
            max_rows=n,
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
        bera_summaries.append(bera_s)

        print(f"\n  RUNNING BERCEA ET AL. (n={n})")
        print("=" * 60)
        _, bercea_s = run_trials(
            max_rows=n,
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
        bercea_summaries.append(bercea_s)

        print(f"\n  RUNNING BÖHM ET AL. (n={n})")
        print("=" * 60)
        _, boehm_s = run_trials(
            max_rows=n,
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
        boehm_summaries.append(boehm_s)

    plot_per_algorithm_phases("Bera", BERA_PHASES,
                             SIZES, bera_summaries)
    plot_per_algorithm_phases("Bercea", BERCEA_PHASES,
                             SIZES, bercea_summaries)
    plot_per_algorithm_phases("Böhm", BOEHM_PHASES,
                             SIZES, boehm_summaries)

    alg_configs = [
        {"name": "Bera",   "phases": BERA_PHASES, "summaries_by_size": bera_summaries},
        {"name": "Bercea", "phases": BERCEA_PHASES, "summaries_by_size": bercea_summaries},
        {"name": "Böhm",   "phases": BOEHM_PHASES,  "summaries_by_size": boehm_summaries},
    ]
    plot_overall_comparison(alg_configs, SIZES)

    print_scalability_table(alg_configs, SIZES)