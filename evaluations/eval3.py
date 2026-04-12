import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc

from runner import run_trials, build_bera_result, build_bercea_result
import csv_loader

FEATURE_CONFIGS = [
    {"name": "Sex", "group_id_features": ["SEX"], "L": 2, "DI": 0.013},
    {"name": "Race (Binary)", "group_id_features": ["RACE_BINARY"], "L": 2, "DI": 0.288},
    {"name": "Age", "group_id_features": ["AGE_BIN"], "L": 4, "DI": 0.042},
    {"name": "Income", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
    {"name": "Race (6-bin)", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    {"name": "Race (9-cat)", "group_id_features": ["RAC1P"], "L": 9, "DI": 0.413},
    {"name": "Inc × Age", "group_id_features": ["INC_BIN", "AGE_BIN"], "L": 16, "DI": None},
]

_ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
}


def plot_pof_vs_di(rows: list[dict], alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    return None

def plot_runtime(rows: list[dict]):
    features = list(dict.fromkeys(r["feature"] for r in rows))
    algorithms = ["Bera", "Bercea"]

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.6), 5))
    bar_width = 0.35
    x = np.arange(len(features))

    for i, alg in enumerate(algorithms):
        means, stds = [], []
        for fname in features:
            row = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
            if row:
                timings = row["all_timings"]
                vals = [t.get("Total Time", 0.0) for t in timings]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        else:
            means.append(0.0)
            stds.append(0.0)
            offset = (i - 0.5) * bar_width
            color = _ALG_PALETTE.get(alg.lower(), "gray")
            bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                          label=alg, color=color, linewidth=0.4, zorder=3)
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.1,
                        f"{mean:.1f}s", ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Total time (s)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("./runtime_by_feature.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    N_SIZE = 20_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 5
    ALPHA = 0.05

    all_rows: list[dict] = []

    for configuration in FEATURE_CONFIGS:
        print(f"  FEATURE: {configuration['name']}  (L={configuration['L']}, DI={configuration['DI']:.3f})")
        print(f"\n  Running Bera [1] ...")
        bera_result, bera_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bera_fc,
            result_builder=build_bera_result,
            group_id_features=configuration["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_centers=K,
            alpha=ALPHA,
            weight_col=None,
        )
        all_rows.append({
            "feature": configuration["name"],
            "L": configuration["L"],
            "DI": configuration["DI"],
            "algorithm": "Bera",
            "pof_mean": bera_summary["All results PoF (mean)"],
            "pof_std": bera_summary["All results PoF (std)"],
            "fair_cost_mean": bera_summary["All results Fair Cost (mean)"],
            "unfair_cost_mean": bera_summary["All results Unfair Cost (mean)"],
            "avg_timing": bera_summary["Avg Timing"],
            "all_timings": bera_summary["_timings"],
        })
        bercea_result, bercea_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bercea_fc,
            result_builder=build_bercea_result,
            group_id_features=configuration["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_cluster=K,
            alpha=ALPHA,
            weight_col=None,
        )
        all_rows.append({
            "feature": configuration["name"],
            "L": configuration["L"],
            "DI": configuration["DI"],
            "algorithm": "Bercea",
            "pof_mean": bercea_summary["All results PoF (mean)"],
            "pof_std": bercea_summary["All results PoF (std)"],
            "fair_cost_mean": bercea_summary["All results Fair Cost (mean)"],
            "unfair_cost_mean": bercea_summary["All results Unfair Cost (mean)"],
            "avg_timing": bercea_summary["Avg Timing"],
            "all_timings": bercea_summary["_timings"],
        })
        plot_pof_vs_di(all_rows, ALPHA)
