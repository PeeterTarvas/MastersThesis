import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc

from runner import run_trials, build_bera_result, build_bercea_result

FEATURE_CONFIGS = [
    {"name": "SEX",            "group_id_features": ["SEX"],                     "L": 2,  "DI": 0.013},
    {"name": "RACE_BINARY",    "group_id_features": ["RACE_BINARY"],             "L": 2,  "DI": 0.288},
    {"name": "AGE_BIN",        "group_id_features": ["AGE_BIN"],                 "L": 4,  "DI": 0.042},
    {"name": "INC_BIN",        "group_id_features": ["INC_BIN"],                 "L": 4,  "DI": 0.094},
    {"name": "RACE_6",         "group_id_features": ["RACE_6"],                  "L": 6,  "DI": 0.343},
    {"name": "AGE_BIN × SEX",  "group_id_features": ["AGE_BIN", "SEX"],          "L": 8,  "DI": 0.040},
    {"name": "RACE_6 × SEX",   "group_id_features": ["RACE_6",  "SEX"],          "L": 12, "DI": 0.311},
]

_ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
}

_PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


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
                      label=f"{alg}" , color=color, linewidth=0.4, zorder=3)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.1,
                    f"{mean:.1f}s", ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Total time (s)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation3_runtime_by_feature.png", dpi=150, bbox_inches="tight")
    plt.show()

    BERA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
                   "Iterative Rounding", "Cost Calculation"]
    BERCEA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
                     "MCF Rounding", "Cost Calculation"]
    alg_phases = {"Bera": BERA_PHASES, "Bercea": BERCEA_PHASES}


    fig2, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for ax_idx, alg in enumerate(algorithms):
        ax2 = axes[ax_idx]
        phases = alg_phases[alg]
        bottoms = np.zeros(len(features))
        for phase_idx, phase in enumerate(phases):
            values_per_feature = []
            for feature_name in features:
                row = next((row for row in rows if row["feature"] == feature_name and row["algorithm"] == alg), None)
                if row:
                    timings = row["all_timings"]
                    value = float(np.mean([t.get(phase, 0.0) for t in timings]))
                else:
                    value = 0
                values_per_feature.append(value)
            color = _PHASE_PALETTE[phase_idx % len(_PHASE_PALETTE)]
            ax2.bar(x, values_per_feature, bar_width * 1.6, bottom=bottoms,
                    label=phase, color=color, linewidth=0.3, zorder=3)
            bottoms += np.array(values_per_feature)

        ax2.set_xticks(x)
        ax2.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
        ax2.set_title(f"{alg} — Phase Timings", fontsize=12)
        ax2.set_ylabel("Time (s)" if ax_idx == 0 else "", fontsize=11)
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    fig2.tight_layout()
    fig2.savefig("evaluation3_runtime_by_feature_phases.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved runtime_by_feature_phases.png")


def print_feature_table(rows: list[dict]) -> None:
    features = list(dict.fromkeys(r["feature"] for r in rows))

    header = (
        f"{'Feature':<18s} {'L':>3s} {'DI':>6s}  "
        f"{'Bera PoF':>16s}  {'Bercea PoF':>16s}  "
        f"{'Bera Time':>12s}  {'Bercea Time':>12s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "Feature,L,DI,"
        "Bera_PoF_mean,Bera_PoF_std,Bercea_PoF_mean,Bercea_PoF_std,"
        "Bera_FairCost_mean,Bercea_FairCost_mean,"
        "Bera_UnfairCost_mean,Bercea_UnfairCost_mean,"
        "Bera_Violations,Bercea_Violations,"
        "Bera_Time_mean,Bera_Time_std,Bercea_Time_mean,Bercea_Time_std"
    ]

    for fname in features:
        f_rows = [r for r in rows if r["feature"] == fname]
        bera_r = next((r for r in f_rows if r["algorithm"] == "Bera"), None)
        bercea_r = next((r for r in f_rows if r["algorithm"] == "Bercea"), None)

        L = f_rows[0]["L"]
        DI = f_rows[0]["DI"]
        di_str = f"{DI:.3f}"

        def _fmt_pof(r):
            return f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}"

        def _fmt_time(r):
            tk = "Total Time"
            vals = [t.get(tk, 0.0) for t in r["all_timings"]]
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1))
            return f"{m:.1f}±{s:.1f}s"

        def _time_vals(r):
            tk = "Total Time"
            vals = [t.get(tk, 0.0) for t in r["all_timings"]]
            return float(np.mean(vals)), float(np.std(vals, ddof=1))

        print(
            f"{fname:<18s} {L:>3d} {di_str:>6s}  "
            f"{_fmt_pof(bera_r):>16s}  {_fmt_pof(bercea_r):>16s}  "
            f"{_fmt_time(bera_r):>12s}  {_fmt_time(bercea_r):>12s}"
        )

        b_pm = f"{bera_r['pof_mean']:.6f}"
        b_ps = f"{bera_r['pof_std']:.6f}"
        c_pm = f"{bercea_r['pof_mean']:.6f}"
        c_ps = f"{bercea_r['pof_std']:.6f}"
        b_fc = f"{bera_r['fair_cost_mean']:.2f}"
        c_fc = f"{bercea_r['fair_cost_mean']:.2f}"
        b_uc = f"{bera_r['unfair_cost_mean']:.2f}"
        c_uc = f"{bercea_r['unfair_cost_mean']:.2f}"
        di_csv = f"{DI:.4f}"
        b_tm, b_ts = _time_vals(bera_r)
        c_tm, c_ts = _time_vals(bercea_r)

        csv_lines.append(
            f"{fname},{L},{di_csv},"
            f"{b_pm},{b_ps},{c_pm},{c_ps},"
            f"{b_fc},{c_fc},{b_uc},{c_uc},"
            f"{b_tm:.2f},{b_ts:.2f},{c_tm:.2f},{c_ts:.2f}"
        )

    print(sep)

    csv_path = "./evaluation3_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


def plot_pof(rows: list[dict]):
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
                means.append(float(row.get("pof_mean", 0.0)))
                std_val = row.get("pof_std", 0.0)
                stds.append(float(std_val) if not np.isnan(std_val) else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - 0.5) * bar_width
        color = _ALG_PALETTE.get(alg.lower(), "gray")
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                      label=alg, color=color, linewidth=0.4, zorder=3)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.axhline(y=1.0, color='#C44E52', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label="Ideal PoF (1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)

    ax.set_ylim(bottom=0.0)

    max_val = max([m + s for m, s in zip(means, stds)] + [1.0])
    ax.set_ylim(top=max_val + (max_val * 0.15))

    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig("evaluation3_pof_by_feature.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved pof_by_feature.png")

if __name__ == "__main__":
    N_SIZE = 20_00
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 3
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
    plot_runtime(all_rows)
    print_feature_table(all_rows)
    plot_pof(all_rows)
