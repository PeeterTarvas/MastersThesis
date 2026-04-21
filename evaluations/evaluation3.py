import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc

from runner import (run_trials, build_bera_result, build_bercea_result,
                    build_backurs_result, build_boehm_result)

FEATURE_CONFIGS = [
    {"name": "SEX", "group_id_features": ["SEX"], "L": 2, "DI": 0.013},
    {"name": "RACE_BINARY", "group_id_features": ["RACE_BINARY"], "L": 2, "DI": 0.288},
    {"name": "AGE_BIN", "group_id_features": ["AGE_BIN"], "L": 4, "DI": 0.042},
    {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
    #{"name": "RACE_6", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    #{"name": "AGE_BIN × SEX", "group_id_features": ["AGE_BIN", "SEX"], "L": 8, "DI": 0.040},
    #{"name": "RACE_6 × SEX", "group_id_features": ["RACE_6", "SEX"], "L": 12, "DI": 0.311},
]

ALGORITHMS = ["Bera", "Bercea", "Backurs", "Böhm"]

ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "backurs": "#AAA868",
    "böhm": "#55A868",
}

PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

BERA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
               "Iterative Rounding", "Cost Calculation"]
BERCEA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
                 "MCF Rounding", "Cost Calculation"]
BACKURS_PHASES = ["Data Preparation", "Vanilla K-Median", "Base Selection",
                  "HST Construction", "Fairlet Decomposition", "Cluster Fairlets"]
BOEHM_PHASES = ["Balance Dataset", "Vanilla k-Median", "Böhm Fair Clustering"]

ALG_PHASE_MAP = {
    "Bera": BERA_PHASES,
    "Bercea": BERCEA_PHASES,
    "Backurs": BACKURS_PHASES,
    "Böhm": BOEHM_PHASES,
}


def plot_runtime(rows: list[dict]):
    features = list(dict.fromkeys(r["feature"] for r in rows))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    n_algs = len(algorithms)

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.6), 5))
    bar_width = 0.8 / n_algs
    x = np.arange(len(features))

    for i, alg in enumerate(algorithms):
        means, stds = [], []
        for fname in features:
            row = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
            if row:
                vals = [t.get("Total Time", 0.0) for t in row["all_timings"]]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (i - n_algs / 2 + 0.5) * bar_width
        color = ALG_PALETTE.get(alg.lower(), "gray")
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                      label=alg, color=color, linewidth=0.4, zorder=3)
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
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

    n_panels = len(algorithms)
    for ax_idx, alg in enumerate(algorithms):
        fig2, ax2 = plt.subplots(figsize=(7 , 5.5))
        phases = ALG_PHASE_MAP[alg]
        bottoms = np.zeros(len(features))
        for phase_idx, phase in enumerate(phases):
            vals_per_feat = []
            for fname in features:
                row = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
                if row:
                    vals_per_feat.append(float(np.mean([t.get(phase, 0.0) for t in row["all_timings"]])))
                else:
                    vals_per_feat.append(0.0)
            color = PHASE_PALETTE[phase_idx % len(PHASE_PALETTE)]
            ax2.bar(x, vals_per_feat, bar_width * 1.6, bottom=bottoms,
                    label=phase, color=color, linewidth=0.3, zorder=3)
            bottoms += np.array(vals_per_feat)
        ax2.set_xticks(x)
        ax2.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
        ax2.set_title(f"{alg} — Phase Timings", fontsize=12)
        ax2.set_ylabel("Time (s)" if ax_idx == 0 else "", fontsize=11)
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        fig2.tight_layout()
        fig2.savefig(f"evaluation3_{alg}_runtime_by_feature_phases.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved runtime_by_feature_phases.png")


def print_feature_table(rows: list[dict]) -> None:
    features = list(dict.fromkeys(r["feature"] for r in rows))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    pof_cols = "  ".join(f"{a + ' PoF':>16s}" for a in algorithms)
    time_cols = "  ".join(f"{a + ' Time':>12s}" for a in algorithms)
    header = f"{'Feature':<18s} {'L':>3s} {'DI':>6s}  {pof_cols}  {time_cols}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_pof = ",".join(f"{a}_PoF_mean,{a}_PoF_std" for a in algorithms)
    csv_fc = ",".join(f"{a}_FairCost_mean" for a in algorithms)
    csv_uc = ",".join(f"{a}_UnfairCost_mean" for a in algorithms)
    csv_time = ",".join(f"{a}_Time_mean,{a}_Time_std" for a in algorithms)
    csv_lines = [f"Feature,L,DI,{csv_pof},{csv_fc},{csv_uc},{csv_time}"]

    for fname in features:
        f_rows = [r for r in rows if r["feature"] == fname]
        L, DI = f_rows[0]["L"], f_rows[0]["DI"]

        def _get(a):
            return next((r for r in f_rows if r["algorithm"] == a), None)

        def _fmt_pof(r):
            return f"{'—':>16s}" if not r else f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}"

        def _tv(r):
            if not r: return 0.0, 0.0
            v = [t.get("Total Time", 0.0) for t in r["all_timings"]]
            return float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)

        def _fmt_time(r):
            if not r: return f"{'—':>12s}"
            m, s = _tv(r)
            return f"{m:.1f}±{s:.1f}s"

        print(f"{fname:<18s} {L:>3d} {DI:>6.3f}  "
              + "  ".join(f"{_fmt_pof(_get(a)):>16s}" for a in algorithms) + "  "
              + "  ".join(f"{_fmt_time(_get(a)):>12s}" for a in algorithms))

        parts = [f"{fname},{L},{DI:.4f}"]
        for a in algorithms:
            r = _get(a)
            if r:
                parts.append(f"{r['pof_mean']:.6f},{r['pof_std']:.6f}")
            else:
                parts.append(",")
        for a in algorithms:
            r = _get(a)
            parts.append(f"{r['fair_cost_mean']:.2f}" if r else "")
        for a in algorithms:
            r = _get(a)
            parts.append(f"{r['unfair_cost_mean']:.2f}" if r else "")
        for a in algorithms:
            r = _get(a)
            if r:
                m, s = _tv(r)
                parts.append(f"{m:.2f},{s:.2f}")
            else:
                parts.append(",")
        csv_lines.append(",".join(parts))

    print(sep)
    csv_path = "./evaluation3_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


def plot_pof(rows: list[dict]):
    features = list(dict.fromkeys(r["feature"] for r in rows))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    n_algs = len(algorithms)

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.6), 5))
    bar_width = 0.8 / n_algs
    x = np.arange(len(features))

    for i, alg in enumerate(algorithms):
        means, stds = [], []
        for fname in features:
            row = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
            if row:
                means.append(float(row["pof_mean"]))
                std_val = row["pof_std"]
                stds.append(float(std_val) if not np.isnan(std_val) else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (i - n_algs / 2 + 0.5) * bar_width
        color = ALG_PALETTE.get(alg.lower(), "gray")
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                      label=alg, color=color, linewidth=0.4, zorder=3)
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.axhline(1.0, color='#C44E52', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label="PoF = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_ylim(bottom=0.0)
    all_tops = [r["pof_mean"] + r["pof_std"] for r in rows]
    ax.set_ylim(top=max(all_tops + [1.0]) * 1.15)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation3_pof_by_feature.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved pof_by_feature.png")

if __name__ == "__main__":
    N_SIZE = 100
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 3
    ALPHA = 0.05

    all_rows: list[dict] = []

    for cfg in FEATURE_CONFIGS:
        feat_name = cfg["name"]
        print(f"\n  FEATURE: {feat_name}  (L={cfg['L']}, DI={cfg['DI']:.3f})")

        print(f"\n  Running Bera [2] ...")
        bera_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=bera_fc, result_builder=build_bera_result,
            group_id_features=cfg["group_id_features"], n_runs=N_RUNS,
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_centers=K, alpha=ALPHA, weight_col=None,
        )
        all_rows.append({"feature": feat_name, "L": cfg["L"], "DI": cfg["DI"],
                         "algorithm": "Bera",
                         "pof_mean": bera_s["All results PoF (mean)"],
                         "pof_std": bera_s["All results PoF (std)"],
                         "fair_cost_mean": bera_s["All results Fair Cost (mean)"],
                         "unfair_cost_mean": bera_s["All results Unfair Cost (mean)"],
                         "all_timings": bera_s["_timings"]})

        print(f"\n  Running Bercea [3] ...")
        bercea_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=bercea_fc, result_builder=build_bercea_result,
            group_id_features=cfg["group_id_features"], n_runs=N_RUNS,
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_cluster=K, alpha=ALPHA, weight_col=None,
        )
        all_rows.append({"feature": feat_name, "L": cfg["L"], "DI": cfg["DI"],
                         "algorithm": "Bercea",
                         "pof_mean": bercea_s["All results PoF (mean)"],
                         "pof_std": bercea_s["All results PoF (std)"],
                         "fair_cost_mean": bercea_s["All results Fair Cost (mean)"],
                         "unfair_cost_mean": bercea_s["All results Unfair Cost (mean)"],
                         "all_timings": bercea_s["_timings"]})

        print(f"\n  Running Backurs [1] ...")
        backurs_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=backurs_fc, result_builder=build_backurs_result,
            group_id_features=cfg["group_id_features"], n_runs=N_RUNS,
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_cluster=K, alpha=ALPHA,
        )
        all_rows.append({"feature": feat_name, "L": cfg["L"], "DI": cfg["DI"],
                         "algorithm": "Backurs",
                         "pof_mean": backurs_s["All results PoF (mean)"],
                         "pof_std": backurs_s["All results PoF (std)"],
                         "fair_cost_mean": backurs_s["All results Fair Cost (mean)"],
                         "unfair_cost_mean": backurs_s["All results Unfair Cost (mean)"],
                         "all_timings": backurs_s["_timings"]})


        boehm_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=boehm_fc,
            result_builder=build_boehm_result,
            group_id_features=cfg["group_id_features"], n_runs=N_RUNS,
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k=K, kmedian_trials=3, kmedian_max_iter=30,
        )
        all_rows.append({"feature": feat_name, "L": cfg["L"], "DI": cfg["DI"],
                         "algorithm": "Böhm",
                         "pof_mean": boehm_s["All results PoF (mean)"],
                         "pof_std": boehm_s["All results PoF (std)"],
                         "fair_cost_mean": boehm_s["All results Fair Cost (mean)"],
                         "unfair_cost_mean": boehm_s["All results Unfair Cost (mean)"],
                         "all_timings": boehm_s["_timings"]})


    plot_runtime(all_rows)
    print_feature_table(all_rows)
    plot_pof(all_rows)
