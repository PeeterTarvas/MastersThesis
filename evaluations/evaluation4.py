import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc

from runner import run_trials, build_bera_result, build_bercea_result, build_backurs_result

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2]

FEATURE_CONFIGS = [
    {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
]
ALGORITHMS = ["Bera", "Bercea", "Backurs"]

ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452", "Backurs": "#AAA868"}
ALG_DASH_OFFSETS = {"Bera": (0, ()), "Bercea": (0, ()), "Backurs": (0, (5, 2))}

def _avg_total_time(timings_list: list[dict]) -> tuple[float, float]:
    vals = [t.get("Total Time", 0.0) for t in timings_list]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return m, s


def plot_pof_vs_alpha(rows: list[dict]) -> None:
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    n_algs = len(algorithms)

    fig, ax = plt.subplots(figsize=(max(8, len(ALPHAS) * 2.2), 5.5))
    x = np.arange(len(ALPHAS))
    width = 0.8 / n_algs

    for feat_cfg in FEATURE_CONFIGS:
        fname = feat_cfg["name"]
        for i, alg in enumerate(algorithms):
            means, stds = [], []
            for alpha in ALPHAS:
                row = next((r for r in rows
                            if r["algorithm"] == alg and r["feature"] == fname
                            and r["alpha"] == alpha), None)
                means.append(float(row["pof_mean"]) if row else 0.0)
                stds.append(float(row["pof_std"]) if row else 0.0)

            offset = (i - n_algs / 2 + 0.5) * width
            color = ALG_COLORS[alg]
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                          label=alg, color=color, edgecolor="black",
                          linewidth=0.4, zorder=3)
            for bar, m, s in zip(bars, means, stds):
                if m > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + s + 0.005,
                            f"{m:.3f}", ha="center", va="bottom",
                            fontsize=7, color="dimgray")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="PoF = 1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=10)
    ax.set_xlabel("α (fairness slack)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_title("Effect of α on PoF", fontsize=13)
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation4_pof_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation4_pof_vs_alpha.png")


def plot_runtime_vs_alpha(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    for alg in algorithms:
        for feat_cfg in FEATURE_CONFIGS:
            fname = feat_cfg["name"]
            subset = [r for r in rows
                      if r["algorithm"] == alg and r["feature"] == fname]
            subset.sort(key=lambda r: r["alpha"])

            alphas = [r["alpha"] for r in subset]
            means = [r["time_mean"] for r in subset]
            stds = [r["time_std"] for r in subset]

            color = ALG_COLORS[alg]
            ax.errorbar(alphas, means, yerr=stds, label=f"{alg} — {fname}",
                        color=color, linestyle="--", marker="s",
                        capsize=4, markersize=6, linewidth=1.6)

    ax.set_xlabel("α (fairness slack)", fontsize=11)
    ax.set_ylabel("Total Runtime (s)", fontsize=11)
    ax.set_title("Runtime vs α  (n=25 k, k=10)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation4_runtime_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation4_runtime_vs_alpha.png")


def print_alpha_table(rows: list[dict]) -> None:
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    pof_cols = "  ".join(f"{a + ' PoF':>16s}" for a in algorithms)
    time_cols = "  ".join(f"{a + ' Time':>14s}" for a in algorithms)
    header = f"{'Feature':<10s} {'α':>5s}  {pof_cols}  {time_cols}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_pof_hdr = ",".join(f"{a}_PoF_mean,{a}_PoF_std" for a in algorithms)
    csv_fc_hdr = ",".join(f"{a}_FairCost_mean" for a in algorithms)
    csv_uc_hdr = ",".join(f"{a}_UnfairCost_mean" for a in algorithms)
    csv_time_hdr = ",".join(f"{a}_Time_mean,{a}_Time_std" for a in algorithms)
    csv_lines = [f"Feature,L,DI,Alpha,{csv_pof_hdr},{csv_fc_hdr},{csv_uc_hdr},{csv_time_hdr}"]

    for feat_cfg in FEATURE_CONFIGS:
        fname = feat_cfg["name"]
        L = feat_cfg["L"]
        DI = feat_cfg["DI"]
        for alpha in ALPHAS:

            def get(alg_name):
                return next((r for r in rows
                             if r["feature"] == fname and r["algorithm"] == alg_name
                             and r["alpha"] == alpha), None)

            def format_pof(r):
                if not r:
                    return "—"
                return f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}"

            def format_time(r):
                if not r:
                    return "—"
                return f"{r['time_mean']:.1f}±{r['time_std']:.1f}s"

            pof_str = "  ".join(f"{format_pof(get(a)):>16s}" for a in algorithms)
            time_str = "  ".join(f"{format_time(get(a)):>14s}" for a in algorithms)

            print(f"{fname:<10s} {alpha:>5.2f}  {pof_str}  {time_str}")

            def get_value(r, key, fmt=".6f"):
                return f"{r[key]:{fmt}}" if r else ""

            csv_pof_vals = ",".join(
                f"{get_value(get(a), 'pof_mean')},{get_value(get(a), 'pof_std')}" for a in algorithms
            )
            csv_fc_vals = ",".join(
                get_value(get(a), 'fair_cost_mean', '.2f') for a in algorithms
            )
            csv_uc_vals = ",".join(
                get_value(get(a), 'unfair_cost_mean', '.2f') for a in algorithms
            )
            csv_time_vals = ",".join(
                f"{get_value(get(a), 'time_mean', '.2f')},{get_value(get(a), 'time_std', '.2f')}" for a in algorithms
            )
            csv_lines.append(
                f"{fname},{L},{DI:.4f},{alpha},{csv_pof_vals},{csv_fc_vals},{csv_uc_vals},{csv_time_vals}"
            )

    print(sep)
    csv_path = "./evaluation4_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    N_SIZE = 10_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 3

    all_rows: list[dict] = []

    for feature_cfg in FEATURE_CONFIGS:
        feature_name = feature_cfg["name"]
        for alpha in ALPHAS:
            print(f"\n  FEATURE: {feature_name} (L={feature_cfg['L']})   α = {alpha}")

            print(f"  Running Bera [2] ...")
            bera_summary = run_trials(
                max_rows=N_SIZE,
                algorithm_fn=bera_fc,
                result_builder=build_bera_result,
                group_id_features=feature_cfg["group_id_features"],
                n_runs=N_RUNS,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_centers=K,
                alpha=alpha,
                weight_col=None,
            )
            bera_tm, bera_ts = _avg_total_time(bera_summary["_timings"])
            all_rows.append({
                "feature": feature_name,
                "L": feature_cfg["L"],
                "DI": feature_cfg["DI"],
                "alpha": alpha,
                "algorithm": "Bera",
                "pof_mean": bera_summary["All results PoF (mean)"],
                "pof_std": bera_summary["All results PoF (std)"],
                "fair_cost_mean": bera_summary["All results Fair Cost (mean)"],
                "unfair_cost_mean": bera_summary["All results Unfair Cost (mean)"],
                "time_mean": bera_tm,
                "time_std": bera_ts,
                "all_timings": bera_summary["_timings"],
            })

            print(f"\n  Running Bercea [3] ...")
            bercea_summary = run_trials(
                max_rows=N_SIZE,
                algorithm_fn=bercea_fc,
                result_builder=build_bercea_result,
                group_id_features=feature_cfg["group_id_features"],
                n_runs=N_RUNS,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_cluster=K,
                alpha=alpha,
                weight_col=None,
            )
            bercea_tm, bercea_ts = _avg_total_time(bercea_summary["_timings"])
            all_rows.append({
                "feature": feature_name,
                "L": feature_cfg["L"],
                "DI": feature_cfg["DI"],
                "alpha": alpha,
                "algorithm": "Bercea",
                "pof_mean": bercea_summary["All results PoF (mean)"],
                "pof_std": bercea_summary["All results PoF (std)"],
                "fair_cost_mean": bercea_summary["All results Fair Cost (mean)"],
                "unfair_cost_mean": bercea_summary["All results Unfair Cost (mean)"],
                "time_mean": bercea_tm,
                "time_std": bercea_ts,
                "all_timings": bercea_summary["_timings"],
            })

            print(f"\n  Running Backurs [1] ...")
            backurs_summary = run_trials(
                max_rows=N_SIZE,
                algorithm_fn=backurs_fc,
                result_builder=build_backurs_result,
                group_id_features=feature_cfg["group_id_features"],
                n_runs=N_RUNS,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_cluster=K,
                alpha=alpha,
            )
            backurs_tm, backurs_ts = _avg_total_time(backurs_summary["_timings"])
            all_rows.append({
                "feature": feature_name,
                "L": feature_cfg["L"],
                "DI": feature_cfg["DI"],
                "alpha": alpha,
                "algorithm": "Backurs",
                "pof_mean": backurs_summary["All results PoF (mean)"],
                "pof_std": backurs_summary["All results PoF (std)"],
                "fair_cost_mean": backurs_summary["All results Fair Cost (mean)"],
                "unfair_cost_mean": backurs_summary["All results Unfair Cost (mean)"],
                "time_mean": backurs_tm,
                "time_std": backurs_ts,
                "all_timings": backurs_summary["_timings"],
            })

    plot_pof_vs_alpha(all_rows)
    plot_runtime_vs_alpha(all_rows)
    print_alpha_table(all_rows)
