import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from evaluations.evaluation4 import _avg_total_time

from runner import run_trials, build_bera_result, build_bercea_result

K_VALUES = [3, 5, 10, 15, 20, 35]
FEATURE_CFG = {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094}
_ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452"}
_ALG_MARKERS = {"Bera": "o", "Bercea": "s"}

def plot_costs_vs_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alg in ["Bera", "Bercea"]:
        subset = [r for r in rows if r["algorithm"] == alg]
        subset.sort(key=lambda r: r["k"])

        ks = [r["k"] for r in subset]
        fair_means = [r["fair_cost_mean"] for r in subset]
        fair_stds = [r["fair_cost_std"] for r in subset]
        unfair_means = [r["unfair_cost_mean"] for r in subset]
        unfair_stds = [r["unfair_cost_std"] for r in subset]

        color = _ALG_COLORS[alg]
        marker = _ALG_MARKERS[alg]

        ax.errorbar(ks, fair_means, yerr=fair_stds,
                    label=f"{alg} — fair cost", color=color,
                    marker=marker, linestyle="-",
                    capsize=4, markersize=6, linewidth=1.6)
        ax.errorbar(ks, unfair_means, yerr=unfair_stds,
                    label=f"{alg} — unfair cost", color=color,
                    marker=marker, linestyle=":",
                    capsize=4, markersize=6, linewidth=1.2, alpha=0.6)

    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Total assignment cost", fontsize=11)
    ax.set_title("Fair vs Unfair Cost vs k  (INC_BIN, n=10 k, α=0.05)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_costs_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_costs_vs_k.png")


def plot_pof_vs_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alg in ["Bera", "Bercea"]:
        subset = [r for r in rows if r["algorithm"] == alg]
        subset.sort(key=lambda r: r["k"])

        ks = [r["k"] for r in subset]
        pof_means = [r["pof_mean"] for r in subset]
        pof_stds = [r["pof_std"] for r in subset]

        color = _ALG_COLORS[alg]
        marker = _ALG_MARKERS[alg]

        ax.errorbar(ks, pof_means, yerr=pof_stds,
                    label=alg, color=color,
                    marker=marker, linestyle="-",
                    capsize=4, markersize=6, linewidth=1.6)

    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label="PoF = 1")
    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_title("PoF vs k  (INC_BIN, n=10 k, α=0.05)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_pof_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_pof_vs_k.png")


def plot_runtime_vs_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alg in ["Bera", "Bercea"]:
        subset = [r for r in rows if r["algorithm"] == alg]
        subset.sort(key=lambda r: r["k"])

        ks = [r["k"] for r in subset]
        means = [r["time_mean"] for r in subset]
        stds = [r["time_std"] for r in subset]

        color = _ALG_COLORS[alg]
        marker = _ALG_MARKERS[alg]

        ax.errorbar(ks, means, yerr=stds, label=alg,
                    color=color, marker=marker, linestyle="-",
                    capsize=4, markersize=6, linewidth=1.6)

    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Total wall-clock time (s)", fontsize=11)
    ax.set_title("Runtime vs k  (INC_BIN, n=10 k, α=0.05)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_runtime_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_runtime_vs_k.png")


def print_k_table(rows: list[dict]) -> None:
    header = (
        f"{'k':>4s}  "
        f"{'Bera PoF':>16s}  {'Bercea PoF':>16s}  "
        f"{'Bera Fair Cost':>12s}  {'Bercea Cost Fair':>12s}  "
        f"{'Unfair Cost':>12s}  "
        f"{'Bera Time':>12s}  {'Bercea Time':>12s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "k,Bera_PoF_mean,Bera_PoF_std,Bercea_PoF_mean,Bercea_PoF_std,"
        "Bera_FairCost_mean,Bera_FairCost_std,Bercea_FairCost_mean,Bercea_FairCost_std,"
        "Bera_UnfairCost_mean,Bera_UnfairCost_std,Bercea_UnfairCost_mean,Bercea_UnfairCost_std,"
        "Bera_Time_mean,Bera_Time_std,Bercea_Time_mean,Bercea_Time_std"
    ]

    for k in K_VALUES:
        bera_r = next((r for r in rows
                       if r["algorithm"] == "Bera" and r["k"] == k), None)
        bercea_r = next((r for r in rows
                         if r["algorithm"] == "Bercea" and r["k"] == k), None)

        def _fmt_pof(r):
            if not r:
                return "—"
            return f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}"

        def _fmt_cost(r, key):
            if not r:
                return "—"
            return f"{r[key]:,.0f}"

        def _fmt_time(r):
            if not r:
                return "—"
            return f"{r['time_mean']:.1f}±{r['time_std']:.1f}s"

        unfair_str = _fmt_cost(bera_r, "unfair_cost_mean")

        print(
            f"{k:>4d}  "
            f"{_fmt_pof(bera_r):>16s}  {_fmt_pof(bercea_r):>16s}  "
            f"{_fmt_cost(bera_r, 'fair_cost_mean'):>12s}  "
            f"{_fmt_cost(bercea_r, 'fair_cost_mean'):>12s}  "
            f"{unfair_str:>12s}  "
            f"{_fmt_time(bera_r):>12s}  {_fmt_time(bercea_r):>12s}"
        )

        def _v(r, key, fmt=".6f"):
            return f"{r[key]:{fmt}}" if r else ""

        csv_lines.append(
            f"{k},"
            f"{_v(bera_r, 'pof_mean')},{_v(bera_r, 'pof_std')},"
            f"{_v(bercea_r, 'pof_mean')},{_v(bercea_r, 'pof_std')},"
            f"{_v(bera_r, 'fair_cost_mean', '.2f')},{_v(bera_r, 'fair_cost_std', '.2f')},"
            f"{_v(bercea_r, 'fair_cost_mean', '.2f')},{_v(bercea_r, 'fair_cost_std', '.2f')},"
            f"{_v(bera_r, 'unfair_cost_mean', '.2f')},{_v(bera_r, 'unfair_cost_std', '.2f')},"
            f"{_v(bercea_r, 'unfair_cost_mean', '.2f')},{_v(bercea_r, 'unfair_cost_std', '.2f')},"
            f"{_v(bera_r, 'time_mean', '.2f')},{_v(bera_r, 'time_std', '.2f')},"
            f"{_v(bercea_r, 'time_mean', '.2f')},{_v(bercea_r, 'time_std', '.2f')}"
        )

    print(sep)
    csv_path = "./evaluation5_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")

if __name__ == "__main__":
    N_SIZE = 15_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    ALPHA = 0.05
    N_RUNS = 10

    all_rows: list[dict] = []

    for k in K_VALUES:
        print(f"\n{'#' * 60}")
        print(f"  k = {k}")
        print(f"{'#' * 60}")

        print(f"\n  Running Bera [1] (k={k}) ...")
        bera_result, bera_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bera_fc,
            result_builder=build_bera_result,
            group_id_features=FEATURE_CFG["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_centers=k,
            alpha=ALPHA,
            weight_col=None,
        )

        bera_tm, bera_ts = _avg_total_time(bera_summary["_timings"])
        all_rows.append({
            "k": k,
            "algorithm": "Bera",
            "pof_mean": bera_summary["All results PoF (mean)"],
            "pof_std": bera_summary["All results PoF (std)"],
            "fair_cost_mean": bera_summary["All results Fair Cost (mean)"],
            "fair_cost_std": bera_summary["All results Fair Cost (std)"],
            "unfair_cost_mean": bera_summary["All results Unfair Cost (mean)"],
            "unfair_cost_std": bera_summary["All results Unfair Cost (std)"],
            "time_mean": bera_tm,
            "time_std": bera_ts,
            "all_timings": bera_summary["_timings"],
        })

        print(f"\n  Running Bercea [2] (k={k}) ...")
        bercea_result, bercea_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bercea_fc,
            result_builder=build_bercea_result,
            group_id_features=FEATURE_CFG["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_cluster=k,
            alpha=ALPHA,
            weight_col=None,
        )
        bercea_tm, bercea_ts = _avg_total_time(bercea_summary["_timings"])

        all_rows.append({
            "k": k,
            "algorithm": "Bercea",
            "pof_mean": bercea_summary["All results PoF (mean)"],
            "pof_std": bercea_summary["All results PoF (std)"],
            "fair_cost_mean": bercea_summary["All results Fair Cost (mean)"],
            "fair_cost_std": bercea_summary["All results Fair Cost (std)"],
            "unfair_cost_mean": bercea_summary["All results Unfair Cost (mean)"],
            "unfair_cost_std": bercea_summary["All results Unfair Cost (std)"],
            "time_mean": bercea_tm,
            "time_std": bercea_ts,
            "all_timings": bercea_summary["_timings"],
        })

    plot_costs_vs_k(all_rows)
    plot_pof_vs_k(all_rows)
    plot_runtime_vs_k(all_rows)
    print_k_table(all_rows)