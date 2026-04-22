from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from fair_clustering.algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from fair_clustering.algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from fair_clustering.algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc
from fair_clustering.algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from fair_clustering.evaluations.evaluation4 import _avg_total_time

from fair_clustering.runner import (run_trials, build_bera_result, build_bercea_result,
                                    build_backurs_result, build_boehm_result)
#15, 20, 35, 50
K_VALUES = [3, 5, 10]
FEATURE_CFG = {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094}

ALGORITHMS = ["Bera", "Bercea", "Backurs", "Böhm"]
ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452", "Backurs": "#AAA868", "Böhm": "#55A868"}
ALG_MARKERS = {"Bera": "o", "Bercea": "s", "Backurs": "D", "Böhm": "^"}


def plot_costs_with_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    for alg in algorithms:
        subset = sorted([r for r in rows if r["algorithm"] == alg], key=lambda r: r["k"])
        if not subset:
            continue
        ks = [r["k"] for r in subset]
        color, marker = ALG_COLORS[alg], ALG_MARKERS[alg]

        ax.errorbar(ks, [r["fair_cost_mean"] for r in subset],
                    yerr=[r["fair_cost_std"] for r in subset],
                    label=f"{alg} — fair", color=color, marker=marker,
                    linestyle="-", capsize=4, markersize=6, linewidth=1.6)
        ax.errorbar(ks, [r["unfair_cost_mean"] for r in subset],
                    yerr=[r["unfair_cost_std"] for r in subset],
                    label=f"{alg} — unfair", color=color, marker=marker,
                    linestyle=":", capsize=4, markersize=6, linewidth=1.2, alpha=0.6)

    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Total assignment cost", fontsize=11)
    ax.set_title("Fair vs Unfair Cost vs k  (INC_BIN, α=0.05)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_costs_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_costs_vs_k.png")


def plot_pof_with_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    for alg in algorithms:
        subset = sorted([r for r in rows if r["algorithm"] == alg], key=lambda r: r["k"])
        if not subset:
            continue
        color, marker = ALG_COLORS[alg], ALG_MARKERS[alg]
        ax.errorbar([r["k"] for r in subset],
                    [r["pof_mean"] for r in subset],
                    yerr=[r["pof_std"] for r in subset],
                    label=alg, color=color, marker=marker,
                    linestyle="-", capsize=4, markersize=6, linewidth=1.6)

    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="PoF = 1")
    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_title("PoF vs k  (INC_BIN, α=0.05)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_pof_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_pof_vs_k.png")


def plot_runtime_with_k(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    for alg in algorithms:
        subset = sorted([r for r in rows if r["algorithm"] == alg], key=lambda r: r["k"])
        if not subset:
            continue
        color, marker = ALG_COLORS[alg], ALG_MARKERS[alg]
        ax.errorbar([r["k"] for r in subset],
                    [r["time_mean"] for r in subset],
                    yerr=[r["time_std"] for r in subset],
                    label=alg, color=color, marker=marker,
                    linestyle="-", capsize=4, markersize=6, linewidth=1.6)

    ax.set_xlabel("k (number of centres)", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_title("Runtime vs k  (INC_BIN, α=0.05)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_runtime_vs_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation5_runtime_vs_k.png")


def print_k_table(rows: list[dict]) -> None:
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]

    pof_cols = "  ".join(f"{a + ' PoF':>16s}" for a in algorithms)
    fc_cols = "  ".join(f"{a + ' Fair':>14s}" for a in algorithms)
    time_cols = "  ".join(f"{a + ' Time':>14s}" for a in algorithms)
    header = f"{'k':>4s}  {pof_cols}  {fc_cols}  {'Unfair Cost':>12s}  {time_cols}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_pof = ",".join(f"{a}_PoF_mean,{a}_PoF_std" for a in algorithms)
    csv_fc = ",".join(f"{a}_FairCost_mean,{a}_FairCost_std" for a in algorithms)
    csv_uc = ",".join(f"{a}_UnfairCost_mean,{a}_UnfairCost_std" for a in algorithms)
    csv_time = ",".join(f"{a}_Time_mean,{a}_Time_std" for a in algorithms)
    csv_lines = [f"k,{csv_pof},{csv_fc},{csv_uc},{csv_time}"]

    for k in K_VALUES:

        def _get(a):
            return next((r for r in rows if r["algorithm"] == a and r["k"] == k), None)

        def _fp(r):
            return "—" if not r else f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}"

        def _fc(r, key):
            return "—" if not r else f"{r[key]:,.0f}"

        def _ft(r):
            return "—" if not r else f"{r['time_mean']:.1f}±{r['time_std']:.1f}s"

        unfair_r = next((_get(a) for a in algorithms if _get(a)), None)

        print(f"{k:>4d}  "
              + "  ".join(f"{_fp(_get(a)):>16s}" for a in algorithms) + "  "
              + "  ".join(f"{_fc(_get(a), 'fair_cost_mean'):>14s}" for a in algorithms) + "  "
              + f"{_fc(unfair_r, 'unfair_cost_mean'):>12s}  "
              + "  ".join(f"{_ft(_get(a)):>14s}" for a in algorithms))

        def _v(r, key, fmt=".6f"):
            return f"{r[key]:{fmt}}" if r else ""

        parts = [str(k)]
        for a in algorithms:
            r = _get(a)
            parts.append(f"{_v(r, 'pof_mean')},{_v(r, 'pof_std')}")
        for a in algorithms:
            r = _get(a)
            parts.append(f"{_v(r, 'fair_cost_mean', '.2f')},{_v(r, 'fair_cost_std', '.2f')}")
        for a in algorithms:
            r = _get(a)
            parts.append(f"{_v(r, 'unfair_cost_mean', '.2f')},{_v(r, 'unfair_cost_std', '.2f')}")
        for a in algorithms:
            r = _get(a)
            parts.append(f"{_v(r, 'time_mean', '.2f')},{_v(r, 'time_std', '.2f')}")
        csv_lines.append(",".join(parts))

    print(sep)
    csv_path = "evaluation5_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")

if __name__ == "__main__":
    N_SIZE = 1000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    ALPHA = 0.05
    N_RUNS = 3

    all_rows: list[dict] = []

    for k in K_VALUES:
        print(f"\n{'#' * 60}")
        print(f"  k = {k}")
        print(f"{'#' * 60}")

        print(f"\n  Running Bera [2] (k={k}) ...")
        bera_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=bera_fc, result_builder=build_bera_result,
            group_id_features=FEATURE_CFG["group_id_features"], n_runs=N_RUNS,
            csv_path="../../../us_census_puma_data.csv",
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_centers=k, alpha=ALPHA, weight_col=None,
        )
        bera_tm, bera_ts = _avg_total_time(bera_s["_timings"])
        all_rows.append({
            "k": k, "algorithm": "Bera",
            "pof_mean": bera_s["All results PoF (mean)"],
            "pof_std": bera_s["All results PoF (std)"],
            "fair_cost_mean": bera_s["All results Fair Cost (mean)"],
            "fair_cost_std": bera_s["All results Fair Cost (std)"],
            "unfair_cost_mean": bera_s["All results Unfair Cost (mean)"],
            "unfair_cost_std": bera_s["All results Unfair Cost (std)"],
            "time_mean": bera_tm, "time_std": bera_ts,
            "all_timings": bera_s["_timings"],
        })

        # ── Bercea ──
        print(f"\n  Running Bercea [3] (k={k}) ...")
        bercea_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=bercea_fc, result_builder=build_bercea_result,
            group_id_features=FEATURE_CFG["group_id_features"], n_runs=N_RUNS,
            csv_path="../../../us_census_puma_data.csv",
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_cluster=k, alpha=ALPHA, weight_col=None,
        )
        bercea_tm, bercea_ts = _avg_total_time(bercea_s["_timings"])
        all_rows.append({
            "k": k, "algorithm": "Bercea",
            "pof_mean": bercea_s["All results PoF (mean)"],
            "pof_std": bercea_s["All results PoF (std)"],
            "fair_cost_mean": bercea_s["All results Fair Cost (mean)"],
            "fair_cost_std": bercea_s["All results Fair Cost (std)"],
            "unfair_cost_mean": bercea_s["All results Unfair Cost (mean)"],
            "unfair_cost_std": bercea_s["All results Unfair Cost (std)"],
            "time_mean": bercea_tm, "time_std": bercea_ts,
            "all_timings": bercea_s["_timings"],
        })

        print(f"\n  Running Backurs [1] (k={k}) ...")
        backurs_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=backurs_fc, result_builder=build_backurs_result,
            group_id_features=FEATURE_CFG["group_id_features"], n_runs=N_RUNS,
            csv_path="../../../us_census_puma_data.csv",
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k_cluster=k, alpha=ALPHA,
        )
        backurs_tm, backurs_ts = _avg_total_time(backurs_s["_timings"])
        all_rows.append({
            "k": k, "algorithm": "Backurs",
            "pof_mean": backurs_s["All results PoF (mean)"],
            "pof_std": backurs_s["All results PoF (std)"],
            "fair_cost_mean": backurs_s["All results Fair Cost (mean)"],
            "fair_cost_std": backurs_s["All results Fair Cost (std)"],
            "unfair_cost_mean": backurs_s["All results Unfair Cost (mean)"],
            "unfair_cost_std": backurs_s["All results Unfair Cost (std)"],
            "time_mean": backurs_tm, "time_std": backurs_ts,
            "all_timings": backurs_s["_timings"],
        })

        boehm_s = run_trials(
            max_rows=N_SIZE, algorithm_fn=boehm_fc,
            result_builder=build_boehm_result,
            group_id_features=FEATURE_CFG["group_id_features"], n_runs=N_RUNS,
            csv_path="../../../us_census_puma_data.csv",
            feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
            k=k, kmedian_trials=3, kmedian_max_iter=30,
        )
        boehm_tm, boehm_ts = _avg_total_time(boehm_s["_timings"])
        all_rows.append({
            "k": k, "algorithm": "Böhm",
            "pof_mean": boehm_s["All results PoF (mean)"],
            "pof_std": boehm_s["All results PoF (std)"],
            "fair_cost_mean": boehm_s["All results Fair Cost (mean)"],
            "fair_cost_std": boehm_s["All results Fair Cost (std)"],
            "unfair_cost_mean": boehm_s["All results Unfair Cost (mean)"],
            "unfair_cost_std": boehm_s["All results Unfair Cost (std)"],
            "time_mean": boehm_tm, "time_std": boehm_ts,
            "all_timings": boehm_s["_timings"],
        })

    plot_costs_with_k(all_rows)
    plot_pof_with_k(all_rows)
    plot_runtime_with_k(all_rows)
    print_k_table(all_rows)