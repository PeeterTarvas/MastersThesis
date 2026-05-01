import os
import json
import argparse
import matplotlib

from fair_clustering.results_encoder import load_summary, save_summary

matplotlib.use("Agg")

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from fair_clustering.algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from fair_clustering.algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from fair_clustering.algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc
from fair_clustering.algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from fair_clustering.evaluations.evaluation4 import _avg_total_time

from fair_clustering.runner import (run_trials, build_bera_result, build_bercea_result,
                                    build_backurs_result, build_boehm_result)

K_VALUES = [3, 5, 10, 20, 35, 50]
FEATURE_CFG = {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094}

ALGORITHMS = ["Bera", "Bercea", "Backurs", "Böhm"]
ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452", "Backurs": "#AAA868", "Böhm": "#55A868"}
ALG_MARKERS = {"Bera": "o", "Bercea": "s", "Backurs": "D", "Böhm": "^"}


def _avg_total_time(timings_list: list[dict]) -> tuple[float, float]:
    vals = [t.get("Total Time", 0.0) for t in timings_list]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return m, s


def _summary_to_row(summary, k, algorithm) -> dict:
    tm, ts = _avg_total_time(summary["_timings"])
    return {
        "k": k,
        "algorithm": algorithm,
        "pof_mean": summary["All results PoF (mean)"],
        "pof_std": summary["All results PoF (std)"],
        "fair_cost_mean": summary["All results Fair Cost (mean)"],
        "fair_cost_std": summary["All results Fair Cost (std)"],
        "unfair_cost_mean": summary["All results Unfair Cost (mean)"],
        "unfair_cost_std": summary["All results Unfair Cost (std)"],
        "time_mean": tm,
        "time_std": ts,
    }


def plot_costs_with_k(rows: list[dict], k_values: list[int]) -> None:
    """
    Grouped bar chart per k.

    Per k group: 1 shared unfair bar (light gray, used by Bera/Bercea/Backurs)
    + 4 fair bars (one per algorithm). Böhm's distinct unfair baseline (different
    because it runs on upsampled data) is shown as a black dashed tick on top of
    the Böhm bar so the reader can compare it directly to the shared baseline.
    """
    algorithms_present = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    n_bars = 1 + len(algorithms_present)  # 1 shared unfair + N alg fair bars
    width = 0.8 / n_bars
    x = np.arange(len(k_values))

    fig, ax = plt.subplots(figsize=(max(10, len(k_values) * 2.0), 6))

    # Shared k-median unfair baseline (Bera/Bercea/Backurs all share this)
    ref_alg_for_shared_unfair = next(
        (a for a in ("Bera", "Bercea", "Backurs") if a in algorithms_present), None
    )
    if ref_alg_for_shared_unfair:
        unfair_means, unfair_stds = [], []
        for k in k_values:
            r = next((r for r in rows
                      if r["algorithm"] == ref_alg_for_shared_unfair and r["k"] == k), None)
            unfair_means.append(r["unfair_cost_mean"] if r else 0.0)
            unfair_stds.append(r["unfair_cost_std"] if r else 0.0)

        offset0 = (0 - n_bars / 2 + 0.5) * width
        ax.bar(x + offset0, unfair_means, width, yerr=unfair_stds, capsize=3,
               color="lightgray", edgecolor="black", linewidth=0.5,
               label="Unfair k-median (shared by Bera / Bercea / Backurs)", zorder=3)

        for xi, k in enumerate(k_values):
            if unfair_means[xi] > 0:
                ax.text(xi + offset0, unfair_means[xi] + unfair_stds[xi],
                        f"{unfair_means[xi]:,.0f}", ha="center", va="bottom",
                        fontsize=7, color="dimgray")

    # Fair cost per algorithm
    for i, alg in enumerate(algorithms_present):
        means, stds = [], []
        for k in k_values:
            r = next((r for r in rows if r["algorithm"] == alg and r["k"] == k), None)
            means.append(r["fair_cost_mean"] if r else 0.0)
            stds.append(r["fair_cost_std"] if r else 0.0)

        offset = ((i + 1) - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      color=ALG_COLORS[alg], edgecolor="black", linewidth=0.5,
                      label=f"{alg} fair", zorder=3)

        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s,
                        f"{m:,.0f}", ha="center", va="bottom",
                        fontsize=7, color="dimgray")

        # Böhm-specific unfair baseline (upsampled): tick on top of Böhm bar
        if alg == "Böhm":
            for xi, k in enumerate(k_values):
                r = next((r for r in rows if r["algorithm"] == "Böhm" and r["k"] == k), None)
                if r:
                    bx = xi + offset
                    ax.hlines(r["unfair_cost_mean"], bx - width / 2, bx + width / 2,
                              colors="black", linestyles=(0, (3, 2)), linewidth=1.4,
                              zorder=4,
                              label="Böhm unfair (upsampled-data baseline)" if xi == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=10)
    ax.set_xlabel("Number of centres (k)", fontsize=11)
    ax.set_ylabel("Total assignment cost (mean across runs)", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation5_costs_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation5_costs_vs_k.png")


def plot_pof_with_k(rows: list[dict], k_values: list[int]) -> None:
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
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(k_values)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_pof_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation5_pof_vs_k.png")


def plot_runtime_with_k(rows: list[dict], k_values: list[int]) -> None:
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
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xticks(k_values)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation5_runtime_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation5_runtime_vs_k.png")


def print_k_table(rows: list[dict], k_values: list[int]) -> None:
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

    for k in k_values:

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
    parser = argparse.ArgumentParser(description="Run Evaluation 5 (effect of k)")
    parser.add_argument("--csv_path", type=str,
                        default="../../../us_census_puma_data.csv",
                        help="Path to ACS PUMS CSV")
    parser.add_argument("--n_size", type=int, default=30_000,
                        help="Sample size n (thesis spec: 30 000 to satisfy mp_min≥30 at k=50)")
    parser.add_argument("--n_runs", type=int, default=30,
                        help="Independent runs N for Bera/Bercea/Backurs (thesis spec: 30)")
    parser.add_argument("--n_runs_boehm", type=int, default=10,
                        help="Independent runs N for Böhm (smaller because cubic matching cost)")
    parser.add_argument("--ckpt_dir", type=str, default="evaluation5_partial",
                        help="Per-cell checkpoint directory")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test: tiny n, 2 runs, k=[3] only")
    args = parser.parse_args()

    all_rows: list[dict] = []

    if args.quick:
        args.n_size = 500
        args.n_runs = 2
        args.n_runs_boehm = 2
        k_values_to_run = [3]
        print("[QUICK MODE] n=500, N=2, k=[3]")
    else:
        k_values_to_run = K_VALUES

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    ALPHA = 0.05

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nConfig: n={args.n_size}, ks={k_values_to_run}, α={ALPHA}")
    print(f"        N={args.n_runs} (Böhm uses N={args.n_runs_boehm})")
    print(f"        Feature={FEATURE_CFG['name']} (L={FEATURE_CFG['L']})")
    print(f"        CSV={args.csv_path}")
    print(f"        Checkpoint dir={ckpt_dir}\n")

    all_rows: list[dict] = []

    for k in k_values_to_run:
        print(f"\n{'#' * 60}\n  k = {k}\n{'#' * 60}")

        cell_path = ckpt_dir / f"bera_k{k}.json"
        if cell_path.exists():
            print(f"  [skip] Bera k={k} — checkpoint exists")
            summary = load_summary(str(cell_path))
        else:
            print(f"\n  Running Bera (k={k})...")
            summary = run_trials(
                max_rows=args.n_size, algorithm_fn=bera_fc, result_builder=build_bera_result,
                group_id_features=FEATURE_CFG["group_id_features"], n_runs=args.n_runs,
                csv_path=args.csv_path,
                feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
                k_centers=k, alpha=ALPHA, weight_col=None,
            )
            save_summary(summary, str(cell_path))
        all_rows.append(_summary_to_row(summary, k, "Bera"))

        cell_path = ckpt_dir / f"bercea_k{k}.json"
        if cell_path.exists():
            print(f"  [skip] Bercea k={k} — checkpoint exists")
            summary = load_summary(str(cell_path))
        else:
            print(f"\n  Running Bercea (k={k})...")
            summary = run_trials(
                max_rows=args.n_size, algorithm_fn=bercea_fc, result_builder=build_bercea_result,
                group_id_features=FEATURE_CFG["group_id_features"], n_runs=args.n_runs,
                csv_path=args.csv_path,
                feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
                k_cluster=k, alpha=ALPHA, weight_col=None,
            )
            save_summary(summary, str(cell_path))
        all_rows.append(_summary_to_row(summary, k, "Bercea"))

        cell_path = ckpt_dir / f"backurs_k{k}.json"
        if cell_path.exists():
            print(f"  [skip] Backurs k={k} — checkpoint exists")
            summary = load_summary(str(cell_path))
        else:
            print(f"\n  Running Backurs (k={k})...")
            summary = run_trials(
                max_rows=args.n_size, algorithm_fn=backurs_fc, result_builder=build_backurs_result,
                group_id_features=FEATURE_CFG["group_id_features"], n_runs=args.n_runs,
                csv_path=args.csv_path,
                feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
                k_cluster=k, alpha=ALPHA,
            )
            save_summary(summary, str(cell_path))
        all_rows.append(_summary_to_row(summary, k, "Backurs"))

        cell_path = ckpt_dir / f"boehm_k{k}.json"
        if cell_path.exists():
            print(f"  [skip] Böhm k={k} — checkpoint exists")
            summary = load_summary(str(cell_path))
        else:
            print(f"\n  Running Böhm (k={k}, N={args.n_runs_boehm})...")
            summary = run_trials(
                max_rows=args.n_size, algorithm_fn=boehm_fc, result_builder=build_boehm_result,
                group_id_features=FEATURE_CFG["group_id_features"], n_runs=args.n_runs_boehm,
                csv_path=args.csv_path,
                feature_cols=FEATURE_COLS, protected_group_col=PROTECTED_COL,
                k=k, kmedian_trials=3, kmedian_max_iter=30,
            )
            save_summary(summary, str(cell_path))
        all_rows.append(_summary_to_row(summary, k, "Böhm"))

    print(f"\n{'=' * 60}\n  Generating Eval 5 plots & tables\n{'=' * 60}")
    plot_costs_with_k(all_rows, k_values_to_run)
    plot_pof_with_k(all_rows, k_values_to_run)
    plot_runtime_with_k(all_rows, k_values_to_run)
    print_k_table(all_rows, k_values_to_run)
    print("\nEvaluation 5 complete.")
