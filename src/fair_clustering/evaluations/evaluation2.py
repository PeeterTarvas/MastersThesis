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
from fair_clustering.algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from fair_clustering.algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc

from fair_clustering.runner import (
    run_trials,
    build_bera_result,
    build_bercea_result,
    build_boehm_result,
    build_backurs_result,
)

ALG_FUNCTIONS = {
    "Bera": (bera_fc, build_bera_result, {"k_centers_kw": "k_centers"}),
    "Bercea": (bercea_fc, build_bercea_result, {"k_centers_kw": "k_cluster"}),
    "Backurs": (backurs_fc, build_backurs_result, {"k_centers_kw": "k_cluster"}),
    "Böhm": (boehm_fc, build_boehm_result, {"k_centers_kw": "k"}),
}

PHASE_KEYS = {
    "Bera": [
        "Data Preparation",
        "Vanilla K-Median",
        "Solve Initial LP",
        "Iterative Rounding",
        "Cost Calculation",
    ],
    "Bercea": [
        "Data Preparation",
        "Vanilla K-Median",
        "Solve Initial LP",
        "MCF Rounding",
        "Cost Calculation",
    ],
    "Böhm": [
        "Balance Dataset",
        "Vanilla k-Median",
        "Böhm Fair Clustering",
    ],
    "Backurs": [
        "Data Preparation",
        "Vanilla K-Median",
        "Base Selection",
        "HST Construction",
        "Fairlet Decomposition",
        "Cluster Fairlets",
    ],
}

SIZES_BY_ALG = {
    "Bera": [10_000, 25_000, 50_000, 100_000, 200_000],
    "Bercea": [10_000, 25_000, 50_000, 100_000, 200_000],
    "Backurs": [10_000, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
    "Böhm": [10_000, 25_000, 50_000],
}

ALGORITHMS = ["Bera", "Bercea", "Backurs", "Böhm"]

PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

ALG_COLORS = {
    "Bera": "#4C72B0",
    "Bercea": "#DD8452",
    "Backurs": "#AAA868",
    "Böhm": "#55A868",
}


def _avg_total_time(timings_list: list[dict]) -> tuple[float, float]:
    vals = [t.get("Total Time", 0.0) for t in timings_list]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return m, s


def _summary_to_row(summary: dict, n: int, algorithm: str) -> dict:
    """Flatten a run_trials summary into the row layout used by plotting."""
    tm, ts = _avg_total_time(summary["_timings"])
    return {
        "n": n,
        "algorithm": algorithm,
        "pof_mean": summary["All results PoF (mean)"],
        "pof_std": summary["All results PoF (std)"],
        "fair_cost_mean": summary["All results Fair Cost (mean)"],
        "fair_cost_std": summary["All results Fair Cost (std)"],
        "unfair_cost_mean": summary["All results Unfair Cost (mean)"],
        "unfair_cost_std": summary["All results Unfair Cost (std)"],
        "time_mean": tm,
        "time_std": ts,
        "_timings": summary["_timings"],
    }


def plot_per_algorithm_phases(
        alg_name: str,
        phase_keys: list[str],
        rows: list[dict],
) -> None:
    """One figure per algorithm: each phase as a line over dataset sizes."""
    subset = sorted(
        [r for r in rows if r["algorithm"] == alg_name],
        key=lambda r: r["n"],
    )
    if not subset:
        print(f"  [skip] No data for {alg_name}")
        return

    sizes = [r["n"] for r in subset]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, phase in enumerate(phase_keys):
        means, stds = [], []
        for r in subset:
            vals = [t.get(phase, 0.0) for t in r["_timings"]]
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

    total_means = [r["time_mean"] for r in subset]
    total_stds = [r["time_std"] for r in subset]
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
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fname = f"evaluation2_scalability_{alg_name.lower().replace(' ', '_').replace('ö', 'o')}_phases.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_overall_comparison(rows: list[dict]) -> None:
    """
    One figure with all algorithms.  Each algorithm is plotted only at
    the sizes for which data exists, so lines may differ in length.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alg in ALGORITHMS:
        subset = sorted(
            [r for r in rows if r["algorithm"] == alg],
            key=lambda r: r["n"],
        )
        if not subset:
            continue
        sizes = [r["n"] for r in subset]
        means = [r["time_mean"] for r in subset]
        stds = [r["time_std"] for r in subset]
        color = ALG_COLORS[alg]
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                    label=alg, color=color, linewidth=1.8)

    ax.set_xlabel("Dataset size (n)", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fname = "evaluation2_scalability_overall.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def print_scalability_table(rows: list[dict]) -> None:
    """Print per-algorithm tables and export a CSV."""

    csv_lines = ["algorithm,n,time_mean,time_std,pof_mean,pof_std,"
                 "fair_cost_mean,fair_cost_std,unfair_cost_mean,unfair_cost_std"]

    for alg in ALGORITHMS:
        subset = sorted(
            [r for r in rows if r["algorithm"] == alg],
            key=lambda r: r["n"],
        )
        if not subset:
            continue

        sizes = [r["n"] for r in subset]
        header = f"{'n':>10s}  {'Time':>14s}  {'PoF':>14s}  {'Fair Cost':>14s}  {'Unfair Cost':>14s}"
        sep = "-" * len(header)
        print(f"\n{'=' * 60}")
        print(f"  {alg}")
        print(f"{'=' * 60}")
        print(header)
        print(sep)

        for r in subset:
            print(f"{r['n']:>10d}  "
                  f"{r['time_mean']:>6.1f}±{r['time_std']:<5.1f}s  "
                  f"{r['pof_mean']:>6.4f}±{r['pof_std']:<6.4f}  "
                  f"{r['fair_cost_mean']:>6.1f}±{r['fair_cost_std']:<5.1f}  "
                  f"{r['unfair_cost_mean']:>6.1f}±{r['unfair_cost_std']:<5.1f}")

            csv_lines.append(
                f"{alg},{r['n']},"
                f"{r['time_mean']:.2f},{r['time_std']:.2f},"
                f"{r['pof_mean']:.6f},{r['pof_std']:.6f},"
                f"{r['fair_cost_mean']:.2f},{r['fair_cost_std']:.2f},"
                f"{r['unfair_cost_mean']:.2f},{r['unfair_cost_std']:.2f}"
            )

        phases = PHASE_KEYS[alg]
        phase_hdr = f"{'Phase':<24s}" + "".join(f"  {'n=' + str(n):>14s}" for n in sizes)
        print(f"\n  Per-phase breakdown (seconds)")
        print(phase_hdr)
        print("-" * len(phase_hdr))
        for phase in phases + ["Total Time"]:
            row_str = f"{phase:<24s}"
            for r in subset:
                vals = [t.get(phase, 0.0) for t in r["_timings"]]
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                row_str += f"  {m:>6.1f}±{s:<5.1f}s"
            print(row_str)

    csv_path = "evaluation2_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation 2: Scalability and Runtime (RACE_BINARY)"
    )
    parser.add_argument("--csv_path", type=str,
                        default="us_census_puma_data.csv",
                        help="Path to ACS PUMS CSV")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Independent runs N per (algorithm, size) cell (thesis spec: 10)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of clusters k")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Fairness slack α")
    parser.add_argument("--ckpt_dir", type=str, default="evaluation2_partial",
                        help="Per-cell checkpoint directory")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test: tiny sizes, 2 runs")
    args = parser.parse_args()

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    GROUP_ID_FEATURES = ["RACE_BINARY"]
    PROTECTED_COL = "GROUP_ID"

    if args.quick:
        sizes_by_alg = {alg: [1_000, 5_000] for alg in ALGORITHMS}
        args.n_runs = 2
        print("[QUICK MODE] n=[1000, 5000], N=2")
    else:
        sizes_by_alg = dict(SIZES_BY_ALG)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nConfig: k={args.k}, α={args.alpha}, N={args.n_runs}")
    print(f"        CSV={args.csv_path}")
    print(f"        Checkpoint dir={ckpt_dir}")
    for alg in ALGORITHMS:
        print(f"        {alg:8s} sizes={sizes_by_alg[alg]}")
    print()

    all_rows: list[dict] = []

    for alg in ALGORITHMS:
        fc_fn, builder, kw_map = ALG_FUNCTIONS[alg]

        for n in sizes_by_alg[alg]:
            cell_id = f"{alg}_n{n}"
            cell_path = ckpt_dir / f"{cell_id}.json"

            if cell_path.exists():
                print(f"  [cached] {alg} | n={n:>10,d}")
                summary = load_summary(str(cell_path))
                row = _summary_to_row(summary, n, alg)
                all_rows.append(row)
                continue

            print(f"\n  Running {alg} | n={n:>10,d}")
            print("=" * 60)

            extra_kwargs: dict = {
                kw_map["k_centers_kw"]: args.k,
                "alpha": args.alpha,
            }

            # Böhm uses different kwargs (no alpha, has kmedian_trials etc.)
            if alg == "Böhm":
                extra_kwargs = {
                    kw_map["k_centers_kw"]: args.k,
                    "kmedian_trials": 3,
                    "kmedian_max_iter": 30,
                }
            elif alg in ("Bera", "Bercea"):
                extra_kwargs["weight_col"] = None

            summary = run_trials(
                max_rows=n,
                algorithm_fn=fc_fn,
                result_builder=builder,
                group_id_features=GROUP_ID_FEATURES,
                n_runs=args.n_runs,
                csv_path=args.csv_path,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                **extra_kwargs,
            )
            save_summary(summary, str(cell_path))

            row = _summary_to_row(summary, n, alg)
            all_rows.append(row)

    print(f"\n{'=' * 60}\n  Generating Eval 2 plots & tables\n{'=' * 60}")

    for alg in ALGORITHMS:
        plot_per_algorithm_phases(alg, PHASE_KEYS[alg], all_rows)

    plot_overall_comparison(all_rows)
    print_scalability_table(all_rows)
    print("\nEvaluation 2 complete.")
