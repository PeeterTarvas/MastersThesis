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

from fair_clustering.runner import run_trials, build_bera_result, build_bercea_result, build_backurs_result

ALG_FUNCTIONS = {
    "Bera": (bera_fc, build_bera_result, {"k_centers_kw": "k_centers"}),
    "Bercea": (bercea_fc, build_bercea_result, {"k_centers_kw": "k_cluster"}),
    "Backurs": (backurs_fc, build_backurs_result, {"k_centers_kw": "k_cluster"}),
}

ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.20]

FEATURE_CONFIGS = [
    {"name": "RACE_6", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.3453},
    # {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.0983},
]

ALGORITHMS = ["Bera", "Bercea", "Backurs"]

ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452", "Backurs": "#AAA868"}
ALG_DASH_OFFSETS = {"Bera": (0, ()), "Bercea": (0, ()), "Backurs": (0, (5, 2))}

def _avg_total_time(timings_list: list[dict]) -> tuple[float, float]:
    vals = [t.get("Total Time", 0.0) for t in timings_list]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return m, s


def _summary_to_row(summary, feature_cfg, alpha, algorithm) -> dict:
    """Reduce a run_trials summary to the row layout used by the plotting fns."""
    tm, ts = _avg_total_time(summary["_timings"])
    return {
        "feature": feature_cfg["name"],
        "L": feature_cfg["L"],
        "DI": feature_cfg["DI"],
        "alpha": alpha,
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


def plot_pof_vs_alpha(rows: list[dict], alphas: list[float]) -> None:
    algorithms = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    n_algs = len(algorithms)

    fig, ax = plt.subplots(figsize=(max(8, len(alphas) * 2.2), 5.5))
    x = np.arange(len(alphas))
    width = 0.8 / n_algs

    for feat_cfg in FEATURE_CONFIGS:
        fname = feat_cfg["name"]
        for i, alg in enumerate(algorithms):
            means, stds = [], []
            for alpha in alphas:
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
    ax.set_xticklabels([str(a) for a in alphas], fontsize=10)
    ax.set_xlabel("α (fairness slack)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation4_pof_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
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
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation4_runtime_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation4_runtime_vs_alpha.png")


def print_alpha_table(rows: list[dict], alphas: list[float]) -> None:
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
        for alpha in alphas:
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
    csv_path = "evaluation4_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation 4 (effect of α)")
    parser.add_argument("--csv_path", type=str,
                        default="../../../us_census_puma_data.csv",
                        help="Path to ACS PUMS CSV")
    parser.add_argument("--n_size", type=int, default=25_000,
                        help="Sample size n (thesis spec: 25 000)")
    parser.add_argument("--n_runs", type=int, default=30,
                        help="Independent runs N (thesis spec: 30)")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters k")
    parser.add_argument("--ckpt_dir", type=str, default="evaluation4_partial",
                        help="Per-cell checkpoint directory")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test: tiny n, 2 runs, single α")
    args = parser.parse_args()

    if args.quick:
        args.n_size = 500
        args.n_runs = 2
        alphas_to_run = [0.05]
        print("[QUICK MODE] n=500, N=2, α=[0.05]")
    else:
        alphas_to_run = ALPHAS

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nConfig: n={args.n_size}, k={args.k}, N={args.n_runs}")
    print(f"        αs={alphas_to_run}")
    print(f"        CSV={args.csv_path}")
    print(f"        Checkpoint dir={ckpt_dir}\n")

    all_rows: list[dict] = []

    for feature_cfg in FEATURE_CONFIGS:
        fname = feature_cfg["name"]
        for alpha in alphas_to_run:
            for alg in ALGORITHMS:
                cell_id = f"{fname}_{alg}_alpha{alpha}"
                cell_path = ckpt_dir / f"{cell_id}.json"
                print(f"\n  Running {alg} | {fname} | α={alpha}")
                fc_fn, builder, kw_map = ALG_FUNCTIONS[alg]
                extra_kwargs = {kw_map["k_centers_kw"]: args.k, "alpha": alpha}
                if alg in ("Bera", "Bercea"):
                    extra_kwargs["weight_col"] = None

                summary = run_trials(
                    max_rows=args.n_size,
                    algorithm_fn=fc_fn,
                    result_builder=builder,
                    group_id_features=feature_cfg["group_id_features"],
                    n_runs=args.n_runs,
                    csv_path=args.csv_path,
                    feature_cols=FEATURE_COLS,
                    protected_group_col=PROTECTED_COL,
                    **extra_kwargs,
                )
                save_summary(summary, str(cell_path))

                row = _summary_to_row(summary, feature_cfg, alpha, alg)
                all_rows.append(row)

    print(f"\n{'=' * 60}\n  Generating Eval 4 plots & tables\n{'=' * 60}")
    plot_pof_vs_alpha(all_rows, alphas_to_run)
    plot_runtime_vs_alpha(all_rows)
    print_alpha_table(all_rows, alphas_to_run)
    print("\nEvaluation 4 complete.")
