import os
import json
import argparse
import matplotlib

# Must be set before importing pyplot for headless DigitalOcean droplet execution
matplotlib.use("Agg")

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from fair_clustering.results_encoder import load_summary, save_summary

from fair_clustering.algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from fair_clustering.algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from fair_clustering.algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc
from fair_clustering.algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc

from fair_clustering.runner import (run_trials, build_bera_result, build_bercea_result,
                                    build_backurs_result, build_boehm_result)

FEATURE_CONFIGS = [
    {"name": "SEX", "group_id_features": ["SEX"], "L": 2, "DI": 0.013},
    {"name": "RACE_BINARY", "group_id_features": ["RACE_BINARY"], "L": 2, "DI": 0.288},
    {"name": "AGE_BIN", "group_id_features": ["AGE_BIN"], "L": 4, "DI": 0.042},
    {"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
    {"name": "RACE_6", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    {"name": "AGE_BIN × SEX", "group_id_features": ["AGE_BIN", "SEX"], "L": 8, "DI": 0.040},
    {"name": "RACE_6 × SEX", "group_id_features": ["RACE_6", "SEX"], "L": 12, "DI": 0.311},
]

ALGORITHMS = ["Bera", "Bercea", "Backurs", "Böhm"]

ALG_FUNCTIONS = {
    "Bera": (bera_fc, build_bera_result, {"k_centers_kw": "k_centers"}),
    "Bercea": (bercea_fc, build_bercea_result, {"k_centers_kw": "k_cluster"}),
    "Backurs": (backurs_fc, build_backurs_result, {"k_centers_kw": "k_cluster"}),
    "Böhm": (boehm_fc, build_boehm_result, {"k_centers_kw": "k"}),
}

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


def _summary_to_row(summary, feature_cfg, algorithm) -> dict:
    """Reduce a run_trials summary to the row layout used by the plotting fns."""
    return {
        "feature": feature_cfg["name"],
        "L": feature_cfg["L"],
        "DI": feature_cfg["DI"],
        "algorithm": algorithm,
        "pof_mean": summary["All results PoF (mean)"],
        "pof_std": summary["All results PoF (std)"],
        "fair_cost_mean": summary["All results Fair Cost (mean)"],
        "unfair_cost_mean": summary["All results Unfair Cost (mean)"],
        "all_timings": summary["_timings"]
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

    # Feature labels now incorporate L and DI to answer thesis questions directly
    feature_labels = [f"{f_cfg['name']}\n(L={f_cfg['L']}, DI={f_cfg['DI']:.3f})" for f_cfg in FEATURE_CONFIGS]

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Total time (s)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation3_runtime_by_feature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation3_runtime_by_feature.png")

    for ax_idx, alg in enumerate(algorithms):
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
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
        ax2.set_xticklabels(feature_labels, fontsize=9, rotation=25, ha="right")
        ax2.set_ylabel("Time (s)" if ax_idx == 0 else "", fontsize=11)
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        fig2.tight_layout()
        fig2.savefig(f"evaluation3_{alg}_runtime_by_feature_phases.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved evaluation3_{alg}_runtime_by_feature_phases.png")


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
    csv_path = "evaluation3_results.csv"
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

    feature_labels = [f"{f_cfg['name']}\n(L={f_cfg['L']}, DI={f_cfg['DI']:.3f})" for f_cfg in FEATURE_CONFIGS]

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_ylim(bottom=0.0)
    all_tops = [r["pof_mean"] + r["pof_std"] for r in rows if "pof_mean" in r]
    if all_tops:
        ax.set_ylim(top=max(all_tops + [1.0]) * 1.15)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("evaluation3_pof_by_feature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved evaluation3_pof_by_feature.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation 3 (Scaling w/ Colors and POF)")
    parser.add_argument("--csv_path", type=str,
                        default="us_census_puma_data.csv",
                        help="Path to ACS PUMS CSV")
    parser.add_argument("--n_size", type=int, default=44_000,
                        help="Sample size n (thesis spec: 44,000 for Race6xSex reqs)")
    parser.add_argument("--n_runs", type=int, default=30,
                        help="Independent runs N (thesis spec: 30)")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters k")
    parser.add_argument("--alpha", type=float, default=0.05, help="Slack parameter")
    parser.add_argument("--ckpt_dir", type=str, default="evaluation3_checkpoints",
                        help="Per-cell checkpoint directory")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test: tiny n, 2 runs")
    args = parser.parse_args()

    if args.quick:
        args.n_size = 500
        args.n_runs = 2
        print("[QUICK MODE] n=500, N=2")

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nConfig: n={args.n_size}, k={args.k}, N={args.n_runs}, α={args.alpha}")
    print(f"        CSV={args.csv_path}")
    print(f"        Checkpoint dir={ckpt_dir}\n")

    all_rows: list[dict] = []

    for cfg in FEATURE_CONFIGS:
        fname = cfg["name"]
        print(f"  FEATURE: {fname}  (L={cfg['L']}, DI={cfg['DI']:.3f})")

        for alg in ALGORITHMS:
            cell_id = f"{fname.replace(' ', '_')}_{alg}"
            cell_path = ckpt_dir / f"{cell_id}.json"
            print(f"\n  Checking {alg} | {fname} ...")
            if cell_path.exists():
                print(f"    -> Loaded from checkpoint: {cell_path}")
                summary = load_summary(str(cell_path))
            else:
                print(f"    -> Running {alg} trials...")
                fc_fn, builder, kw_map = ALG_FUNCTIONS[alg]
                extra_kwargs = {kw_map["k_centers_kw"]: args.k}

                if alg != "Böhm":
                    extra_kwargs["alpha"] = args.alpha
                if alg in ("Bera", "Bercea"):
                    extra_kwargs["weight_col"] = None
                if alg == "Böhm":
                    extra_kwargs["kmedian_trials"] = 3
                    extra_kwargs["kmedian_max_iter"] = 30
                    if args.n_runs > 10 and not args.quick:
                        print(
                            f"      [WARNING]: Running Böhm with N={args.n_runs} and n={args.n_size} will take significant time.")

                summary = run_trials(
                    max_rows=args.n_size,
                    algorithm_fn=fc_fn,
                    result_builder=builder,
                    group_id_features=cfg["group_id_features"],
                    n_runs=args.n_runs,
                    csv_path=args.csv_path,
                    feature_cols=FEATURE_COLS,
                    protected_group_col=PROTECTED_COL,
                    **extra_kwargs,
                )
                save_summary(summary, str(cell_path))
                print(f"    -> Saved checkpoint to {cell_path}")

            row = _summary_to_row(summary, cfg, alg)
            all_rows.append(row)

    print(f"\n{'=' * 60}\n  Generating Eval 3 plots & tables\n{'=' * 60}")
    plot_runtime(all_rows)
    print_feature_table(all_rows)
    plot_pof(all_rows)
    print("\nEvaluation 3 complete.")
