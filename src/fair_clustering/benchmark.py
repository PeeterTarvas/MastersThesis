"""
benchmark.py
============

Direct wall-clock measurement of the two subroutines that determine
the practical scale ceiling of the four fair-clustering algorithms in
this thesis:

    1. Hungarian matching (scipy.optimize.linear_sum_assignment) ---
       this is the dominant cost in Bohm.
    2. Fair k-median LP solve (scipy.optimize.linprog with HiGHS) ---
       this is the dominant cost in Bera and Bercea.

Unlike a full asymptotic-fit benchmark, this script measures only at
the sizes that actually appear in the thesis tables. There is no
extrapolation: every number that ends up in the LaTeX comes from a
direct measurement at that size. If a future evaluation uses a size
not in the lists below, add it to MATCHING_SIZES or LP_SIZES and
re-run.

Usage
-----
    python benchmark.py --output_dir results/

Quick smoke test (smaller sizes; verifies the script and the
environment but does not produce thesis-grade numbers):

    python benchmark.py --output_dir results/ --quick

Outputs
-------
    results/matching_walltime.csv  -- one row per (n_max, trial)
    results/lp_walltime.csv        -- one row per (n, trial)
    results/summary.json           -- median + IQR per size,
                                      ready to paste into LaTeX
    results/bohm_per_run.csv       -- M(L) * matching_median, the
                                      direct Bohm per-run estimate

Approximate runtime on a c-8 droplet:
    --quick      ~1 minute
    full run     ~30 minutes
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import eye, kron, lil_matrix

MATCHING_SIZES = [3000, 6000, 10000, 13000, 17000, 35000]

LP_SIZES = [5000, 10000, 25000, 50000, 100000]

RACE6_PROPS = np.array([0.6816, 0.0991, 0.0890, 0.0618, 0.0546, 0.0139])
assert abs(RACE6_PROPS.sum() - 1.0) < 1e-3


@dataclass
class Result:
    name: str
    size: int
    trials: int
    median_s: float
    iqr_s: float
    raw_s: list[float]


def summarise(durations: list[float]) -> tuple[float, float]:
    arr = np.array(durations, dtype=float)
    median = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr


def benchmark_matching(
        sizes: list[int],
        trials: int,
        seed_base: int = 1000,
        ram_limit_gb: float = 12.0,
) -> list[Result]:
    print("\n=== Böhm matching ===")
    print("Inputs: dense uniform-random cost matrices in [0, 1].")
    print(f"{'n_max':>6}  {'trials':>6}  {'median(s)':>12}  {'iqr(s)':>10}")

    out: list[Result] = []
    for n in sizes:
        size_gb = 8 * n * n / (1024 ** 3)
        if size_gb > ram_limit_gb:
            print(f"  skipping n_max={n}: cost matrix {size_gb:.1f} GB "
                  f"> {ram_limit_gb} GB cap")
            continue

        durations: list[float] = []
        for trial in range(trials):
            rng = np.random.default_rng(seed_base + trial)
            cost = rng.random((n, n), dtype=np.float64)
            t0 = time.perf_counter()
            _ = linear_sum_assignment(cost)
            t1 = time.perf_counter()
            durations.append(t1 - t0)
            del cost

        median, iqr = summarise(durations)
        out.append(Result(name="matching", size=n, trials=trials,
                          median_s=median, iqr_s=iqr, raw_s=durations))
        print(f"{n:>6d}  {trials:>6d}  {median:>12.4f}  {iqr:>10.4f}")
    return out


def _build_fair_lp(n: int, k: int, L: int, alpha: float, seed: int):
    """Construct the fair-k-median LP at the given configuration."""
    rng = np.random.default_rng(seed)

    # Synthetic geographic data: 5 Gaussian "cities" + uniform background.
    # This produces realistic-shape distance matrices without requiring
    # the full ACS PUMS file on the benchmark host.
    cities = rng.uniform(0.1, 0.9, size=(5, 2))
    chunks = []
    per_city = n // 6
    for c in cities:
        chunks.append(rng.normal(loc=c, scale=0.05, size=(per_city, 2)))
    chunks.append(rng.uniform(size=(n - 5 * per_city, 2)))
    X = np.clip(np.vstack(chunks)[:n], 0.0, 1.0)

    centres = rng.uniform(size=(k, 2))
    probs = RACE6_PROPS if L == 6 else np.ones(L) / L
    groups = rng.choice(L, size=n, p=probs)

    f = np.array([(groups == h).mean() for h in range(L)])
    lo = np.maximum(0.0, f - alpha)
    hi = np.minimum(1.0, f + alpha)

    # Manhattan (L1) distances, matching the thesis's distance metric.
    D = np.abs(X[:, None, :] - centres[None, :, :]).sum(axis=2)
    cost = D.ravel()

    A_eq = kron(eye(n), np.ones((1, k)), format="csr")
    b_eq = np.ones(n)

    A_ub = lil_matrix((2 * L * k, n * k), dtype=np.float64)
    b_ub = np.zeros(2 * L * k)
    row = 0
    for j in range(k):
        for h in range(L):
            in_h = (groups == h).astype(np.float64)
            cols = np.arange(n) * k + j
            A_ub[row, cols] = lo[h] - in_h
            A_ub[row + 1, cols] = in_h - hi[h]
            row += 2

    return cost, A_eq.tocsc(), b_eq, A_ub.tocsc(), b_ub


def benchmark_fair_lp(
        sizes: list[int],
        k: int,
        L: int,
        alpha: float,
        trials: int,
        seed_base: int = 2000,
) -> list[Result]:
    print("\n=== Fair k-median LP solve (HiGHS dual simplex) ===")
    print(f"k={k}, L={L}, alpha={alpha}, Race-6 marginals.")
    print(f"{'n':>6}  {'trials':>6}  {'median(s)':>12}  {'iqr(s)':>10}")

    out: list[Result] = []
    for n in sizes:
        durations: list[float] = []
        for trial in range(trials):
            cost, A_eq, b_eq, A_ub, b_ub = _build_fair_lp(
                n, k, L, alpha, seed=seed_base + trial,
            )
            t0 = time.perf_counter()
            res = linprog(
                cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0.0, 1.0)] * (n * k),
                method="highs-ds",
                options={"disp": False, "presolve": True},
            )
            t1 = time.perf_counter()
            if res.status != 0:
                print(f"  warning: LP status {res.status} at n={n}, "
                      f"trial={trial}: {res.message}")
            durations.append(t1 - t0)

        median, iqr = summarise(durations)
        out.append(Result(name="fair_lp", size=n, trials=trials,
                          median_s=median, iqr_s=iqr, raw_s=durations))
        print(f"{n:>6d}  {trials:>6d}  {median:>12.4f}  {iqr:>10.4f}")
    return out


def compute_bohm_per_run(matching_results: list[Result]) -> list[dict]:
    """Compute M(L) * t_match for each measured n_max, at L in {2, 6}."""
    rows = []
    for r in matching_results:
        for L in (2, 6):
            M = L * (L - 1)
            t_run = M * r.median_s
            n_for_attribute = round(r.size / 0.6816)
            rows.append({
                "L": L,
                "M_matchings": M,
                "n_max": r.size,
                "n_for_attribute": n_for_attribute,
                "t_match_median_s": r.median_s,
                "t_run_s": t_run,
                "t_run_min": t_run / 60.0,
                "t_run_h": t_run / 3600.0,
            })
    return rows


def hardware_fingerprint() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import numpy as _np
        info["numpy_version"] = _np.__version__
    except Exception:
        pass
    try:
        import scipy as _sp
        info["scipy_version"] = _sp.__version__
    except Exception:
        pass
    try:
        import os
        info["cpu_count_logical"] = os.cpu_count()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results", type=Path)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller sizes, fewer trials. Smoke test only.")
    ap.add_argument("--ram_limit_gb", default=12.0, type=float,
                    help="Skip matching sizes whose cost matrix exceeds this.")
    ap.add_argument("--trials", default=10, type=int,
                    help="Number of trials per size (default 10).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        match_sizes = [1000, 3000, 6816]
        lp_sizes = [1000, 5000]
        trials = 3
    else:
        match_sizes = MATCHING_SIZES
        lp_sizes = LP_SIZES
        trials = args.trials

    print("=== Hardware fingerprint ===")
    hw = hardware_fingerprint()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    matching_results = benchmark_matching(
        sizes=match_sizes, trials=trials,
        ram_limit_gb=args.ram_limit_gb,
    )
    lp_results = benchmark_fair_lp(
        sizes=lp_sizes, k=10, L=6, alpha=0.05, trials=trials,
    )

    # Save raw and summary CSVs.
    raw_match = [
        {"n_max": r.size, "trial": i, "time_s": t}
        for r in matching_results
        for i, t in enumerate(r.raw_s)
    ]
    pd.DataFrame(raw_match).to_csv(
        args.output_dir / "matching_walltime.csv", index=False)

    raw_lp = [
        {"n": r.size, "trial": i, "time_s": t}
        for r in lp_results
        for i, t in enumerate(r.raw_s)
    ]
    pd.DataFrame(raw_lp).to_csv(
        args.output_dir / "lp_walltime.csv", index=False)

    bohm_rows = compute_bohm_per_run(matching_results)
    pd.DataFrame(bohm_rows).to_csv(
        args.output_dir / "bohm_per_run.csv", index=False)

    summary = {
        "hardware": hw,
        "matching": [{
            "n_max": r.size,
            "trials": r.trials,
            "median_s": r.median_s,
            "iqr_s": r.iqr_s,
        } for r in matching_results],
        "fair_lp": [{
            "n": r.size,
            "trials": r.trials,
            "median_s": r.median_s,
            "iqr_s": r.iqr_s,
        } for r in lp_results],
        "bohm_per_run": bohm_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))

    # Print the Bohm per-run table the way it'll appear in the LaTeX.
    print("\n=== Bohm per-run estimates from measured matching times ===")
    print(f"{'L':>2}  {'M':>2}  {'n_max':>6}  {'n':>6}  "
          f"{'t_run (s)':>10}  {'t_run (min)':>11}  {'t_run (h)':>9}")
    for r in bohm_rows:
        print(f"{r['L']:>2}  {r['M_matchings']:>2}  "
              f"{r['n_max']:>6}  {r['n_for_attribute']:>6}  "
              f"{r['t_run_s']:>10.1f}  "
              f"{r['t_run_min']:>11.2f}  "
              f"{r['t_run_h']:>9.3f}")

    print(f"\nWrote: {args.output_dir / 'matching_walltime.csv'}")
    print(f"Wrote: {args.output_dir / 'lp_walltime.csv'}")
    print(f"Wrote: {args.output_dir / 'bohm_per_run.csv'}")
    print(f"Wrote: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
