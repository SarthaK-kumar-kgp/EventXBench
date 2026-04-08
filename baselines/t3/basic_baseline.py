#!/usr/bin/env python3
"""T3 Pre-check Pipeline Baseline -- Evidence Grading.

Evaluates majority-class, random, and pre-check pipeline baselines
for T3 evidence grading using a market-level 70/30 train-test split.
Reports Cohen's Kappa and Macro F1.

Usage:
    python t3_precheck_baseline.py
    python t3_precheck_baseline.py --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

import eventxbench


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def split_by_market(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = df["condition_id"].unique()
    train_markets, test_markets = train_test_split(
        markets, test_size=test_size, random_state=random_state
    )
    train_df = df[df["condition_id"].isin(train_markets)].copy()
    test_df = df[df["condition_id"].isin(test_markets)].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _run_majority(y_true: np.ndarray) -> dict:
    counts = Counter(y_true.tolist())
    majority_class = counts.most_common(1)[0][0]
    y_pred = np.full(len(y_true), majority_class)

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "majority",
        "majority_class": majority_class,
        "n": len(y_true),
        "kappa": kappa,
        "macro_f1": f1,
    }


def _run_random(y_true: np.ndarray, seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    counts = Counter(y_true.tolist())
    grades = sorted(counts.keys())
    total = sum(counts.values())
    priors = np.array([counts[g] / total for g in grades])

    kappas: list[float] = []
    f1s: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(grades, size=len(y_true), p=priors)
        kappas.append(cohen_kappa_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n": len(y_true),
        "mean_kappa": float(np.mean(kappas)),
        "mean_macro_f1": float(np.mean(f1s)),
    }


def _run_precheck_pipeline(test_df: pd.DataFrame) -> dict:
    """Use llm_grade as prediction, filling NaNs with majority class (3)."""
    df = test_df.copy()
    df["candidate_grade_filled"] = df["llm_grade"].fillna(3).astype(int)

    y_true = df["final_grade"].values
    y_pred = df["candidate_grade_filled"].values

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "precheck_pipeline",
        "n": len(y_true),
        "nan_fill_value": 3,
        "kappa": kappa,
        "macro_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 pre-check pipeline baselines")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    df.sort_values(by=["createdAt"], ascending=True, inplace=True)

    # Split
    train_df, test_df = split_by_market(
        df, test_size=args.test_size, random_state=args.random_state
    )

    y_true = test_df["final_grade"].values

    print(f"T3 samples: {len(df)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size:  {len(test_df)}")
    print(f"Grade distribution (test): {dict(sorted(Counter(y_true.tolist()).items()))}")

    # Majority baseline
    maj = _run_majority(y_true)
    print(f"\n[Majority] always predict grade={maj['majority_class']}")
    print(f"  Kappa={maj['kappa']:.4f}, Macro F1={maj['macro_f1']:.4f}")

    # Random baseline
    rand = _run_random(y_true)
    print(f"\n[Random Prior] sample from training distribution (seeds={rand['seeds']})")
    print(f"  Mean Kappa={rand['mean_kappa']:.4f}, Mean Macro F1={rand['mean_macro_f1']:.4f}")

    # Pre-check pipeline baseline
    pre = _run_precheck_pipeline(test_df)
    print(f"\n[Pre-check Pipeline] use llm_grade (NaN → {pre['nan_fill_value']})")
    print(f"  Kappa={pre['kappa']:.4f}, Macro F1={pre['macro_f1']:.4f}")


if __name__ == "__main__":
    main()