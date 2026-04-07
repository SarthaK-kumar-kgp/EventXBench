#!/usr/bin/env python3
"""T6 Basic Baselines -- Cross-Market Propagation.

Evaluates majority-class and random-prior baselines for T6 cross-market
propagation classification.  Reports Macro-F1.

Usage:
    python -m baselines.t6.basic_baseline
    python -m baselines.t6.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import eventxbench

LABEL_ORDER = ["no_cross_market_effect", "primary_mover", "propagated_signal"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p == lab)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != lab and p == lab)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _majority_baseline(y_true: list[str], labels: list[str]) -> dict:
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    mf1 = _macro_f1(y_true, y_pred, labels)
    return {
        "baseline": "majority",
        "majority_label": majority,
        "n": len(y_true),
        "macro_f1": mf1,
    }


def _random_baseline(y_true: list[str], labels: list[str], seeds: list[int] | None = None, train_labels: list[str] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    prior_source = train_labels if train_labels is not None else y_true
    counts = Counter(prior_source)
    total = len(y_true)
    priors = np.array([counts.get(lab, 0) / len(prior_source) for lab in labels])

    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(labels, size=total, p=priors).tolist()
        f1_scores.append(_macro_f1(y_true, y_pred, labels))

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n": total,
        "mean_macro_f1": float(np.mean(f1_scores)),
        "per_seed_macro_f1": [round(f, 4) for f in f1_scores],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 basic baselines")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--feature-file", default=None,
                        help="Path to feature JSONL with split column (e.g. t6_db_features.jsonl)")
    args = parser.parse_args()

    if args.feature_file:
        full_df = pd.read_json(args.feature_file, lines=True)
        full_df = full_df[full_df["insufficient_data_flag"] == False].copy()
        full_df = full_df[full_df["label"].isin(LABEL_ORDER)].copy()
        train_df = full_df[full_df["split"] == "train"].copy()
        test_df = full_df[(full_df["split"] == "test") & (full_df["confound_flag"] == False)].copy()
    else:
        data = eventxbench.load_task("t6", local_dir=args.local_dir)
        if isinstance(data, tuple):
            train_df, test_df = data
            full_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            full_df = data

        if "insufficient_data_flag" in full_df.columns:
            full_df = full_df[full_df["insufficient_data_flag"] == False].reset_index(drop=True)
        full_df = full_df[full_df["label"].isin(LABEL_ORDER)].reset_index(drop=True)

        if "split" in full_df.columns:
            train_df = full_df[full_df["split"] == "train"].copy()
            test_df = full_df[full_df["split"] == "test"].copy()
        else:
            split_idx = int(len(full_df) * 0.8)
            train_df = full_df.iloc[:split_idx].copy()
            test_df = full_df.iloc[split_idx:].copy()

        if "confound_flag" in test_df.columns:
            test_df = test_df[test_df["confound_flag"] == False].reset_index(drop=True)

    eval_labels = LABEL_ORDER
    y_true = test_df["label"].tolist()

    print(f"T6 train: {len(train_df)}, test: {len(test_df)}")
    print(f"Test class distribution: {dict(Counter(y_true))}")

    # Majority baseline (majority from train)
    majority_label = train_df["label"].value_counts().idxmax()
    y_pred_maj = [majority_label] * len(y_true)
    mf1_maj = _macro_f1(y_true, y_pred_maj, eval_labels)
    print(f"\n[Majority] always predict '{majority_label}'")
    print(f"  Macro-F1: {mf1_maj:.4f}")

    # Random baseline (priors from train)
    rand = _random_baseline(y_true, eval_labels, train_labels=train_df["label"].tolist())
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Macro-F1: {rand['mean_macro_f1']:.4f}")
    print(f"  Per-seed: {rand['per_seed_macro_f1']}")


if __name__ == "__main__":
    main()
