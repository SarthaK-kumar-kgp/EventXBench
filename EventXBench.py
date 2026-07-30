"""EventXBench dataset loading script for Hugging Face `datasets` library.

This script is auto-detected by HF when the repo contains a .py file with the
same name as the repo. It defines dataset configs for each task (t1--t6) and
for the auxiliary data (posts, markets, ohlcv).

Usage:
    from datasets import load_dataset

    # Load a specific task
    ds = load_dataset("mlsys-io/EventXBench", "t1")
    train_df = ds["train"].to_pandas()

    # Load all configs
    ds = load_dataset("mlsys-io/EventXBench", "t4")
"""
from __future__ import annotations

import json
import os

import datasets


_DESCRIPTION = (
    "EventX: A multimodal benchmark linking Twitter/X posts to "
    "Polymarket prediction market dynamics across seven tasks."
)

_HOMEPAGE = "https://github.com/mlsys-io/EventXBench"
_LICENSE = "cc-by-nc-4.0"

# T3 needs an explicit schema (unlike every other config, which still relies
# on auto-inference from the first write batch). Its "train" split's first
# ~2,325 rows are a contiguous run of `auto`-labeled rows (all four
# deterministic checks passed), so llm_grade/llm_confidence/threshold/all the
# flag columns are 100% null in that first batch. datasets' auto-inference
# would lock those in as an untyped null column, then crash the moment a real
# value shows up later ("Couldn't cast array of type string to null"). This is
# the union of "train" and "gold" columns; missing keys per-split are filled
# null automatically by the Arrow writer once features are explicit.
T3_FEATURES = datasets.Features(
    {
        "tweet_id": datasets.Value("int64"),
        "condition_id": datasets.Value("string"),
        "tweet": datasets.Value("string"),
        "question": datasets.Value("string"),
        "description": datasets.Value("string"),
        "predicate": datasets.Value("string"),
        "requires_official": datasets.Value("bool"),
        # train-only
        "deadline": datasets.Value("string"),
        "threshold": datasets.Value("string"),
        "final_grade": datasets.Value("float64"),
        "label_source": datasets.Value("string"),
        "candidate_grade": datasets.Value("float64"),
        "llm_grade": datasets.Value("float64"),
        "llm_confidence": datasets.Value("float64"),
        "check_source": datasets.Value("string"),
        "check_time": datasets.Value("string"),
        "check_threshold": datasets.Value("string"),
        "check_predicate": datasets.Value("string"),
        "needs_llm": datasets.Value("bool"),
        "predicate_satisfied": datasets.Value("bool"),
        "flag_conditional": datasets.Value("bool"),
        "flag_sarcasm": datasets.Value("bool"),
        "flag_source_unclear": datasets.Value("bool"),
        "flag_threshold_ambiguous": datasets.Value("bool"),
        "needs_human_review": datasets.Value("bool"),
        "created_at": datasets.Value("string"),
        # gold-only
        "gold_grade": datasets.Value("int64"),
        "resolution_method": datasets.Value("string"),
        "A1_grade": datasets.Value("float64"),
        "A2_grade": datasets.Value("float64"),
        "A3_grade": datasets.Value("float64"),
    }
)

_URLS = {
    "t1_train": "data/t1/train.jsonl",
    "t1_test": "data/t1/test.jsonl",
    "t2_test": "data/t2/test.jsonl",
    "t3_train": "data/t3/train.jsonl",
    "t3_gold": "data/t3/gold.jsonl",
    "t4_train": "data/t4/train.jsonl",
    "t4_test": "data/t4/test.jsonl",
    "t5_train": "data/t5/train.jsonl",
    "t5_test": "data/t5/test.jsonl",
    "t6_train": "data/t6/train.jsonl",
    "t6_validation": "data/t6/validation.jsonl",
    "t6_test": "data/t6/test.jsonl",
    "t7_train": "data/t7/train.jsonl",
    "t7_test": "data/t7/test.jsonl",
}


class EventXBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for EventXBench."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EventXBench(datasets.GeneratorBasedBuilder):
    """EventXBench dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        EventXBenchConfig(
            name="t1",
            version=VERSION,
            description="T1: Conditional Market Volume Prediction (3-class)",
        ),
        EventXBenchConfig(
            name="t2",
            version=VERSION,
            description="T2: Post-to-Market Linking",
        ),
        EventXBenchConfig(
            name="t3",
            version=VERSION,
            description=(
                "T3: Evidence Grading (ordinal 0-5). 'train' split = full "
                "silver-labeled export (final_grade); 'gold' split = the "
                "separate, rare-grade-enriched, human-adjudicated audit pool "
                "(gold_grade) - the actual held-out ground truth."
            ),
        ),
        EventXBenchConfig(
            name="t4",
            version=VERSION,
            description="T4: Market Movement Prediction (direction x magnitude)",
        ),
        EventXBenchConfig(
            name="t5",
            version=VERSION,
            description="T5: Volume & Price Impact (decay classification)",
        ),
        EventXBenchConfig(
            name="t6",
            version=VERSION,
            description="T6: Cross-Market Propagation (3-class)",
        ),
        EventXBenchConfig(
            name="t7",
            version=VERSION,
            description="T7: Impact Persistence / Decay classification (3-class)",
        ),
    ]

    DEFAULT_CONFIG_NAME = "t1"

    def _info(self):
        # Every other config still relies on auto-inference from the first
        # write batch (features=None) - unchanged, verified safe. T3 alone
        # needs an explicit schema; see T3_FEATURES above for why.
        features = T3_FEATURES if self.config.name == "t3" else None
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        config = self.config.name

        # Determine which files to download
        files_to_dl = {}
        for key, url in _URLS.items():
            if key.startswith(config + "_"):
                files_to_dl[key] = url

        downloaded = dl_manager.download_and_extract(files_to_dl)

        splits = []
        train_key = f"{config}_train"
        validation_key = f"{config}_validation"
        test_key = f"{config}_test"
        gold_key = f"{config}_gold"

        if train_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded[train_key]},
                )
            )
        if validation_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded[validation_key]},
                )
            )
        if test_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": downloaded[test_key]},
                )
            )
        if gold_key in downloaded:
            # Custom named split - human-adjudicated audit pool (T3 only).
            # Not a train/validation/test split; do not conflate with TEST.
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.NamedSplit("gold"),
                    gen_kwargs={"filepath": downloaded[gold_key]},
                )
            )

        return splits

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    yield idx, json.loads(line)
