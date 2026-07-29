# EventX Leaderboard

Results reported on the standard test splits. All classification metrics are macro-averaged.

## Resolution Tier

### T2: Post-to-Market Linking

| Model | Acc@1 | MRR |
|-------|-------|-----|
| BGE top-1 (retrieval only) | 0.379 | 0.525 |
| BM25 top-1 | 0.072 | 0.107 |
| GPT-4o (0-shot) | **0.659** | **0.779** |
| GPT-4o (3-shot) | 0.650 | 0.772 |
| Sonnet 4.5 (0-shot) | 0.625 | 0.779 |
| Sonnet 4.5 (3-shot) | 0.649 | 0.763 |
| Grok 4.1 (0-shot) | 0.564 | 0.728 |
| Grok 4.1 (3-shot) | 0.555 | 0.718 |
| Qwen3.5-4B (0-shot) | 0.320 | 0.577 |
| Qwen3.5-4B (3-shot) | 0.569 | 0.738 |
| Qwen3.5-27B (0-shot) | 0.620 | 0.757 |
| Qwen3.5-27B (3-shot) | 0.605 | 0.750 |

### T3: Evidence Grading vs. Human Gold (2,687-instance gold audit pool)

Evaluated against the human-adjudicated `gold_grade` (T3_Reproducible_Package's
2,687-instance, rare-grade-enriched audit pool) - **not** silver `final_grade`.
The two are not interchangeable ground truth; see `data/README.md`.

**Anchors:** silver pipeline κ_w = 0.582 (fails the project's own 0.6 reliability
bar); human inter-annotator agreement κ_w = 0.775–0.848 (`metrics.md` Phase 5).
κ_w / κ_u = quadratic-weighted / unweighted Cohen's κ. Models served via
provider APIs.

| Model | 0-shot κ_w | 0-shot κ_u | 0-shot F1 | 3-shot κ_w | 3-shot κ_u | 3-shot F1 |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Gemma-4-26B | 0.652 | 0.230 | 0.331 | 0.688 | 0.323 | 0.417 |
| GPT-4o | 0.659 | 0.190 | 0.288 | 0.699 | 0.221 | 0.330 |
| Grok-4.5 | 0.637 | 0.197 | 0.306 | **0.774** | **0.416** | **0.447** |
| Qwen3.6-35B | 0.564 | 0.266 | 0.360 | 0.663 | 0.349 | 0.430 |
| DeepSeek-V3 | 0.645 | 0.213 | 0.304 | 0.618 | 0.331 | 0.421 |
| Claude Sonnet-5 | 0.564 | 0.247 | 0.328 | 0.539 | 0.264 | 0.344 |

Even the best 3-shot result (Grok-4.5, κ_w 0.774) sits below the human IAA
floor (0.775–0.848) and only modestly above the silver pipeline's own
agreement with gold (κ_w 0.582) - evidence grading remains unsolved at the
gold standard, and 3-shot doesn't reliably help (Claude Sonnet-5, DeepSeek-V3
both *regress* on κ_w with 3-shot).

#### Non-LLM baselines vs. silver (`final_grade`, 70/30 market-level split, `random_state=42`)

| Method | Kappa (unweighted) | Macro-F1 |
|--------|---------------------|----------|
| Majority | 0.0000 | 0.0853 |
| Random (single seed=42) | 0.0038 | 0.1709 |
| LightGBM (`requires_official` + tweet/predicate embeddings) | 0.4562 | 0.3963 |

Reference numbers from `T3_Reproducible_Package/metrics.md` Phase 6 - pending
re-run in this repo with the corrected `baselines/t3/basic_baseline.py` /
`lgbm_baseline.py` once the underlying data files are rebuilt. **The previous
"Pre-check pipeline: 0.686 QWK" / "LightGBM: 0.849 QWK" rows here were removed**
- they were produced by a mislabeled auto-grade shortcut and a feature set that
included near-label-leaking deterministic-check columns, both fixed on this
branch; those old numbers should not be cited.

## Forecast Tier

### T1: Conditional Market Volume Prediction

| Model | Macro-F1 | P@10 |
|-------|----------|------|
| LightGBM | **0.508** | **0.600** |
| GPT-4o (0-shot) | 0.266 | 0.600 |
| GPT-4o (3-shot) | 0.343 | 0.400 |
| Sonnet 4.5 (0-shot) | 0.362 | 0.600 |
| Sonnet 4.5 (3-shot) | 0.334 | 0.500 |
| Grok 4.1 (0-shot) | 0.329 | 0.300 |
| Grok 4.1 (3-shot) | 0.297 | 0.400 |
| Qwen3.5-4B (0-shot) | 0.304 | 0.400 |
| Qwen3.5-4B (3-shot) | 0.276 | 0.400 |
| Qwen3.5-27B (0-shot) | 0.240 | 0.300 |
| Qwen3.5-27B (3-shot) | 0.315 | 0.200 |

### T4: Market Movement Prediction

| Model | Dir-Acc | Mag Macro-F1 | Spearman rho |
|-------|---------|--------------|--------------|
| Random walk | 0.503 | 0.364 | 0.011 |
| LightGBM | **0.682** | 0.556 | 0.322 |
| GPT-4o (0-shot) | 0.605 | 0.370 | 0.121 |
| GPT-4o (3-shot) | 0.648 | 0.366 | 0.181 |
| GPT-4o + image | 0.590 | 0.373 | 0.084 |
| GPT-4o + prices | 0.638 | 0.373 | 0.120 |
| Sonnet 4.5 (0-shot) | 0.598 | 0.545 | 0.209 |
| Sonnet 4.5 (3-shot) | 0.604 | **0.554** | **0.356** |
| Grok 4.1 (0-shot) | 0.560 | 0.479 | 0.317 |
| Grok 4.1 (3-shot) | 0.585 | 0.447 | 0.295 |
| Qwen3.5-4B (0-shot) | 0.597 | 0.381 | -0.050 |
| Qwen3.5-4B (3-shot) | 0.581 | 0.263 | -0.045 |
| Qwen3.5-27B (0-shot) | 0.623 | 0.448 | 0.037 |
| Qwen3.5-27B (3-shot) | 0.619 | 0.449 | 0.118 |

### T5: Volume and Price Impact (Continuous)

| Model | rho (price_impact) | rho (volume_multiplier) |
|-------|-------------------|------------------------|
| LightGBM | 0.343 | 0.363 |
| GPT-4o (0-shot) | 0.056 | -0.039 |
| GPT-4o + prices | 0.268 | -0.046 |
| Sonnet 4.5 (0-shot) | 0.308 | 0.073 |
| Sonnet 4.5 (3-shot) | **0.417** | **0.211** |
| Grok 4.1 (0-shot) | 0.176 | -0.054 |
| Grok 4.1 (3-shot) | 0.135 | 0.019 |
| Qwen3.5-4B (0-shot) | -0.038 | -0.122 |
| Qwen3.5-4B (3-shot) | 0.067 | 0.068 |
| Qwen3.5-27B (0-shot) | 0.179 | 0.078 |
| Qwen3.5-27B (3-shot) | 0.331 | 0.162 |

### T7: Impact Persistence (Decay Classification)

| Model | Macro-F1 |
|-------|----------|
| LightGBM | **0.518** |
| GPT-4o (0-shot) | 0.296 |
| GPT-4o + prices | 0.285 |
| Sonnet 4.5 (0-shot) | 0.277 |
| Sonnet 4.5 (3-shot) | 0.251 |
| Grok 4.1 (0-shot) | 0.284 |
| Grok 4.1 (3-shot) | 0.273 |
| Qwen3.5-4B (0-shot) | 0.309 |
| Qwen3.5-4B (3-shot) | 0.303 |
| Qwen3.5-27B (0-shot) | 0.290 |
| Qwen3.5-27B (3-shot) | 0.320 |

### T6: Cross-Market Propagation

| Model | Macro-F1 | MAE (min) |
|-------|----------|-----------|
| Graph heuristic | 0.250 | 24.49 |
| LightGBM | 0.274 | 24.56 |
| GPT-4o (0-shot) | 0.334 | -- |
| **GPT-4o (3-shot)** | **0.345** | -- |
| Sonnet 4.5 (0-shot) | 0.237 | -- |
| Sonnet 4.5 (3-shot) | 0.259 | -- |
| Grok 4.1 (0-shot) | 0.319 | -- |
| Grok 4.1 (3-shot) | 0.288 | -- |
| Qwen3.5-4B (0-shot) | 0.306 | -- |
| Qwen3.5-4B (3-shot) | 0.251 | -- |
| Qwen3.5-27B (0-shot) | 0.301 | -- |
| Qwen3.5-27B (3-shot) | 0.295 | -- |

## How to Submit

1. Run your model on the test split for each task
2. Format predictions per [`evaluation/README.md`](evaluation/README.md)
3. Evaluate locally: `python evaluation/evaluate.py --task all --predictions-dir results/`
4. Open a pull request adding your row with a link to your method
