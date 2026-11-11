# Dataset Difficulty V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the difficulty score semantics, add group-safe splitting,
instrument real CLIP runtime, and validate the complete workflow on CIFAR-10.

**Architecture:** Preserve the existing loading → quality → anomaly → CLIP →
semantics → scoring → split → report pipeline. Add transparent component
columns and a separate group-aware branch. Runtime metrics and validation
scripts remain thin adapters around the existing CLI.

**Tech Stack:** Python 3.11+, OpenCV, Pandas, PyTorch, Transformers CLIP,
Matplotlib, pytest, torchvision CIFAR-10.

**Spec:** `docs/superpowers/specs/2026-11-11-dataset-difficulty-v2-design.md`

## Global Constraints

- Do not replace existing architecture or remove working features.
- Label Rarity must not affect `difficulty_score`.
- Entropy and frequency complexity must not be assumed to imply difficulty.
- Anomaly samples remain separate from difficulty bands.
- Empty metrics remain null.
- `group_id` groups must never span splits.
- No simulated embeddings may be reported as real CLIP validation.
- Do not claim GPU/CPU completion without recorded real outputs.

## Review Focus

- Visual-only scoring must not be presented as full semantic difficulty.
- Group-aware assignment must preserve ratios while preventing leakage.
- Existing no-group split output must remain stable.
- Real CLIP cache metadata must distinguish hits, misses, and failures.
- CIFAR-10 32x32 images must not all be marked invalid by a 64px default.

---

### Task 1: Correct Difficulty Components and Report Interpretation

**Files:**
- Modify: `src/dataset_difficulty/scoring.py`
- Modify: `src/dataset_difficulty/reports.py`
- Modify: `src/dataset_difficulty/pipeline.py`
- Modify: `configs/default.yaml`
- Modify: `src/dataset_difficulty/default.yaml`
- Test: `tests/test_scoring.py`
- Test: `tests/test_reports.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces columns: `visual_accessibility_risk`,
  `visual_complexity_score`, `difficulty_visual_component`,
  `difficulty_visual_weight`, `difficulty_visual_effective_weight`,
  `visual_contribution`, `difficulty_semantic_component`,
  `difficulty_semantic_weight`, `difficulty_semantic_effective_weight`,
  `semantic_contribution`, `difficulty_scope`, and
  `label_rarity_in_difficulty`.
- `score_difficulty(...)` keeps its signature.
- Summary adds `difficulty_scope` and `score_interpretation`.

- [ ] **Step 1: Write failing scoring tests**

Assert label rarity changes do not change `difficulty_score`; entropy and
frequency changes do not change the score; contributions sum to the score;
visual-only rows have scope `visual_only`; reports include raw visual metrics
and explain that visual complexity is descriptive.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_scoring.py tests/test_reports.py -v`
Expected: FAIL on missing contribution/scope contracts.

- [ ] **Step 3: Implement transparent two-component scoring and report text**

Use only blur/brightness/contrast for visual accessibility. Put entropy and
frequency into an unweighted descriptive complexity statistic. Use configured
visual/semantic weights normalized over available components.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_scoring.py tests/test_reports.py -v`
Expected: PASS.

### Task 2: Add Optional Group-Aware Dataset Splitting

**Files:**
- Modify: `src/dataset_difficulty/loading.py`
- Modify: `src/dataset_difficulty/splitting.py`
- Modify: `src/dataset_difficulty/pipeline.py`
- Modify: `src/dataset_difficulty/cli.py`
- Modify: `src/dataset_difficulty/reports.py`
- Modify: `configs/default.yaml`
- Modify: `src/dataset_difficulty/default.yaml`
- Test: `tests/test_loading.py`
- Test: `tests/test_splitting.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- `load_dataset(..., group_column: str = "group_id") -> DatasetLoadResult`
- Loaded samples include `group_id`.
- `assign_splits(...)` preserves its signature and automatically branches
  when non-empty `group_id` values exist.
- Split summary adds `group_count`, `groups_spanning_splits`,
  `duplicate_groups_spanning_splits`, `class_distribution`,
  `difficulty_distribution`, and `leakage_risk`.
- CLI adds `--group-column`.

- [ ] **Step 1: Write failing group tests**

Assert CSV group IDs load; same group appears only once across splits; mixed
groups remain intact; blank groups use sample-level units; no-group behavior
matches the existing counts; leakage diagnostics are present.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_loading.py tests/test_splitting.py -v`
Expected: FAIL because `group_id` and group metadata are absent.

- [ ] **Step 3: Implement loader, CLI, group allocator, and diagnostics**

Retain the existing row-level allocator for blank group IDs. Use a deterministic
whole-group greedy allocator for non-empty groups.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_loading.py tests/test_splitting.py tests/test_cli_e2e.py -v`
Expected: PASS.

### Task 3: Instrument Real Semantic Runtime and Cache Behavior

**Files:**
- Modify: `src/dataset_difficulty/features.py`
- Modify: `src/dataset_difficulty/pipeline.py`
- Test: `tests/test_features.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- `ClipFeatureExtractor.runtime_info() -> dict`
- `AnalysisResult.summary["performance"]` and
  `AnalysisResult.summary["semantic_runtime"]`.
- Runtime fields include device, model, batch size, cache hit/miss counts,
  failures, stage seconds, and CUDA peak memory when applicable.

- [ ] **Step 1: Write failing runtime tests**

Use real small fake-extractor runs to assert stage timing keys, cache
hit/miss counts, failure IDs, and backend/device fields exist. Keep fake
outputs labeled as tests only.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_features.py tests/test_pipeline.py -v`
Expected: FAIL on missing runtime summary contract.

- [ ] **Step 3: Implement timing/cache/runtime instrumentation**

Measure stage boundaries with `time.perf_counter`. Report CUDA peak memory
from PyTorch only when `cuda` is the actual device.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_features.py tests/test_pipeline.py -v`
Expected: PASS.

### Task 4: Add CIFAR-10 Preparation and V1/V2 Comparison Utilities

**Files:**
- Create: `scripts/prepare_cifar10.py`
- Create: `scripts/compare_v1_v2.py`
- Test: `tests/test_validation_scripts.py`
- Modify: `.gitignore`

**Interfaces:**
- `prepare_cifar10.py --root PATH --output PATH --split test --limit 10000`
  writes `images/` and `labels.csv` with group_id column.
- `compare_v1_v2.py --before-csv PATH --after-csv PATH --output PATH` writes
  JSON and Markdown comparison.

- [ ] **Step 1: Write failing script tests**

Create a small fake torchvision-shaped CIFAR directory and assert export
layout, label/group columns, and V1/V2 metric comparison fields.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_validation_scripts.py -v`
Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement scripts**

Use torchvision CIFAR-10 loading and cv2 image writing. Compute V1 scoring
from V2 raw columns without calling the V2 scorer.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_validation_scripts.py -v`
Expected: PASS.

### Task 5: Run Real CLIP CPU/GPU and CIFAR-10 Validation

**Files:**
- Modify: `SKILL.md`
- Modify: `README.md`
- Modify: `references/metrics.md`
- Modify: `references/configuration.md`
- Modify: `references/outputs.md`
- Create: `validation/before_after.md`
- Create: `validation/real_clip/`
- Create: `validation/cifar10_test/`
- Modify: `TEST_RESULTS.md`

- [ ] **Step 1: Download/load real CLIP model with explicit authorization**

Record model ID, Transformers version, PyTorch version, and cache location.

- [ ] **Step 2: Run CPU validation**

Run a small real-image subset with `--device cpu`, batch 8, and unique cache.
Record stage timing, cache misses, failures, and nulls.

- [ ] **Step 3: Run GPU validation**

Run the same input with `--device cuda`, batch 32, and a separate cache.
Record peak allocated/reserved VRAM.

- [ ] **Step 4: Verify cache hit and resume**

Rerun each identical configuration and confirm zero new embedding calls from
cache hit counts. Then run a partial-cache worker and confirm remaining images
are extracted without recomputing hits.

- [ ] **Step 5: Export and analyze CIFAR-10 test split**

Export 10,000 canonical test images at `min_image_size: 32`, run the full CLI
on GPU with real CLIP, and verify all required artifacts.

- [ ] **Step 6: Generate V1/V2 comparison**

Run `compare_v1_v2.py` on the CIFAR-10 `samples.csv` output and document score,
weight, and band changes.

- [ ] **Step 7: Final validation**

Run the complete pytest suite, Skill validator, inspect the CIFAR-10 HTML and
preview images, and record real limitations.

### Task 6: Review and Publish Branch

- [ ] **Step 1: Run fresh whole-branch code review**
- [ ] **Step 2: Fix Critical/Important findings with RED→GREEN tests**
- [ ] **Step 3: Commit all changes**
- [ ] **Step 4: Push the feature branch and report the GitHub URL**
