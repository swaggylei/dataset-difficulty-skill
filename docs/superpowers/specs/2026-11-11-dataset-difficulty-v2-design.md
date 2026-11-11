# Dataset Difficulty Skill V2 Design

## Intent

Improve the existing training-free difficulty prior without replacing its
architecture. V2 corrects the interpretation of label frequency and visual
complexity, adds leakage-safe group splitting, and requires real CLIP and
CIFAR-10 validation rather than simulated-only evidence.

## Goals

- Remove label rarity from the composite difficulty score.
- Keep label rarity as an independent class-distribution statistic.
- Preserve every difficulty component, configured weight, and contribution.
- Separate image quality, descriptive visual complexity, and difficulty proxy.
- Stop treating higher entropy or frequency complexity as intrinsically harder.
- Support optional `group_id` and keep every group in one split.
- Report split ratios, class/difficulty distributions, and leakage risk.
- Record real CLIP device, batch, cache, timing, VRAM, and failures.
- Validate on real CIFAR-10 data.

## Non-Goals

- Learning a supervised difficulty label from downstream model loss.
- Claiming that the composite score is ground-truth task difficulty.
- Replacing the current anomaly, cache, or reporting architecture.
- Treating CPU-only or simulated-embedding tests as GPU/model validation.

## Scoring Model

The composite prior has exactly two task-facing components:

1. `visual_accessibility_risk`: a dataset-relative accessibility signal from
   low sharpness, exposure extremes, and low contrast. It is not an assertion
   that poor quality equals semantic difficulty.
2. `semantic_complexity`: the existing CLIP centroid, confusion, and
   augmentation-instability composite.

`label_rarity_score` remains available with `class_count`, but its weight and
contribution are zero and `label_rarity_in_difficulty` is false.

`visual_complexity_score` combines percentile ranks of image entropy, spectral
entropy, and high-frequency energy ratio as a descriptive statistic only.
Those signals have no weight in the composite score.

For each main component the output stores:

- raw component value;
- configured weight;
- effective normalized weight among available components;
- contribution value;
- availability flag.

The service also reports `difficulty_scope=visual_semantic` or `visual_only`.
When CLIP is unavailable, a prominent warning states that the score is based
on visual accessibility only and is not the complete semantic difficulty
prior.

The existing `visual_difficulty_component` column is retained as a
compatibility alias for `visual_accessibility_risk`.

## Visual Interpretation

The report shows raw blur, brightness, contrast, image entropy, spectral
entropy, high-frequency ratio, absolute quality risk, descriptive visual
complexity, visual accessibility risk, semantic complexity, and the composite
score. The HTML explains that high texture or entropy is descriptive, not
automatically difficult.

## Group-Aware Splitting

CSV/JSON inputs may provide a configurable `group_id` column. Folder mode uses
blank group IDs unless a sidecar label file supplies them.

- All non-empty group IDs are treated as indivisible split units.
- Blank group IDs fall back to one sample per split unit.
- If no non-empty group IDs exist, the current stratified algorithm remains
  unchanged.
- Group-aware allocation greedily minimizes weighted error across
  `(label, difficulty_band, split)` targets while preserving whole groups.
- The split summary reports requested/actual ratios, class distribution,
  difficulty distribution, group counts, group leakage count, duplicate-group
  leakage count, and risk level/messages.

## Runtime Instrumentation

`summary.json` gains:

- stage timings;
- actual semantic device;
- CLIP model name;
- configured batch size;
- cache hit/miss counts;
- failed embedding count and sample IDs;
- GPU peak allocated/reserved memory when CUDA is used;
- model-loaded status.

This information is generated from the real run and never synthesized.

## Validation Assets

- `scripts/prepare_cifar10.py` exports the canonical CIFAR-10 test split.
- `scripts/compare_v1_v2.py` recomputes the V1 score from V2 raw metrics and
  writes score/band comparison statistics.
- `validation/clip_cpu/`, `validation/clip_gpu/`, and
  `validation/cifar10_test/` contain real run evidence.
- `validation/before_after.md` records key metric changes.

## Acceptance Criteria

- Existing tests continue to pass after compatible changes.
- New tests prove label rarity contributes zero to the score.
- New tests prove entropy/frequency do not affect the composite score.
- New tests prove visual-only mode is explicit.
- New tests prove group IDs never span splits.
- New tests prove no-group mode retains existing behavior.
- Real CLIP CPU and GPU runs both execute the full pipeline.
- A second identical run demonstrates cache hits.
- An interrupted/new-worker style run demonstrates resume from partial cache.
- CIFAR-10 produces all required report artifacts.
- Unsupported or unverified claims are listed as limitations.
