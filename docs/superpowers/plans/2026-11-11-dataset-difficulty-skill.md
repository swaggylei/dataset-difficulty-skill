# Dataset Difficulty Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Deliver an installable, tested Agent Skill and Python CLI that
performs training-free difficulty-prior analysis for image classification
datasets.

**Architecture:** A `src` package separates loading, quality metrics, anomaly
detection, CLIP extraction/cache, semantic metrics, scoring, splitting,
reporting, and orchestration. A thin `scripts/analyze_dataset.py` entrypoint
adds `src` to `sys.path` for in-place Skill execution. CLIP is an injectable
protocol so semantic math can be tested without model downloads.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib,
PyYAML, PyTorch, Transformers CLIP, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-dataset-difficulty-skill-design.md`

## Global Constraints

- Support image classification only.
- Never mutate input images or label files.
- Keep visual quality risk separate from semantic difficulty.
- Preserve every raw metric and every component score in output.
- Use configurable YAML rules and percentile bands; do not require clustering.
- Leave unavailable metrics null; do not fabricate semantic values.
- Default to local-only CLIP model loading.
- Keep feature caches content-addressed and resumable.
- Treat anomalies as separate from Easy/Medium/Hard.

## Review Focus

- Unicode Windows paths in image and label files must load correctly.
- CSV/JSON rows with conflicting labels must not silently choose one label.
- Missing CLIP must produce a valid report with null semantic fields.
- Tiny class/band strata must not crash joint splitting.
- Re-running with the same seed must produce identical splits.

---

### Task 1: Project Skeleton, Configuration, and Dataset Readers

**Files:**
- Create: `pyproject.toml`
- Create: `configs/default.yaml`
- Create: `src/dataset_difficulty/__init__.py`
- Create: `src/dataset_difficulty/config.py`
- Create: `src/dataset_difficulty/loading.py`
- Test: `tests/conftest.py`
- Test: `tests/test_loading.py`

**Interfaces:**
- Produces: `load_config(path: str | Path | None, overrides: dict | None = None) -> dict`
- Produces: `load_dataset(root: Path, label_file: Path | None = None, data_format: str = "auto", extensions: list[str] | None = None, path_column: str = "path", label_column: str = "label", exclude_paths: list[Path] | None = None) -> DatasetLoadResult`
- `DatasetLoadResult.samples` is a DataFrame with `sample_id`, `path`,
  `rel_path`, `label`, and `label_source`.
- `DatasetLoadResult.warnings` is a list of human-readable strings.

- [ ] **Step 1: Write failing loader and config tests**

Assert folder labels come from parent directories; CSV paths resolve relative
to the label file; JSON list and path-to-label mapping forms load; duplicate
paths with different labels keep a `conflicting_label` marker; excluded paths
are skipped; YAML overrides merge recursively.

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_loading.py -v`
Expected: FAIL because `dataset_difficulty.loading` and `config` do not exist.

- [ ] **Step 3: Implement configuration and loading**

Use `pathlib`, `json`, `yaml`, and `pandas`; normalize paths and sample IDs
with stable SHA-256 hashes; never decode image bytes in the loader.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest tests/test_loading.py -v`
Expected: PASS.

### Task 2: Image Quality and Anomaly Detection

**Files:**
- Create: `src/dataset_difficulty/quality.py`
- Create: `src/dataset_difficulty/anomalies.py`
- Test: `tests/test_quality.py`
- Test: `tests/test_anomalies.py`

**Interfaces:**
- Produces: `analyze_image_quality(path: Path, config: dict) -> dict`
- Produces: `detect_anomalies(samples: DataFrame, quality: DataFrame, config: dict) -> DataFrame`
- Quality fields include `width`, `height`, `blur_laplacian_var`,
  `brightness`, `contrast`, `spectral_entropy`, `high_freq_energy_ratio`,
  `image_entropy`, and `decode_ok`.
- Anomaly fields include `is_anomaly`, `anomaly_types`, `duplicate_group`,
  `is_duplicate_canonical`, and `quality_issue_flags`.

- [ ] **Step 1: Write failing quality and anomaly tests**

Use generated PNGs for sharp, blurred, dark, low-contrast, corrupt, exact
duplicate, near-duplicate, and tiny-image cases. Assert that raw metrics are
in expected ranges and that anomaly reason sets contain the expected codes.

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_quality.py tests/test_anomalies.py -v`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement quality and anomaly modules**

Read with `cv2.imdecode(np.fromfile(...))`. Compute spectral entropy from a
normalized FFT power distribution and histogram entropy from an 8-bit image
histogram. Use content SHA-256 for exact duplicates and a deterministic
average-hash Hamming distance for optional near duplicates. Use absolute
quality thresholds plus MAD-based statistical outliers.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest tests/test_quality.py tests/test_anomalies.py -v`
Expected: PASS.

### Task 3: CLIP Feature Extraction, Cache, and Semantic Metrics

**Files:**
- Create: `src/dataset_difficulty/features.py`
- Create: `src/dataset_difficulty/semantics.py`
- Test: `tests/test_features.py`
- Test: `tests/test_semantics.py`

**Interfaces:**
- Protocol: `FeatureExtractor.encode_images(images: Sequence[Image.Image]) -> np.ndarray`
- Produces: `ClipFeatureExtractor(model_name: str, device: str, allow_download: bool, batch_size: int)`
- Produces: `FeatureCache(cache_dir: Path, signature: str)`
- Produces: `extract_features(samples: DataFrame, extractor: FeatureExtractor, cache: FeatureCache, augmentations: int, seed: int) -> FeatureResult`
- `FeatureResult.base` and `.augmented` are arrays aligned to sample rows;
  unavailable rows are all-NaN.
- Produces: `compute_semantic_metrics(samples: DataFrame, base: np.ndarray, augmented: np.ndarray, config: dict) -> DataFrame`
- Semantic fields include `class_centroid_distance`,
  `class_confusion_risk`, `cross_class_margin`, `augmentation_stability`,
  `augmentation_instability`, `semantic_complexity`, and
  `semantic_available`.

- [ ] **Step 1: Write failing feature and semantic tests**

Use a deterministic fake extractor. Assert cache miss then hit behavior, cache
reuse across calls, NaN alignment for failed images, normalized embeddings,
centroid distance, class confusion, augmentation instability, and null
semantic scores when embeddings are missing.

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_features.py tests/test_semantics.py -v`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement extraction, caching, and semantic metrics**

CLIP uses `transformers.CLIPModel` and `CLIPImageProcessor` with
`local_files_only=not allow_download`. Cache each base and augmented vector in
an NPZ keyed by content hash and model/preprocess signature. Calculate cosine
geometry without treating vector entropy as probability.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest tests/test_features.py tests/test_semantics.py -v`
Expected: PASS.

### Task 4: Difficulty Scoring, Bands, and Joint Splits

**Files:**
- Create: `src/dataset_difficulty/scoring.py`
- Create: `src/dataset_difficulty/splitting.py`
- Test: `tests/test_scoring.py`
- Test: `tests/test_splitting.py`

**Interfaces:**
- Produces: `score_difficulty(samples: DataFrame, quality: DataFrame, semantic: DataFrame, config: dict) -> DataFrame`
- Difficulty fields include `visual_quality_risk`,
  `visual_difficulty_component`, `semantic_difficulty_component`,
  `label_rarity_score`, `difficulty_score`, `difficulty_band`,
  `component_coverage`, and `sample_status`.
- Produces: `assign_splits(scored: DataFrame, ratios: dict, seed: int, exclude_anomalies: bool = True) -> SplitResult`
- `SplitResult.assignment` adds `split`; `.summary` is JSON-serializable.

- [ ] **Step 1: Write failing scoring and split tests**

Assert orientation of each rank, correct Easy/Medium/Hard bands, anomaly
isolation, NaN semantic handling, recursive weight normalization, reproducible
joint splits, and graceful handling of tiny strata.

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_scoring.py tests/test_splitting.py -v`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement scoring and splitting**

Use `pandas.rank(pct=True, method="average", na_option="keep")` for
dataset-relative signals. Combine only available components and store
coverage. Allocate each `(label, band)` group with a seeded shuffle and
largest-remainder quotas while preserving deterministic tie-breaking.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest tests/test_scoring.py tests/test_splitting.py -v`
Expected: PASS.

### Task 5: Reports and End-to-End Pipeline

**Files:**
- Create: `src/dataset_difficulty/reports.py`
- Create: `src/dataset_difficulty/pipeline.py`
- Create: `src/dataset_difficulty/cli.py`
- Create: `src/dataset_difficulty/__main__.py`
- Create: `scripts/analyze_dataset.py`
- Test: `tests/test_reports.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `write_reports(result: AnalysisResult, output_dir: Path, config: dict) -> dict`
- Produces: `analyze_dataset(data_root: Path, output_dir: Path, config: dict, label_file: Path | None = None, extractor: FeatureExtractor | None = None, semantic_backend: str = "auto") -> AnalysisResult`
- CLI command: `python scripts/analyze_dataset.py DATA_ROOT --output OUTPUT_DIR --config configs/default.yaml`

- [ ] **Step 1: Write failing report and pipeline tests**

Assert all required artifacts exist, JSON contains no NaN tokens, CSV includes
raw and component columns, HTML contains sample rows and preview images, a
no-semantic run succeeds with null semantic fields, and a fake-extractor run
populates semantic metrics.

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_reports.py tests/test_pipeline.py -v`
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement reports and orchestration**

Write CSVs with `pandas`, JSON with a recursive NaN-to-null sanitizer,
standalone HTML with escaped values, and Matplotlib preview sheets. Pipeline
stages log progress, save semantic cache per batch, and continue on per-image
quality failures.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest tests/test_reports.py tests/test_pipeline.py -v`
Expected: PASS.

### Task 6: Skill Instructions, References, Example, and Packaging

**Files:**
- Modify: `SKILL.md`
- Create: `README.md`
- Create: `requirements.txt`
- Modify: `agents/openai.yaml`
- Create: `references/metrics.md`
- Create: `references/configuration.md`
- Create: `references/outputs.md`
- Create: `examples/generate_example_dataset.py`
- Create: `tests/test_cli_e2e.py`

**Interfaces:**
- Skill invocation instructions call the CLI above and inspect `summary.json`
  before interpreting results.
- Example generator creates read-only toy images under
  `examples/toy_dataset/` and a `labels.csv` file.

- [ ] **Step 1: Write failing CLI end-to-end test**

Run the CLI as a subprocess on a temporary dataset, assert exit code zero and
all report artifacts.

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/test_cli_e2e.py -v`
Expected: FAIL until the script and package entrypoint are complete.

- [ ] **Step 3: Write Skill instructions and generate the example**

Keep `SKILL.md` focused on trigger conditions, input checks, CLI invocation,
result interpretation, safety limits, and references. Do not duplicate all
metric formulas in the entrypoint.

- [ ] **Step 4: Run the test to verify GREEN**

Run: `python -m pytest tests/test_cli_e2e.py -v`
Expected: PASS.

### Task 7: Full Verification and Example Report

**Files:**
- Generated: `examples/output/`

- [ ] **Step 1: Install development dependencies**

Run: `python -m pip install -e ".[dev]"`
Expected: package and pytest install successfully.

- [ ] **Step 2: Generate the example dataset and run the analyzer**

Run: `python examples/generate_example_dataset.py`
Run: `python scripts/analyze_dataset.py examples/toy_dataset --labels examples/toy_dataset/labels.csv --format csv --output examples/output --no-semantic`
Expected: exit code zero with report artifacts.

- [ ] **Step 3: Run all tests and Skill validation**

Run: `python -m pytest -v`
Run: `python <skill-creator>/scripts/quick_validate.py .`
Expected: all tests pass and validation succeeds.

- [ ] **Step 4: Inspect outputs**

Open `examples/output/report.html` visually, verify images render, verify the
JSON contains no NaN/Infinity tokens, and confirm input dataset hashes or
modification times did not change during analysis.
