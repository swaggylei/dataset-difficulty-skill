# Dataset Difficulty Skill Design

## Intent

Build a model-independent, training-free prior analysis tool for image
classification datasets. The tool helps a user find anomalous samples,
compare relative sample difficulty, and create reproducible
class-and-difficulty-stratified splits before any downstream model is
trained.

## Goals

- Read folder-per-class, CSV, and JSON classification datasets.
- Preserve raw image-quality, semantic, label, anomaly, and split signals.
- Produce a transparent weighted difficulty score from separately reported
  components.
- Keep anomalies out of Easy/Medium/Hard difficulty bands.
- Support resumable CLIP feature extraction through a content-addressed cache.
- Generate machine-readable and human-reviewable reports without mutating the
  input dataset.
- Expose a deterministic CLI that an Agent Skill can invoke.

## Non-Goals

- Object detection, segmentation, or video analysis.
- Fine-tuning or using downstream model losses.
- Treating poor image quality as proof that a semantic category is difficult.
- Claiming that difficulty scores are calibrated accuracy predictions.
- Downloading a CLIP model without explicit user authorization.

## Constraints

- Target Python 3.11; the local verification environment may run a compatible
  newer minor version when 3.11 is unavailable.
- OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, PyYAML, PyTorch, and a
  CLIP-capable Transformers runtime are supported.
- Every weight and threshold is configurable through YAML.
- Missing metrics remain null/NaN and are excluded from score aggregation.
- Original images and labels are read-only. All generated files go to an
  explicit output or cache directory.
- Semantic extraction defaults to local-only model resolution.

## Metric Model

Image quality records blur (Laplacian variance), normalized brightness,
contrast, spectral entropy, high-frequency energy ratio, and grayscale
histogram entropy. Absolute quality thresholds identify quality anomalies.
Relative quality ranks contribute a bounded visual component to the difficulty
prior; quality risk is never the sole reason a sample is labeled Hard.

CLIP semantic records include cosine distance to the class centroid,
cross-class confusion risk, augmentation instability, and their observable
inputs. Semantic component scores are quantile ranks within the dataset.
Embeddings are L2-normalized. No vector entropy is treated as a probability.

Label accessibility records class rarity. Empty or conflicting labels are
anomalies and do not receive a difficulty score.

The final score is a weighted sum of available component scores. Missing
components are reported, and the available weight is stored with each row.
Default band thresholds are the 33rd and 67th percentiles of scored,
non-anomalous samples.

## Anomaly Model

Each sample has a boolean anomaly status and a semicolon-separated set of
reason codes. The initial reason codes are:

- `corrupt_image`
- `empty_label`
- `conflicting_label`
- `exact_duplicate`
- `near_duplicate`
- `insufficient_resolution`
- `statistical_outlier`
- `quality_outlier`

Anomalies remain in the per-sample report but are excluded from split
assignment by default. Duplicate groups are recorded for review.

## Split Model

Non-anomalous samples are grouped by `(label, difficulty_band)`. A seeded,
deterministic allocator distributes each group across Train/Val/Test to
approximate configured ratios. Groups with too few samples are recorded as
fallbacks. The split summary reports exact counts by class, band, and split.

## Output Contract

The output directory contains:

- `samples.csv`: all raw metrics, components, anomaly reasons, and split.
- `summary.json`: configuration, counts, warnings, and artifact list.
- `report.html`: standalone review surface with tables and image previews.
- `hard_samples.png`: a preview sheet of the highest-difficulty samples.
- `anomaly_samples.png`: a preview sheet of anomalous samples.
- `split_manifest.csv`: the train/validation/test manifest.
- `split_summary.json`: split counts and stratification diagnostics.

The CLIP cache is separate from the report and stores one NPZ entry per image
content hash and model signature so interrupted runs can resume.

## Safety and Interpretation

The Skill must tell users that scores are relative training-time priors, not
accuracy predictions. Corrupt images, label conflicts, and duplicates are
data-integrity findings and should be handled before interpreting difficulty.
Reports must disclose when semantic features were unavailable or supplied by
a precomputed backend.
