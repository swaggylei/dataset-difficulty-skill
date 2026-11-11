# Metric Reference

## Image Quality

- `blur_laplacian_var`: variance of the Laplacian on grayscale pixels. Lower
  values generally indicate less local detail.
- `brightness`: grayscale mean divided by 255, range 0-1.
- `contrast`: grayscale standard deviation divided by 255.
- `spectral_entropy`: normalized entropy of the shifted FFT power
  distribution after subtracting the mean. Range 0-1.
- `high_freq_energy_ratio`: FFT power outside the central 10% radius divided
  by total power.
- `image_entropy`: Shannon entropy of the 256-bin grayscale histogram, in
  bits (0-8).

Absolute thresholds produce `quality_issue_flags`. A quality issue is an
integrity review signal, not proof of semantic difficulty.

## Visual Accessibility vs Descriptive Complexity

`visual_accessibility_risk` uses percentile ranks:

- lower blur value -> higher blur-risk component;
- brightness farther from 0.5 -> higher risk;
- lower contrast -> higher risk.

`visual_complexity_score` separately summarizes percentile ranks of image
entropy, spectral entropy, and high-frequency energy. It is descriptive only:
higher values are not assumed to imply higher task difficulty and have no
weight in the composite score.

## Label Accessibility

`label_rarity_score` is `1 - class_count / largest_class_count` for non-empty
labels. It measures relative underrepresentation in this dataset.

Empty and conflicting labels are anomalies. They are not treated as ordinary
rare classes.

`label_rarity_weight=0`, `label_rarity_contribution=0`, and
`label_rarity_in_difficulty=False`. Label rarity does not alter
`difficulty_score`.

## CLIP Semantic Metrics

All embeddings are L2-normalized.

- `class_centroid_distance`: `1 - cosine(image, class_centroid)`. It remains
  null when the class has fewer than `semantic.min_class_samples`.
- `cross_class_margin`: own-class centroid similarity minus the largest
  other-class centroid similarity.
- `class_confusion_risk`: `max(0, -cross_class_margin)`.
- `augmentation_stability`: mean cosine similarity between the base embedding
  and deterministic augmented variants.
- `augmentation_instability`: `1 - augmentation_stability`.

The semantic component percentile-ranks centroid distance, confusion risk,
and instability, then applies the configured semantic weights. Vector entropy
is not used because CLIP dimensions do not form a probability distribution.

## Composite Difficulty

Available components:

- visual accessibility risk;
- semantic complexity;

Configured (`difficulty_*_weight`), effective normalized
(`difficulty_*_effective_weight`), and contribution
(`visual_contribution`, `semantic_contribution`) values remain in
`samples.csv`. Contributions sum to `difficulty_score` for scored samples.

`difficulty_scope` is `visual_semantic` or `visual_only` for scored samples.
Hard anomalies receive a null `difficulty_score` and `difficulty_band=Anomaly`.
Statistical-only anomalies are retained as `Statistically_Unusual` and are
scored by default.

Default bands use the 33rd and 67th percentiles of scored samples:
Easy, Medium, Hard. Percentile thresholds are dataset-relative.

## Anomaly Reasons

- `corrupt_image`
- `empty_label`
- `conflicting_label`
- `exact_duplicate`
- `near_duplicate`
- `insufficient_resolution`
- `quality_outlier`
- `statistical_outlier`

Exact duplicates use content SHA-256. Near duplicates use an 8x8 average-hash
Hamming distance, capped by `anomaly.near_duplicate.max_samples`.
Statistical outliers use per-label median/MAD robust z-scores.

By default, `split.include_statistical_only=true`: pure statistical outliers
are soft anomalies, remain visible in anomaly reports, and enter the joint
split. Set it to `false` to exclude them.

## Limits

- CLIP geometry is model-dependent.
- Dataset-relative ranks should not be compared across unrelated datasets.
- Severe class imbalance and tiny strata can make split ratios approximate.
- Image quality, visual complexity, and semantic difficulty are distinct and
  are not interchangeable.
- No metric in this Skill is a calibrated accuracy prediction.
