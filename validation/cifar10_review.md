# CIFAR-10 Result Review

## Distribution

- Total images: 10,000
- Non-anomalous scored samples: 9,206
- Anomalies: 794
- Bands among scored samples: Easy 3,038; Medium 3,130; Hard 3,038
- Split ratios among scored samples: 0.69998 / 0.15001 / 0.15001
- `group_id` leakage: 0
- Duplicate-group leakage: 0
- Leakage risk: low

Anomalies were concentrated in airplane (124), ship (115), and bird (114)
images. These classes frequently contain small objects, uniform sky/water
backgrounds, or low histogram complexity that triggered statistical or
quality flags.

## Hard-Sample Inspection

The highest-ranked samples included:

- dark/low-contrast airplane scenes;
- small deer, bird, and airplane objects with limited spatial evidence;
- dog and ship images with low contrast or distant subjects;
- images whose CLIP centroid distance or semantic component rank was high
  even when absolute blur was not low.

All 3,038 Hard-band samples had empty absolute `quality_issue_flags`. This is
the intended behavior: visual accessibility contributes to difficulty through
dataset-relative evidence, but a poor-quality flag is not treated as proof of
semantic difficulty.

Hard-band mean components:

| Component | Mean |
|---|---:|
| Visual accessibility risk | 0.6148 |
| Semantic complexity | 0.6607 |
| Class centroid distance | 0.1132 |
| Class confusion risk | 0.0052 |
| Augmentation instability | 0.0069 |
| Descriptive visual complexity | 0.4400 |

The preview sheet visually supports the ranking: most top examples are small,
low-contrast, blurred, or semantically ambiguous. No direct Easy/Hard
crossings occurred in the V1/V2 band comparison; changes were adjacent-band
shifts around thresholds.

## Caveat

This review checks observable plausibility, not ground-truth difficulty.
Confirmation still requires downstream training/validation experiments.
