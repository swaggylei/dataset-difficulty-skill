# CIFAR-10 Result Review

## Distribution

- Total images: 10,000
- Scored samples: 9,626 (9,206 ordinary + 420 statistical-only soft anomalies)
- Anomalies: 794
- Bands among scored samples: Easy 3,177; Medium 3,272; Hard 3,177
- Hard-excluded anomalies: 374
- Split ratios among scored samples: 0.69998 / 0.15001 / 0.15001
- `group_id` leakage: 0
- Duplicate-group leakage: 0
- Leakage risk: low

Anomalies were concentrated in airplane (124), ship (115), and bird (114)
images. These classes frequently contain small objects, uniform sky/water
backgrounds, or low histogram complexity that triggered statistical or
quality flags. The anomaly audit separates 374 low-quality-but-valid samples
from 420 statistical-only soft anomalies; no corrupt/unreadable images were
found.

## Hard-Sample Inspection

The highest-ranked samples included:

- dark/low-contrast airplane scenes;
- small deer, bird, and airplane objects with limited spatial evidence;
- dog and ship images with low contrast or distant subjects;
- images whose CLIP centroid distance or semantic component rank was high
  even when absolute blur was not low.

All 3,177 Hard-band samples had empty absolute `quality_issue_flags`. This is
the intended behavior: visual accessibility contributes to difficulty through
dataset-relative evidence, but a poor-quality flag is not treated as proof of
semantic difficulty.

Hard-band mean components:

| Component | Mean |
|---|---:|
| Visual accessibility risk | 0.6210 |
| Semantic complexity | 0.6603 |
| Class centroid distance | 0.1132 |
| Class confusion risk | 0.0051 |
| Augmentation instability | 0.0069 |
| Descriptive visual complexity | 0.4404 |

The preview sheet visually supports the ranking: most top examples are small,
low-contrast, blurred, or semantically ambiguous. The V1/V2 comparison now
includes soft anomalies: 89 Easy samples moved directly to Hard and 2,149
bands changed overall, so the comparison is descriptive rather than a claim
that the old and new band definitions are equivalent.

## Caveat

This review checks observable plausibility, not ground-truth difficulty.
Confirmation still requires downstream training/validation experiments.
