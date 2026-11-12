# CIFAR-10 Result Review

## Distribution

- Total images: 10,000
- Scored samples: 10,000 (9,206 ordinary + 420 statistical-only + 374
  quality-flagged soft anomalies)
- Anomalies: 794
- Bands among scored samples: Easy 3,300; Medium 3,400; Hard 3,300
- Hard-excluded anomalies: 0
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

Of the 3,300 Hard-band samples, 301 have readable quality flags and remain in
the split as quality soft anomalies. This is intentional: quality flags remain
visible and contribute through visual accessibility, but low quality is not
treated as proof of semantic difficulty and is not automatically excluded.

Hard-band mean components:

| Component | Mean |
|---|---:|
| Visual accessibility risk | 0.6310 |
| Semantic complexity | 0.6672 |
| Class centroid distance | 0.1141 |
| Class confusion risk | 0.0054 |
| Augmentation instability | 0.0070 |
| Descriptive visual complexity | 0.4430 |

The preview sheet visually supports the ranking: most top examples are small,
low-contrast, blurred, or semantically ambiguous. The V1/V2 comparison now
includes soft anomalies: 334 Easy samples moved directly to Hard and 2,634
bands changed overall, so the comparison is descriptive rather than a claim
that the old and new band definitions are equivalent.

## Caveat

This review checks observable plausibility, not ground-truth difficulty.
Confirmation still requires downstream training/validation experiments.
