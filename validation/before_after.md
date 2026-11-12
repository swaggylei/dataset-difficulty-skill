# V1 vs V2 Key Metrics

Comparison sample: all 10,000 CIFAR-10 test images. The current V2 output
retains 420 statistical-only and 374 readable quality-flagged soft anomalies.

| Metric | V1 | V2 |
|---|---:|---:|
| Mean difficulty score | 0.3874 | 0.5024 |
| Median difficulty score | 0.3809 | 0.4903 |
| Mean Label Rarity contribution | 0.00000 | 0.00000 |
| Mean visual contribution | 0.1112 | 0.1802 |
| Mean semantic contribution | 0.2762 | 0.3222 |

- Spearman rank correlation between V1 and V2: `0.7851`
- Difficulty-band changes: `2,634`
- Easy -> Medium: `407`
- Medium -> Easy: `741`
- Medium -> Hard: `409`
- Hard -> Medium: `743`
- Easy -> Hard: `334`

The high rank correlation means V2 preserves the broad ordering while making
the component semantics explicit. Band changes are expected because V2 removes
Label Rarity, changes visual weights from entropy/frequency-inclusive to
accessibility-only, and renormalizes the configured visual/semantic weights.

Machine-readable details: `validation/before_after/comparison.json`.
