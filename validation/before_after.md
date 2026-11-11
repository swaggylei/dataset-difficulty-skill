# V1 vs V2 Key Metrics

Comparison sample: 9,626 scored CIFAR-10 test images, including 420
statistical-only soft anomalies.

| Metric | V1 | V2 |
|---|---:|---:|
| Mean difficulty score | 0.3986 | 0.4962 |
| Median difficulty score | 0.3929 | 0.4883 |
| Mean Label Rarity contribution | 0.00649 | 0.00000 |
| Mean visual contribution | 0.1052 | 0.1614 |
| Mean semantic contribution | 0.2869 | 0.3348 |

- Spearman rank correlation between V1 and V2: `0.8905`
- Difficulty-band changes: `2,149`
- Easy -> Medium: `477`
- Medium -> Easy: `566`
- Medium -> Hard: `464`
- Hard -> Medium: `553`
- Easy -> Hard: `89`

The high rank correlation means V2 preserves the broad ordering while making
the component semantics explicit. Band changes are expected because V2 removes
Label Rarity, changes visual weights from entropy/frequency-inclusive to
accessibility-only, and renormalizes the configured visual/semantic weights.

Machine-readable details: `validation/before_after/comparison.json`.
