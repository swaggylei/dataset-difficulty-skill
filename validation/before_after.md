# V1 vs V2 Key Metrics

Comparison sample: 9,206 non-anomalous CIFAR-10 test images.

| Metric | V1 | V2 |
|---|---:|---:|
| Mean difficulty score | 0.4092 | 0.5000 |
| Median difficulty score | 0.4012 | 0.4895 |
| Mean Label Rarity contribution | 0.00919 | 0.00000 |
| Mean visual contribution | 0.1000 | 0.1500 |
| Mean semantic contribution | 0.3000 | 0.3500 |

- Spearman rank correlation between V1 and V2: `0.9210`
- Difficulty-band changes: `2,116`
- Easy -> Medium: `545`
- Medium -> Easy: `545`
- Medium -> Hard: `513`
- Hard -> Medium: `513`
- No Easy/Hard direct crossings occurred.

The high rank correlation means V2 preserves the broad ordering while making
the component semantics explicit. Band changes are expected because V2 removes
Label Rarity, changes visual weights from entropy/frequency-inclusive to
accessibility-only, and renormalizes the configured visual/semantic weights.

Machine-readable details: `validation/before_after/comparison.json`.
