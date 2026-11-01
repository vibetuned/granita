# Stylus-only set (symbols-tablet, 9889 samples) per arm

## tiny
## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.04%  |
| Top-3 Accuracy | 99.99%  |
| Macro F1       | 99.04%  |
| Weighted F1    | 99.04%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.54%     |
| articulation | 97.93%     |
| dynamic      | 99.85%     |
| fingering    | 97.80%     |
| ornament     | 98.93%     |
| pedal        | 99.16%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      6340 | 99.53%  |
| stroke   |      3549 | 98.17%  |

## Top Confusions

| True              | Predicted               |   Count |
|-------------------|-------------------------|---------|
| fing1             | articStaccatissimoAbove |      47 |
| fermata           | turn                    |       8 |
| mordent           | articTenutoAbove        |       4 |
| accidentalNatural | accidentalSharp         |       3 |
| pedal             | accidentalSharp         |       3 |
| dynamicFFF        | dynamicFF               |       3 |
| dynamicSforzando  | fing3                   |       3 |
| mordent           | turn                    |       2 |


## granitanet
## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.81%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.81%  |
| Weighted F1    | 99.81%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 100.00%    |
| articulation | 99.41%     |
| dynamic      | 100.00%    |
| fingering    | 99.35%     |
| ornament     | 99.97%     |
| pedal        | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      6340 | 99.95%  |
| stroke   |      3549 | 99.55%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| fing1                   | articStaccatissimoAbove |      16 |
| articStaccatissimoAbove | fing1                   |       2 |
| arpeg                   | fing1                   |       1 |


## regla
## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.72%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.72%  |
| Weighted F1    | 99.72%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 100.00%    |
| articulation | 99.13%     |
| dynamic      | 100.00%    |
| fingering    | 99.07%     |
| ornament     | 99.93%     |
| pedal        | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      6340 | 99.94%  |
| stroke   |      3549 | 99.32%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| fing1                   | articStaccatissimoAbove |      23 |
| articStaccatissimoAbove | fing1                   |       4 |
| mordent                 | trill                   |       1 |


## iformer
## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.58%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.57%  |
| Weighted F1    | 99.57%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 100.00%    |
| articulation | 98.81%     |
| dynamic      | 100.00%    |
| fingering    | 98.45%     |
| ornament     | 99.93%     |
| pedal        | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      6340 | 99.94%  |
| stroke   |      3549 | 98.93%  |

## Top Confusions

| True   | Predicted               |   Count |
|--------|-------------------------|---------|
| fing1  | articStaccatissimoAbove |      38 |
| arpeg  | fing1                   |       2 |
| fing2  | fing3                   |       1 |
| fing5  | fing3                   |       1 |


## lsnet
## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.70%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.70%  |
| Weighted F1    | 99.70%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.93%     |
| articulation | 99.12%     |
| dynamic      | 100.00%    |
| fingering    | 99.08%     |
| ornament     | 99.87%     |
| pedal        | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      6340 | 99.87%  |
| stroke   |      3549 | 99.38%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| fing1                   | articStaccatissimoAbove |      21 |
| articStaccatissimoAbove | fing1                   |       5 |
| arpeg                   | articStaccatissimoAbove |       1 |
| arpeg                   | fing1                   |       1 |
| trill                   | mordent                 |       1 |
| accidentalNatural       | accidentalSharp         |       1 |

