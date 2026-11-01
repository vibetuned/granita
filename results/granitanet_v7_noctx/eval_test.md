# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15546 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.88%  |
| Top-3 Accuracy | 99.89%  |
| Macro F1       | 97.81%  |
| Weighted F1    | 97.88%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.26%     |
| articulation | 96.37%     |
| barline      | 95.41%     |
| clef         | 100.00%    |
| dynamic      | 99.54%     |
| fingering    | 98.69%     |
| note         | 92.51%     |
| ornament     | 99.96%     |
| pedal        | 99.59%     |
| rest         | 97.92%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      8714 | 99.33%  |
| stroke   |      1476 | 97.97%  |
| crop     |      2415 | 99.21%  |
| hand     |      1119 | 82.84%  |
| emnist   |      1822 | 98.30%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| augmentationDot    | articStaccatoAbove |      33 |
| note64th           | note32nd           |      28 |
| articStaccatoAbove | augmentationDot    |      28 |
| note32nd           | note64th           |      21 |
| fing1              | barlineSingle      |      20 |
| note16th           | note32nd           |      17 |
| note32nd           | note16th           |      14 |
| noteQuarter        | noteHalf           |      13 |
| noteHalf           | noteQuarter        |      10 |
| note16th           | note8th            |       8 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 96.87%     |
| fing2                   | fingering    | 98.32%     |
| fing3                   | fingering    | 99.86%     |
| fing4                   | fingering    | 99.60%     |
| fing5                   | fingering    | 98.77%     |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.80%     |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 85.35%     |
| articTenutoAbove        | articulation | 98.37%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.33%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.32%     |
| accidentalNatural       | accidental   | 99.15%     |
| accidentalSharp         | accidental   | 99.48%     |
| accidentalDoubleSharp   | accidental   | 98.58%     |
| accidentalDoubleFlat    | accidental   | 99.78%     |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicMezzo            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 99.57%     |
| dynamicSforzando        | dynamic      | 98.73%     |
| dynamicZ                | dynamic      | 98.20%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 99.32%     |
| dynamicMF               | dynamic      | 99.31%     |
| dynamicFF               | dynamic      | 99.86%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.73%     |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 99.59%     |
| noteWhole               | note         | 97.04%     |
| noteHalf                | note         | 96.38%     |
| noteQuarter             | note         | 95.63%     |
| note8th                 | note         | 96.62%     |
| note16th                | note         | 92.33%     |
| note32nd                | note         | 86.77%     |
| note64th                | note         | 90.13%     |
| augmentationDot         | note         | 85.22%     |
| restWholeHalf           | rest         | 97.57%     |
| restQuarter             | rest         | 98.55%     |
| rest8th                 | rest         | 98.28%     |
| rest16th                | rest         | 97.67%     |
| rest32nd                | rest         | 97.29%     |
| rest64th                | rest         | 98.14%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 95.41%     |
