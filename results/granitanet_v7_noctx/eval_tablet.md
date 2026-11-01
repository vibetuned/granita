# granitanet on the test split of /home/flux/projects/musicset/data/symbols-tablet/dataset.npz (15280 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.82%  |
| Top-3 Accuracy | 99.99%  |
| Macro F1       | 98.81%  |
| Weighted F1    | 98.81%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.97%     |
| articulation | 95.74%     |
| barline      | 89.92%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 98.12%     |
| note         | 97.59%     |
| ornament     | 100.00%    |
| pedal        | 100.00%    |
| rest         | 99.92%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     11731 | 99.30%  |
| stroke   |      3549 | 97.21%  |

## Top Confusions

| True                    | Predicted             |   Count |
|-------------------------|-----------------------|---------|
| articStaccatoAbove      | augmentationDot       |      54 |
| fing1                   | barlineSingle         |      50 |
| augmentationDot         | articStaccatoAbove    |      42 |
| articStaccatissimoAbove | barlineSingle         |      16 |
| articStaccatoAbove      | noteWhole             |       6 |
| augmentationDot         | noteWhole             |       4 |
| note32nd                | note64th              |       3 |
| restWholeHalf           | articTenutoAbove      |       3 |
| augmentationDot         | accidentalDoubleSharp |       1 |
| barlineSingle           | fing1                 |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 90.58%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 82.11%     |
| articTenutoAbove        | articulation | 99.50%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.08%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 100.00%    |
| accidentalSharp         | accidental   | 100.00%    |
| accidentalDoubleSharp   | accidental   | 99.83%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicMezzo            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicZ                | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 100.00%    |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 100.00%    |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 100.00%    |
| noteWhole               | note         | 98.36%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| note32nd                | note         | 99.50%     |
| note64th                | note         | 99.50%     |
| augmentationDot         | note         | 83.36%     |
| restWholeHalf           | rest         | 99.50%     |
| restQuarter             | rest         | 100.00%    |
| rest8th                 | rest         | 100.00%    |
| rest16th                | rest         | 100.00%    |
| rest32nd                | rest         | 100.00%    |
| rest64th                | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 89.92%     |
