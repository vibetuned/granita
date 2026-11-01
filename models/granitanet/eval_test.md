# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15546 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.59%  |
| Top-3 Accuracy | 99.94%  |
| Macro F1       | 98.57%  |
| Weighted F1    | 98.59%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.34%     |
| articulation | 98.09%     |
| barline      | 97.32%     |
| clef         | 100.00%    |
| dynamic      | 99.65%     |
| fingering    | 99.25%     |
| note         | 94.96%     |
| ornament     | 99.92%     |
| pedal        | 100.00%    |
| rest         | 98.52%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      8714 | 99.69%  |
| stroke   |      1476 | 98.85%  |
| crop     |      2415 | 99.71%  |
| hand     |      1119 | 86.95%  |
| emnist   |      1822 | 98.79%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| note64th                | note32nd           |      24 |
| articStaccatoAbove      | augmentationDot    |      13 |
| note32nd                | note16th           |      12 |
| note32nd                | note64th           |      12 |
| augmentationDot         | articStaccatoAbove |      12 |
| note16th                | note8th            |      11 |
| noteHalf                | noteQuarter        |      11 |
| note16th                | note32nd           |      11 |
| articStaccatissimoAbove | barlineSingle      |      10 |
| note8th                 | note16th           |       9 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.47%     |
| fing2                   | fingering    | 98.60%     |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.73%     |
| fing5                   | fingering    | 99.46%     |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.80%     |
| arpeg                   | ornament     | 99.81%     |
| articStaccatoAbove      | articulation | 94.44%     |
| articTenutoAbove        | articulation | 99.38%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 96.84%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.66%     |
| accidentalNatural       | accidental   | 99.15%     |
| accidentalSharp         | accidental   | 98.97%     |
| accidentalDoubleSharp   | accidental   | 98.95%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 99.85%     |
| dynamicMezzo            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 99.43%     |
| dynamicZ                | dynamic      | 98.76%     |
| dynamicPP               | dynamic      | 99.86%     |
| dynamicMP               | dynamic      | 99.32%     |
| dynamicMF               | dynamic      | 99.31%     |
| dynamicFF               | dynamic      | 99.86%     |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 99.73%     |
| dynamicSforzato         | dynamic      | 99.86%     |
| pedal                   | pedal        | 100.00%    |
| noteWhole               | note         | 99.34%     |
| noteHalf                | note         | 96.67%     |
| noteQuarter             | note         | 96.85%     |
| note8th                 | note         | 96.48%     |
| note16th                | note         | 92.81%     |
| note32nd                | note         | 90.18%     |
| note64th                | note         | 92.87%     |
| augmentationDot         | note         | 94.44%     |
| restWholeHalf           | rest         | 98.97%     |
| restQuarter             | rest         | 98.38%     |
| rest8th                 | rest         | 98.79%     |
| rest16th                | rest         | 97.70%     |
| rest32nd                | rest         | 98.29%     |
| rest64th                | rest         | 98.97%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 97.32%     |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.76%  |
| Top-3 Accuracy | 99.92%  |
| Macro F1       | 97.67%  |
| Weighted F1    | 97.77%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.17%     |
| articulation | 95.22%     |
| barline      | 93.10%     |
| clef         | 100.00%    |
| dynamic      | 99.55%     |
| fingering    | 98.62%     |
| note         | 92.93%     |
| ornament     | 99.88%     |
| pedal        | 99.59%     |
| rest         | 97.66%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      8714 | 99.25%  |
| stroke   |      1476 | 97.15%  |
| crop     |      2415 | 98.84%  |
| hand     |      1119 | 83.91%  |
| emnist   |      1822 | 98.19%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      47 |
| augmentationDot         | articStaccatoAbove |      30 |
| note64th                | note32nd           |      25 |
| fing1                   | barlineSingle      |      25 |
| articStaccatissimoAbove | barlineSingle      |      15 |
| note32nd                | note64th           |      14 |
| note16th                | note32nd           |      14 |
| note32nd                | note16th           |      13 |
| note8th                 | note16th           |      11 |
| rest32nd                | rest16th           |      10 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 96.14%     |
| fing2                   | fingering    | 98.46%     |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.60%     |
| fing5                   | fingering    | 98.91%     |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.80%     |
| arpeg                   | ornament     | 99.61%     |
| articStaccatoAbove      | articulation | 81.33%     |
| articTenutoAbove        | articulation | 98.56%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 96.39%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.66%     |
| accidentalNatural       | accidental   | 98.80%     |
| accidentalSharp         | accidental   | 99.14%     |
| accidentalDoubleSharp   | accidental   | 98.23%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 99.85%     |
| dynamicMezzo            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 99.86%     |
| dynamicSforzando        | dynamic      | 98.59%     |
| dynamicZ                | dynamic      | 98.48%     |
| dynamicPP               | dynamic      | 99.86%     |
| dynamicMP               | dynamic      | 99.32%     |
| dynamicMF               | dynamic      | 99.31%     |
| dynamicFF               | dynamic      | 99.86%     |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 99.73%     |
| dynamicSforzato         | dynamic      | 99.86%     |
| pedal                   | pedal        | 99.59%     |
| noteWhole               | note         | 97.37%     |
| noteHalf                | note         | 96.85%     |
| noteQuarter             | note         | 96.53%     |
| note8th                 | note         | 96.47%     |
| note16th                | note         | 91.75%     |
| note32nd                | note         | 89.07%     |
| note64th                | note         | 92.17%     |
| augmentationDot         | note         | 83.22%     |
| restWholeHalf           | rest         | 97.75%     |
| restQuarter             | rest         | 98.21%     |
| rest8th                 | rest         | 98.61%     |
| rest16th                | rest         | 97.01%     |
| rest32nd                | rest         | 96.42%     |
| rest64th                | rest         | 97.93%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 93.10%     |
