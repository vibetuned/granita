# granitanet on the test split of /home/flux/projects/musicset/data/symbols-tablet/dataset.npz (15280 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.74%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 98.74%  |
| Weighted F1    | 98.74%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.90%     |
| articulation | 95.37%     |
| barline      | 89.66%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 98.11%     |
| note         | 97.48%     |
| ornament     | 99.97%     |
| pedal        | 100.00%    |
| rest         | 99.89%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     11731 | 99.28%  |
| stroke   |      3549 | 96.96%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      60 |
| fing1                   | barlineSingle      |      50 |
| augmentationDot         | articStaccatoAbove |      44 |
| articStaccatissimoAbove | barlineSingle      |      17 |
| augmentationDot         | noteWhole          |       6 |
| articStaccatoAbove      | noteWhole          |       5 |
| restWholeHalf           | articTenutoAbove   |       4 |
| barlineSingle           | fing1              |       1 |
| arpeg                   | barlineSingle      |       1 |
| accidentalNatural       | accidentalSharp    |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 90.55%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.83%     |
| articStaccatoAbove      | articulation | 80.78%     |
| articTenutoAbove        | articulation | 99.34%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 96.74%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 99.83%     |
| accidentalSharp         | accidental   | 99.83%     |
| accidentalDoubleSharp   | accidental   | 100.00%    |
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
| noteWhole               | note         | 98.20%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| note32nd                | note         | 99.83%     |
| note64th                | note         | 99.83%     |
| augmentationDot         | note         | 81.97%     |
| restWholeHalf           | rest         | 99.33%     |
| restQuarter             | rest         | 100.00%    |
| rest8th                 | rest         | 100.00%    |
| rest16th                | rest         | 100.00%    |
| rest32nd                | rest         | 100.00%    |
| rest64th                | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 89.66%     |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.74%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 98.74%  |
| Weighted F1    | 98.74%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.90%     |
| articulation | 95.37%     |
| barline      | 89.66%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 98.11%     |
| note         | 97.48%     |
| ornament     | 99.97%     |
| pedal        | 100.00%    |
| rest         | 99.89%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     11731 | 99.28%  |
| stroke   |      3549 | 96.96%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      60 |
| fing1                   | barlineSingle      |      50 |
| augmentationDot         | articStaccatoAbove |      44 |
| articStaccatissimoAbove | barlineSingle      |      17 |
| augmentationDot         | noteWhole          |       6 |
| articStaccatoAbove      | noteWhole          |       5 |
| restWholeHalf           | articTenutoAbove   |       4 |
| barlineSingle           | fing1              |       1 |
| arpeg                   | barlineSingle      |       1 |
| accidentalNatural       | accidentalSharp    |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 90.55%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.83%     |
| articStaccatoAbove      | articulation | 80.78%     |
| articTenutoAbove        | articulation | 99.34%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 96.74%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 99.83%     |
| accidentalSharp         | accidental   | 99.83%     |
| accidentalDoubleSharp   | accidental   | 100.00%    |
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
| noteWhole               | note         | 98.20%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| note32nd                | note         | 99.83%     |
| note64th                | note         | 99.83%     |
| augmentationDot         | note         | 81.97%     |
| restWholeHalf           | rest         | 99.33%     |
| restQuarter             | rest         | 100.00%    |
| rest8th                 | rest         | 100.00%    |
| rest16th                | rest         | 100.00%    |
| rest32nd                | rest         | 100.00%    |
| rest64th                | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 89.66%     |
