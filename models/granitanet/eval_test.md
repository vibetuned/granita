# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (16624 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.68%  |
| Top-3 Accuracy | 99.90%  |
| Macro F1       | 98.79%  |
| Weighted F1    | 98.68%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.55%     |
| articulation | 98.56%     |
| barline      | 98.44%     |
| clef         | 99.94%     |
| dynamic      | 99.97%     |
| fingering    | 99.69%     |
| note         | 94.51%     |
| ornament     | 99.89%     |
| pedal        | 99.80%     |
| rest         | 98.38%     |
| timesig      | 99.36%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.79%  |
| stroke   |      1476 | 99.05%  |
| crop     |      2915 | 99.49%  |
| hand     |      1520 | 88.95%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| note64th           | note32nd           |      33 |
| note16th           | note32nd           |      22 |
| noteQuarter        | noteHalf           |      16 |
| augmentationDot    | articStaccatoAbove |      14 |
| note16th           | note8th            |      12 |
| articStaccatoAbove | augmentationDot    |       9 |
| note32nd           | note16th           |       8 |
| note64th           | note16th           |       7 |
| noteHalf           | noteQuarter        |       6 |
| rest32nd           | rest64th           |       6 |
| rest32nd           | rest16th           |       6 |
| restQuarter        | accidentalNatural  |       5 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.67%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.80%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.43%     |
| articStaccatoAbove      | articulation | 95.00%     |
| articTenutoAbove        | articulation | 99.58%     |
| articAccentAbove        | articulation | 99.81%     |
| articStaccatissimoAbove | articulation | 98.60%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 98.64%     |
| accidentalSharp         | accidental   | 99.31%     |
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
| dynamicForzando         | dynamic      | 99.80%     |
| dynamicSforzato         | dynamic      | 99.80%     |
| pedal                   | pedal        | 99.80%     |
| noteWhole               | note         | 99.31%     |
| noteHalf                | note         | 96.54%     |
| noteQuarter             | note         | 95.82%     |
| note8th                 | note         | 97.54%     |
| note16th                | note         | 91.38%     |
| note32nd                | note         | 89.12%     |
| note64th                | note         | 90.91%     |
| augmentationDot         | note         | 95.47%     |
| restWholeHalf           | rest         | 99.32%     |
| restQuarter             | rest         | 97.89%     |
| rest8th                 | rest         | 99.12%     |
| rest16th                | rest         | 98.08%     |
| rest32nd                | rest         | 97.08%     |
| rest64th                | rest         | 98.77%     |
| gClef                   | clef         | 99.83%     |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 99.65%     |
| timeSigCutCommon        | timesig      | 99.83%     |
| timeSig2over2           | timesig      | 99.46%     |
| timeSig2over4           | timesig      | 98.91%     |
| timeSig3over4           | timesig      | 99.47%     |
| timeSig3over8           | timesig      | 98.33%     |
| timeSig4over4           | timesig      | 99.64%     |
| timeSig6over8           | timesig      | 99.66%     |
| timeSig9over8           | timesig      | 98.81%     |
| timeSig12over8          | timesig      | 99.82%     |
| barlineSingle           | barline      | 98.44%     |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.94%  |
| Top-3 Accuracy | 99.87%  |
| Macro F1       | 98.05%  |
| Weighted F1    | 97.94%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.27%     |
| articulation | 96.11%     |
| barline      | 94.04%     |
| clef         | 99.89%     |
| dynamic      | 99.97%     |
| fingering    | 99.06%     |
| note         | 92.43%     |
| ornament     | 99.77%     |
| pedal        | 99.80%     |
| rest         | 97.53%     |
| timesig      | 99.26%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.51%  |
| stroke   |      1476 | 96.61%  |
| crop     |      2915 | 99.07%  |
| hand     |      1520 | 86.05%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      45 |
| note64th                | note32nd           |      38 |
| augmentationDot         | articStaccatoAbove |      24 |
| note16th                | note32nd           |      24 |
| fing1                   | barlineSingle      |      21 |
| noteQuarter             | noteHalf           |      19 |
| note32nd                | note16th           |      14 |
| note16th                | note8th            |      12 |
| note32nd                | note64th           |      11 |
| articStaccatissimoAbove | barlineSingle      |      11 |
| rest32nd                | rest64th           |       9 |
| noteHalf                | noteQuarter        |       8 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 95.69%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.60%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 98.85%     |
| articStaccatoAbove      | articulation | 83.81%     |
| articTenutoAbove        | articulation | 99.37%     |
| articAccentAbove        | articulation | 99.81%     |
| articStaccatissimoAbove | articulation | 97.77%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 97.81%     |
| accidentalSharp         | accidental   | 98.95%     |
| accidentalDoubleSharp   | accidental   | 100.00%    |
| accidentalDoubleFlat    | accidental   | 99.78%     |
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
| dynamicForzando         | dynamic      | 99.80%     |
| dynamicSforzato         | dynamic      | 99.80%     |
| pedal                   | pedal        | 99.80%     |
| noteWhole               | note         | 98.11%     |
| noteHalf                | note         | 95.76%     |
| noteQuarter             | note         | 95.22%     |
| note8th                 | note         | 97.11%     |
| note16th                | note         | 90.20%     |
| note32nd                | note         | 86.11%     |
| note64th                | note         | 89.60%     |
| augmentationDot         | note         | 87.29%     |
| restWholeHalf           | rest         | 99.14%     |
| restQuarter             | rest         | 96.99%     |
| rest8th                 | rest         | 98.94%     |
| rest16th                | rest         | 97.20%     |
| rest32nd                | rest         | 95.37%     |
| rest64th                | rest         | 97.53%     |
| gClef                   | clef         | 99.83%     |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 99.83%     |
| timeSigCommon           | timesig      | 99.65%     |
| timeSigCutCommon        | timesig      | 99.83%     |
| timeSig2over2           | timesig      | 99.28%     |
| timeSig2over4           | timesig      | 98.91%     |
| timeSig3over4           | timesig      | 99.30%     |
| timeSig3over8           | timesig      | 97.84%     |
| timeSig4over4           | timesig      | 99.64%     |
| timeSig6over8           | timesig      | 99.50%     |
| timeSig9over8           | timesig      | 98.81%     |
| timeSig12over8          | timesig      | 99.82%     |
| barlineSingle           | barline      | 94.04%     |
