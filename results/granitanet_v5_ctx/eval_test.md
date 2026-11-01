# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15873 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.54%  |
| Top-3 Accuracy | 99.93%  |
| Macro F1       | 98.62%  |
| Weighted F1    | 98.55%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.17%     |
| articulation | 98.59%     |
| barline      | 98.31%     |
| clef         | 99.83%     |
| dynamic      | 99.98%     |
| fingering    | 99.56%     |
| note         | 93.88%     |
| ornament     | 99.92%     |
| pedal        | 100.00%    |
| rest         | 98.22%     |
| timesig      | 99.17%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.83%  |
| stroke   |      1476 | 98.64%  |
| crop     |      2164 | 99.77%  |
| hand     |      1520 | 87.63%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| note64th           | note32nd           |      22 |
| note32nd           | note64th           |      19 |
| note16th           | note32nd           |      17 |
| note32nd           | note16th           |      15 |
| articStaccatoAbove | augmentationDot    |      14 |
| noteQuarter        | noteHalf           |      14 |
| noteHalf           | noteQuarter        |      13 |
| augmentationDot    | articStaccatoAbove |       9 |
| fing1              | barlineSingle      |       7 |
| restQuarter        | accidentalNatural  |       7 |
| noteWhole          | augmentationDot    |       6 |
| rest64th           | rest32nd           |       6 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 97.81%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.60%     |
| articStaccatoAbove      | articulation | 94.99%     |
| articTenutoAbove        | articulation | 99.39%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.76%     |
| articMarcatoAbove       | articulation | 99.81%     |
| accidentalFlat          | accidental   | 99.49%     |
| accidentalNatural       | accidental   | 97.90%     |
| accidentalSharp         | accidental   | 98.99%     |
| accidentalDoubleSharp   | accidental   | 99.47%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicMezzo            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicZ                | dynamic      | 99.80%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 100.00%    |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 100.00%    |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 100.00%    |
| noteWhole               | note         | 98.24%     |
| noteHalf                | note         | 95.06%     |
| noteQuarter             | note         | 94.83%     |
| note8th                 | note         | 97.30%     |
| note16th                | note         | 91.92%     |
| note32nd                | note         | 86.93%     |
| note64th                | note         | 92.31%     |
| augmentationDot         | note         | 94.49%     |
| restWholeHalf           | rest         | 99.09%     |
| restQuarter             | rest         | 97.97%     |
| rest8th                 | rest         | 98.24%     |
| rest16th                | rest         | 98.15%     |
| rest32nd                | rest         | 97.36%     |
| rest64th                | rest         | 98.53%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 99.48%     |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 99.58%     |
| timeSigCutCommon        | timesig      | 99.79%     |
| timeSig2over2           | timesig      | 98.96%     |
| timeSig2over4           | timesig      | 98.35%     |
| timeSig3over4           | timesig      | 98.74%     |
| timeSig3over8           | timesig      | 98.13%     |
| timeSig4over4           | timesig      | 99.58%     |
| timeSig6over8           | timesig      | 99.79%     |
| timeSig9over8           | timesig      | 98.74%     |
| timeSig12over8          | timesig      | 100.00%    |
| barlineSingle           | barline      | 98.31%     |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.81%  |
| Top-3 Accuracy | 99.89%  |
| Macro F1       | 97.92%  |
| Weighted F1    | 97.81%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 98.86%     |
| articulation | 96.28%     |
| barline      | 94.43%     |
| clef         | 99.71%     |
| dynamic      | 99.98%     |
| fingering    | 99.06%     |
| note         | 92.33%     |
| ornament     | 99.88%     |
| pedal        | 100.00%    |
| rest         | 96.65%     |
| timesig      | 99.11%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.49%  |
| stroke   |      1476 | 97.02%  |
| crop     |      2164 | 99.31%  |
| hand     |      1520 | 84.61%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| articStaccatoAbove | augmentationDot    |      38 |
| augmentationDot    | articStaccatoAbove |      29 |
| note64th           | note32nd           |      22 |
| fing1              | barlineSingle      |      20 |
| note32nd           | note64th           |      19 |
| note16th           | note32nd           |      17 |
| note32nd           | note16th           |      15 |
| noteWhole          | augmentationDot    |      14 |
| noteQuarter        | noteHalf           |      14 |
| rest64th           | rest32nd           |      13 |
| noteHalf           | noteQuarter        |      13 |
| restQuarter        | accidentalNatural  |      13 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 95.32%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.39%     |
| articStaccatoAbove      | articulation | 85.13%     |
| articTenutoAbove        | articulation | 99.18%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.27%     |
| articMarcatoAbove       | articulation | 99.81%     |
| accidentalFlat          | accidental   | 99.49%     |
| accidentalNatural       | accidental   | 96.36%     |
| accidentalSharp         | accidental   | 98.99%     |
| accidentalDoubleSharp   | accidental   | 99.47%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicMezzo            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicZ                | dynamic      | 99.80%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 100.00%    |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 100.00%    |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 100.00%    |
| noteWhole               | note         | 95.58%     |
| noteHalf                | note         | 95.06%     |
| noteQuarter             | note         | 94.66%     |
| note8th                 | note         | 97.12%     |
| note16th                | note         | 91.92%     |
| note32nd                | note         | 86.93%     |
| note64th                | note         | 92.31%     |
| augmentationDot         | note         | 85.06%     |
| restWholeHalf           | rest         | 98.18%     |
| restQuarter             | rest         | 96.73%     |
| rest8th                 | rest         | 97.89%     |
| rest16th                | rest         | 96.51%     |
| rest32nd                | rest         | 94.20%     |
| rest64th                | rest         | 96.39%     |
| gClef                   | clef         | 99.82%     |
| fClef                   | clef         | 99.31%     |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 99.58%     |
| timeSigCutCommon        | timesig      | 99.79%     |
| timeSig2over2           | timesig      | 98.55%     |
| timeSig2over4           | timesig      | 98.35%     |
| timeSig3over4           | timesig      | 98.74%     |
| timeSig3over8           | timesig      | 97.93%     |
| timeSig4over4           | timesig      | 99.58%     |
| timeSig6over8           | timesig      | 99.79%     |
| timeSig9over8           | timesig      | 98.74%     |
| timeSig12over8          | timesig      | 100.00%    |
| barlineSingle           | barline      | 94.43%     |
