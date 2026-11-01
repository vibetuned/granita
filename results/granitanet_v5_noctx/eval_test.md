# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15873 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.88%  |
| Top-3 Accuracy | 99.93%  |
| Macro F1       | 97.99%  |
| Weighted F1    | 97.88%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 98.86%     |
| articulation | 96.68%     |
| barline      | 95.24%     |
| clef         | 99.71%     |
| dynamic      | 99.98%     |
| fingering    | 99.14%     |
| note         | 92.62%     |
| ornament     | 99.80%     |
| pedal        | 100.00%    |
| rest         | 96.65%     |
| timesig      | 99.00%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.44%  |
| stroke   |      1476 | 97.70%  |
| crop     |      2164 | 99.54%  |
| hand     |      1520 | 84.67%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| augmentationDot    | articStaccatoAbove |      38 |
| articStaccatoAbove | augmentationDot    |      29 |
| fing1              | barlineSingle      |      21 |
| note64th           | note32nd           |      20 |
| note16th           | note32nd           |      18 |
| noteQuarter        | noteHalf           |      14 |
| note32nd           | note64th           |      14 |
| restQuarter        | accidentalNatural  |      13 |
| noteWhole          | augmentationDot    |      13 |
| noteHalf           | noteQuarter        |      12 |
| rest64th           | rest32nd           |      12 |
| note32nd           | note16th           |      12 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 95.71%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.60%     |
| arpeg                   | ornament     | 99.39%     |
| articStaccatoAbove      | articulation | 86.11%     |
| articTenutoAbove        | articulation | 99.18%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.12%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 99.49%     |
| accidentalNatural       | accidental   | 96.35%     |
| accidentalSharp         | accidental   | 98.64%     |
| accidentalDoubleSharp   | accidental   | 99.82%     |
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
| noteWhole               | note         | 95.93%     |
| noteHalf                | note         | 95.26%     |
| noteQuarter             | note         | 94.83%     |
| note8th                 | note         | 96.45%     |
| note16th                | note         | 91.97%     |
| note32nd                | note         | 88.27%     |
| note64th                | note         | 93.53%     |
| augmentationDot         | note         | 84.75%     |
| restWholeHalf           | rest         | 98.18%     |
| restQuarter             | rest         | 95.73%     |
| rest8th                 | rest         | 97.55%     |
| rest16th                | rest         | 96.47%     |
| rest32nd                | rest         | 94.94%     |
| rest64th                | rest         | 97.03%     |
| gClef                   | clef         | 99.82%     |
| fClef                   | clef         | 99.31%     |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 99.16%     |
| timeSigCutCommon        | timesig      | 98.95%     |
| timeSig2over2           | timesig      | 98.76%     |
| timeSig2over4           | timesig      | 98.14%     |
| timeSig3over4           | timesig      | 98.74%     |
| timeSig3over8           | timesig      | 98.54%     |
| timeSig4over4           | timesig      | 99.58%     |
| timeSig6over8           | timesig      | 99.79%     |
| timeSig9over8           | timesig      | 98.53%     |
| timeSig12over8          | timesig      | 99.79%     |
| barlineSingle           | barline      | 95.24%     |
