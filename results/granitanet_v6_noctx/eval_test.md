# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (16624 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.06%  |
| Top-3 Accuracy | 99.83%  |
| Macro F1       | 98.17%  |
| Weighted F1    | 98.06%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.27%     |
| articulation | 96.80%     |
| barline      | 94.82%     |
| clef         | 99.77%     |
| dynamic      | 99.97%     |
| fingering    | 99.17%     |
| note         | 92.73%     |
| ornament     | 99.85%     |
| pedal        | 99.80%     |
| rest         | 97.49%     |
| timesig      | 99.29%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.51%  |
| stroke   |      1476 | 97.29%  |
| crop     |      2915 | 99.28%  |
| hand     |      1520 | 86.25%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      32 |
| note64th                | note32nd           |      27 |
| augmentationDot         | articStaccatoAbove |      27 |
| noteQuarter             | noteHalf           |      24 |
| note32nd                | note16th           |      21 |
| fing1                   | barlineSingle      |      21 |
| note16th                | note8th            |      12 |
| note32nd                | note64th           |      12 |
| note16th                | note32nd           |      11 |
| note64th                | note16th           |       9 |
| rest32nd                | rest64th           |       9 |
| articStaccatissimoAbove | barlineSingle      |       8 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 95.87%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.23%     |
| articStaccatoAbove      | articulation | 86.02%     |
| articTenutoAbove        | articulation | 99.58%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.39%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 97.97%     |
| accidentalSharp         | accidental   | 99.13%     |
| accidentalDoubleSharp   | accidental   | 99.65%     |
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
| noteWhole               | note         | 97.08%     |
| noteHalf                | note         | 95.50%     |
| noteQuarter             | note         | 94.72%     |
| note8th                 | note         | 96.95%     |
| note16th                | note         | 90.78%     |
| note32nd                | note         | 87.82%     |
| note64th                | note         | 90.50%     |
| augmentationDot         | note         | 88.47%     |
| restWholeHalf           | rest         | 99.15%     |
| restQuarter             | rest         | 96.82%     |
| rest8th                 | rest         | 98.60%     |
| rest16th                | rest         | 97.01%     |
| rest32nd                | rest         | 95.59%     |
| rest64th                | rest         | 97.74%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 99.83%     |
| cClef                   | clef         | 99.50%     |
| timeSigCommon           | timesig      | 99.82%     |
| timeSigCutCommon        | timesig      | 100.00%    |
| timeSig2over2           | timesig      | 99.46%     |
| timeSig2over4           | timesig      | 98.55%     |
| timeSig3over4           | timesig      | 98.77%     |
| timeSig3over8           | timesig      | 98.33%     |
| timeSig4over4           | timesig      | 99.10%     |
| timeSig6over8           | timesig      | 99.66%     |
| timeSig9over8           | timesig      | 99.16%     |
| timeSig12over8          | timesig      | 100.00%    |
| barlineSingle           | barline      | 94.82%     |
