# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15873 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.91%  |
| Top-3 Accuracy | 99.91%  |
| Macro F1       | 98.03%  |
| Weighted F1    | 97.92%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 98.79%     |
| articulation | 97.00%     |
| barline      | 95.08%     |
| clef         | 99.77%     |
| dynamic      | 99.97%     |
| fingering    | 99.07%     |
| note         | 92.91%     |
| ornament     | 99.88%     |
| pedal        | 100.00%    |
| rest         | 96.70%     |
| timesig      | 98.85%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.45%  |
| stroke   |      1476 | 97.76%  |
| crop     |      2164 | 99.54%  |
| hand     |      1520 | 84.93%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| augmentationDot         | articStaccatoAbove |      34 |
| articStaccatoAbove      | augmentationDot    |      27 |
| fing1                   | barlineSingle      |      21 |
| note64th                | note32nd           |      20 |
| note16th                | note32nd           |      19 |
| note32nd                | note64th           |      19 |
| noteQuarter             | noteHalf           |      18 |
| noteWhole               | augmentationDot    |      14 |
| restQuarter             | accidentalNatural  |      13 |
| rest64th                | rest32nd           |      11 |
| rest32nd                | rest16th           |       8 |
| articStaccatissimoAbove | barlineSingle      |       8 |
| timeSig3over4           | timeSig2over4      |       8 |
| note8th                 | note16th           |       7 |
| rest16th                | rest8th            |       6 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 95.51%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.82%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.39%     |
| articStaccatoAbove      | articulation | 87.67%     |
| articTenutoAbove        | articulation | 99.18%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.32%     |
| articMarcatoAbove       | articulation | 99.81%     |
| accidentalFlat          | accidental   | 99.32%     |
| accidentalNatural       | accidental   | 96.18%     |
| accidentalSharp         | accidental   | 98.82%     |
| accidentalDoubleSharp   | accidental   | 99.65%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicMezzo            | dynamic      | 99.79%     |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicZ                | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 100.00%    |
| dynamicFFF              | dynamic      | 99.80%     |
| dynamicForzando         | dynamic      | 100.00%    |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 100.00%    |
| noteWhole               | note         | 96.27%     |
| noteHalf                | note         | 95.71%     |
| noteQuarter             | note         | 95.62%     |
| note8th                 | note         | 96.93%     |
| note16th                | note         | 92.48%     |
| note32nd                | note         | 88.73%     |
| note64th                | note         | 92.17%     |
| augmentationDot         | note         | 85.33%     |
| restWholeHalf           | rest         | 98.35%     |
| restQuarter             | rest         | 95.53%     |
| rest8th                 | rest         | 97.55%     |
| rest16th                | rest         | 96.63%     |
| rest32nd                | rest         | 95.10%     |
| rest64th                | rest         | 97.03%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 99.31%     |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 99.37%     |
| timeSigCutCommon        | timesig      | 99.58%     |
| timeSig2over2           | timesig      | 98.35%     |
| timeSig2over4           | timesig      | 97.53%     |
| timeSig3over4           | timesig      | 97.24%     |
| timeSig3over8           | timesig      | 98.76%     |
| timeSig4over4           | timesig      | 99.37%     |
| timeSig6over8           | timesig      | 99.37%     |
| timeSig9over8           | timesig      | 99.38%     |
| timeSig12over8          | timesig      | 99.59%     |
| barlineSingle           | barline      | 95.08%     |
