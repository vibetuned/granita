# granitanet on the test split of /home/flux/projects/musicset/data/symbols-brush/dataset.npz (15873 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 97.68%  |
| Top-3 Accuracy | 99.88%  |
| Macro F1       | 97.80%  |
| Weighted F1    | 97.69%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 98.79%     |
| articulation | 96.73%     |
| barline      | 94.39%     |
| clef         | 99.71%     |
| dynamic      | 99.98%     |
| fingering    | 98.91%     |
| note         | 91.82%     |
| ornament     | 99.84%     |
| pedal        | 100.00%    |
| rest         | 96.84%     |
| timesig      | 98.56%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.40%  |
| stroke   |      1476 | 97.83%  |
| crop     |      2164 | 99.72%  |
| hand     |      1520 | 82.50%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| augmentationDot         | articStaccatoAbove |      37 |
| articStaccatoAbove      | augmentationDot    |      28 |
| note64th                | note32nd           |      24 |
| note32nd                | note64th           |      22 |
| note16th                | note32nd           |      22 |
| fing1                   | barlineSingle      |      20 |
| noteWhole               | augmentationDot    |      17 |
| note32nd                | note16th           |      14 |
| noteHalf                | noteQuarter        |      13 |
| noteQuarter             | noteHalf           |      11 |
| restQuarter             | accidentalNatural  |      11 |
| rest64th                | rest32nd           |      10 |
| note16th                | note8th            |       8 |
| articStaccatissimoAbove | barlineSingle      |       8 |
| rest32nd                | rest16th           |       7 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 94.74%     |
| fing2                   | fingering    | 99.80%     |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.80%     |
| arpeg                   | ornament     | 99.39%     |
| articStaccatoAbove      | articulation | 87.16%     |
| articTenutoAbove        | articulation | 98.98%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.70%     |
| articMarcatoAbove       | articulation | 99.81%     |
| accidentalFlat          | accidental   | 99.32%     |
| accidentalNatural       | accidental   | 97.21%     |
| accidentalSharp         | accidental   | 98.13%     |
| accidentalDoubleSharp   | accidental   | 99.30%     |
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
| noteWhole               | note         | 96.23%     |
| noteHalf                | note         | 95.59%     |
| noteQuarter             | note         | 95.05%     |
| note8th                 | note         | 96.26%     |
| note16th                | note         | 90.38%     |
| note32nd                | note         | 84.97%     |
| note64th                | note         | 91.40%     |
| augmentationDot         | note         | 84.71%     |
| restWholeHalf           | rest         | 97.81%     |
| restQuarter             | rest         | 96.03%     |
| rest8th                 | rest         | 97.89%     |
| rest16th                | rest         | 96.81%     |
| rest32nd                | rest         | 95.45%     |
| rest64th                | rest         | 97.05%     |
| gClef                   | clef         | 99.82%     |
| fClef                   | clef         | 99.48%     |
| cClef                   | clef         | 99.83%     |
| timeSigCommon           | timesig      | 99.58%     |
| timeSigCutCommon        | timesig      | 99.79%     |
| timeSig2over2           | timesig      | 97.93%     |
| timeSig2over4           | timesig      | 97.13%     |
| timeSig3over4           | timesig      | 97.07%     |
| timeSig3over8           | timesig      | 97.27%     |
| timeSig4over4           | timesig      | 99.37%     |
| timeSig6over8           | timesig      | 99.79%     |
| timeSig9over8           | timesig      | 98.11%     |
| timeSig12over8          | timesig      | 99.58%     |
| barlineSingle           | barline      | 94.39%     |
