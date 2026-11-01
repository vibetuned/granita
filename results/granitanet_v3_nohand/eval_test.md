# granitanet on the test split of /home/flux/projects/musicset/data/symbols/dataset.npz (15873 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 91.80%  |
| Top-3 Accuracy | 94.34%  |
| Macro F1       | 92.30%  |
| Weighted F1    | 92.14%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 92.61%     |
| articulation | 86.00%     |
| barline      | 90.13%     |
| clef         | 97.22%     |
| dynamic      | 98.91%     |
| fingering    | 94.54%     |
| note         | 83.44%     |
| ornament     | 95.09%     |
| pedal        | 97.51%     |
| rest         | 90.42%     |
| timesig      | 91.32%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     10713 | 99.49%  |
| stroke   |      1476 | 96.82%  |
| crop     |      2164 | 99.58%  |
| hand     |      1520 | 21.71%  |

## Top Confusions

| True               | Predicted               |   Count |
|--------------------|-------------------------|---------|
| noteQuarter        | articStaccatissimoAbove |      65 |
| noteHalf           | articStaccatissimoAbove |      59 |
| articStaccatoAbove | augmentationDot         |      56 |
| augmentationDot    | articStaccatoAbove      |      42 |
| noteWhole          | augmentationDot         |      35 |
| barlineSingle      | fing1                   |      35 |
| accidentalFlat     | articStaccatissimoAbove |      32 |
| timeSig6over8      | arpeg                   |      22 |
| note8th            | articStaccatissimoAbove |      18 |
| rest8th            | articStaccatissimoAbove |      18 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 83.94%     |
| fing2                   | fingering    | 97.80%     |
| fing3                   | fingering    | 97.89%     |
| fing4                   | fingering    | 96.14%     |
| fing5                   | fingering    | 96.92%     |
| trill                   | ornament     | 99.23%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.80%     |
| arpeg                   | ornament     | 76.44%     |
| articStaccatoAbove      | articulation | 73.78%     |
| articTenutoAbove        | articulation | 98.78%     |
| articAccentAbove        | articulation | 99.60%     |
| articStaccatissimoAbove | articulation | 59.92%     |
| articMarcatoAbove       | articulation | 97.92%     |
| accidentalFlat          | accidental   | 88.36%     |
| accidentalNatural       | accidental   | 90.81%     |
| accidentalSharp         | accidental   | 90.54%     |
| accidentalDoubleSharp   | accidental   | 93.36%     |
| accidentalDoubleFlat    | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 91.82%     |
| dynamicMezzo            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 97.78%     |
| dynamicSforzando        | dynamic      | 98.82%     |
| dynamicZ                | dynamic      | 99.80%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 99.80%     |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 98.84%     |
| dynamicSforzato         | dynamic      | 100.00%    |
| pedal                   | pedal        | 97.51%     |
| noteWhole               | note         | 93.28%     |
| noteHalf                | note         | 83.83%     |
| noteQuarter             | note         | 83.91%     |
| note8th                 | note         | 83.58%     |
| note16th                | note         | 83.58%     |
| note32nd                | note         | 83.06%     |
| note64th                | note         | 84.88%     |
| augmentationDot         | note         | 71.43%     |
| restWholeHalf           | rest         | 96.23%     |
| restQuarter             | rest         | 84.58%     |
| rest8th                 | rest         | 89.48%     |
| rest16th                | rest         | 89.45%     |
| rest32nd                | rest         | 89.84%     |
| rest64th                | rest         | 92.92%     |
| gClef                   | clef         | 99.11%     |
| fClef                   | clef         | 96.18%     |
| cClef                   | clef         | 96.36%     |
| timeSigCommon           | timesig      | 96.11%     |
| timeSigCutCommon        | timesig      | 92.64%     |
| timeSig2over2           | timesig      | 91.68%     |
| timeSig2over4           | timesig      | 89.96%     |
| timeSig3over4           | timesig      | 90.55%     |
| timeSig3over8           | timesig      | 90.74%     |
| timeSig4over4           | timesig      | 90.74%     |
| timeSig6over8           | timesig      | 89.69%     |
| timeSig9over8           | timesig      | 90.45%     |
| timeSig12over8          | timesig      | 90.58%     |
| barlineSingle           | barline      | 90.13%     |
