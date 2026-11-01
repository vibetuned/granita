# granitanet on the test split of /home/flux/projects/musicset/data/symbols-tablet/dataset.npz (18276 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.00%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.00%  |
| Weighted F1    | 99.00%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 100.00%    |
| articulation | 95.80%     |
| barline      | 89.55%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 98.02%     |
| note         | 97.68%     |
| ornament     | 100.00%    |
| pedal        | 100.00%    |
| rest         | 99.80%     |
| timesig      | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     14727 | 99.44%  |
| stroke   |      3549 | 97.18%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| fing1                   | barlineSingle      |      54 |
| articStaccatoAbove      | augmentationDot    |      51 |
| augmentationDot         | articStaccatoAbove |      42 |
| articStaccatissimoAbove | barlineSingle      |      16 |
| augmentationDot         | noteWhole          |       6 |
| restWholeHalf           | articTenutoAbove   |       6 |
| articStaccatoAbove      | noteWhole          |       5 |
| note16th                | note32nd           |       1 |
| articStaccatissimoAbove | rest8th            |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 90.11%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 82.93%     |
| articTenutoAbove        | articulation | 99.01%     |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.08%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 100.00%    |
| accidentalSharp         | accidental   | 100.00%    |
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
| note16th                | note         | 99.83%     |
| note32nd                | note         | 99.83%     |
| note64th                | note         | 100.00%    |
| augmentationDot         | note         | 83.58%     |
| restWholeHalf           | rest         | 98.99%     |
| restQuarter             | rest         | 100.00%    |
| rest8th                 | rest         | 99.83%     |
| rest16th                | rest         | 100.00%    |
| rest32nd                | rest         | 100.00%    |
| rest64th                | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| timeSigCommon           | timesig      | 100.00%    |
| timeSigCutCommon        | timesig      | 100.00%    |
| timeSig2over2           | timesig      | 100.00%    |
| timeSig2over4           | timesig      | 100.00%    |
| timeSig3over4           | timesig      | 100.00%    |
| timeSig3over8           | timesig      | 100.00%    |
| timeSig4over4           | timesig      | 100.00%    |
| timeSig6over8           | timesig      | 100.00%    |
| timeSig9over8           | timesig      | 100.00%    |
| timeSig12over8          | timesig      | 100.00%    |
| barlineSingle           | barline      | 89.55%     |
