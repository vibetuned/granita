# granitanet on the test split of /home/flux/projects/musicset/data/symbols-kakigori/dataset.npz (13742 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.57%  |
| Top-3 Accuracy | 99.98%  |
| Macro F1       | 99.59%  |
| Weighted F1    | 99.57%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.96%     |
| articulation | 98.30%     |
| barline      | 96.93%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 99.92%     |
| navigation   | 100.00%    |
| note         | 98.86%     |
| ornament     | 99.80%     |
| pedal        | 99.79%     |
| rest         | 99.89%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |     11183 | 99.76%  |
| crop     |      2559 | 98.75%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| articStaccatoAbove      | augmentationDot         |      25 |
| articStaccatissimoAbove | barlineSingle           |      10 |
| augmentationDot         | articStaccatoAbove      |       6 |
| arpeg                   | barlineSingle           |       3 |
| note32nd                | note16th                |       2 |
| note16th                | rest32nd                |       1 |
| barlineSingle           | articStaccatissimoAbove |       1 |
| note16th                | arpeg                   |       1 |
| noteQuarter             | note8th                 |       1 |
| note16th                | note32nd                |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 99.59%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 99.81%     |
| arpeg                   | ornament     | 99.19%     |
| articStaccatoAbove      | articulation | 94.00%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.71%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 100.00%    |
| accidentalSharp         | accidental   | 99.79%     |
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
| pedal                   | pedal        | 99.79%     |
| noteWhole               | note         | 100.00%    |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 99.57%     |
| note8th                 | note         | 99.40%     |
| note16th                | note         | 98.98%     |
| note32nd                | note         | 99.19%     |
| note64th                | note         | 100.00%    |
| augmentationDot         | note         | 93.71%     |
| restWholeHalf           | rest         | 100.00%    |
| restQuarter             | rest         | 100.00%    |
| rest8th                 | rest         | 99.80%     |
| rest16th                | rest         | 100.00%    |
| rest32nd                | rest         | 99.79%     |
| rest64th                | rest         | 99.76%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| barlineSingle           | barline      | 96.93%     |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |
