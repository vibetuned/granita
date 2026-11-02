# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta/dataset.npz (13655 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.85%  |
| Top-3 Accuracy | 99.95%  |
| Macro F1       | 98.73%  |
| Weighted F1    | 98.85%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.56%     |
| articulation | 96.79%     |
| clef         | 100.00%    |
| dynamic      | 99.93%     |
| fingering    | 99.66%     |
| navigation   | 100.00%    |
| note         | 94.73%     |
| ornament     | 99.79%     |
| rest         | 97.90%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.26%  |
| stroke   |      1262 | 99.05%  |
| crop     |      2044 | 99.17%  |
| hand     |       740 | 90.68%  |
| emnist   |      2288 | 99.78%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| augmentationDot         | articStaccatoAbove |      33 |
| articStaccatoAbove      | augmentationDot    |      31 |
| note16th                | note8th            |      19 |
| noteHalf                | noteQuarter        |      13 |
| noteWhole               | augmentationDot    |      11 |
| noteQuarter             | noteHalf           |       7 |
| restQuarter             | accidentalNatural  |       6 |
| note8th                 | note16th           |       6 |
| articStaccatoAbove      | noteWhole          |       5 |
| articStaccatissimoAbove | fing1              |       4 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.59%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.72%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.17%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 85.83%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.32%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.83%     |
| accidentalNatural       | accidental   | 98.84%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 99.58%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.86%     |
| dynamicSforzato         | dynamic      | 100.00%    |
| noteWhole               | note         | 96.50%     |
| noteHalf                | note         | 96.86%     |
| noteQuarter             | note         | 96.24%     |
| note8th                 | note         | 96.27%     |
| note16th                | note         | 95.80%     |
| augmentationDot         | note         | 86.70%     |
| restQuarter             | rest         | 97.90%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |
