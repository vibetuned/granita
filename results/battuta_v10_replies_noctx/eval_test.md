# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta-replies/dataset.npz (13655 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.76%  |
| Top-3 Accuracy | 99.95%  |
| Macro F1       | 98.63%  |
| Weighted F1    | 98.76%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.56%     |
| articulation | 96.46%     |
| clef         | 100.00%    |
| dynamic      | 99.84%     |
| fingering    | 99.50%     |
| navigation   | 100.00%    |
| note         | 94.55%     |
| ornament     | 99.89%     |
| rest         | 97.54%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.11%  |
| stroke   |      1262 | 99.21%  |
| crop     |      2044 | 99.22%  |
| hand     |       740 | 90.41%  |
| emnist   |      2288 | 99.65%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| augmentationDot         | articStaccatoAbove      |      43 |
| articStaccatoAbove      | augmentationDot         |      28 |
| note16th                | note8th                 |      20 |
| noteHalf                | noteQuarter             |      14 |
| noteWhole               | augmentationDot         |      11 |
| articStaccatissimoAbove | fing1                   |       8 |
| restQuarter             | accidentalNatural       |       7 |
| note8th                 | note16th                |       5 |
| articStaccatoAbove      | noteWhole               |       4 |
| fing1                   | articStaccatissimoAbove |       4 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 97.76%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.72%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.79%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 99.86%     |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 85.04%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.46%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 98.68%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 99.71%     |
| dynamicForte            | dynamic      | 99.71%     |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 99.86%     |
| dynamicMF               | dynamic      | 99.86%     |
| dynamicFF               | dynamic      | 99.72%     |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 99.72%     |
| dynamicSforzato         | dynamic      | 99.87%     |
| noteWhole               | note         | 96.32%     |
| noteHalf                | note         | 97.15%     |
| noteQuarter             | note         | 96.86%     |
| note8th                 | note         | 95.99%     |
| note16th                | note         | 95.79%     |
| augmentationDot         | note         | 85.16%     |
| restQuarter             | rest         | 97.54%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |
