# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta/dataset.npz (13655 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.30%  |
| Top-3 Accuracy | 99.98%  |
| Macro F1       | 99.27%  |
| Weighted F1    | 99.30%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.67%     |
| articulation | 98.81%     |
| clef         | 100.00%    |
| dynamic      | 99.89%     |
| fingering    | 99.75%     |
| navigation   | 99.97%     |
| note         | 96.79%     |
| ornament     | 99.92%     |
| rest         | 97.73%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.78%  |
| stroke   |      1262 | 99.21%  |
| crop     |      2044 | 99.76%  |
| hand     |       740 | 92.43%  |
| emnist   |      2288 | 99.65%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| noteHalf           | noteQuarter        |      16 |
| articStaccatoAbove | augmentationDot    |      14 |
| note16th           | note8th            |      14 |
| restQuarter        | accidentalNatural  |       6 |
| note8th            | note16th           |       6 |
| augmentationDot    | articStaccatoAbove |       6 |
| noteWhole          | augmentationDot    |       4 |
| noteQuarter        | noteHalf           |       3 |
| restQuarter        | note8th            |       3 |
| restQuarter        | noteQuarter        |       2 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.87%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.86%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.79%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 95.47%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 99.81%     |
| articStaccatissimoAbove | articulation | 99.16%     |
| articMarcatoAbove       | articulation | 99.60%     |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 99.01%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 99.72%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 99.43%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.86%     |
| dynamicSforzato         | dynamic      | 100.00%    |
| noteWhole               | note         | 98.90%     |
| noteHalf                | note         | 96.83%     |
| noteQuarter             | note         | 96.43%     |
| note8th                 | note         | 96.25%     |
| note16th                | note         | 96.45%     |
| augmentationDot         | note         | 95.89%     |
| restQuarter             | rest         | 97.73%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 99.86%     |
| coda                    | navigation   | 100.00%    |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.61%  |
| Top-3 Accuracy | 99.96%  |
| Macro F1       | 98.43%  |
| Weighted F1    | 98.60%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.45%     |
| articulation | 95.24%     |
| clef         | 100.00%    |
| dynamic      | 99.89%     |
| fingering    | 99.55%     |
| navigation   | 99.97%     |
| note         | 94.23%     |
| ornament     | 99.79%     |
| rest         | 97.19%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.43%  |
| stroke   |      1262 | 96.99%  |
| crop     |      2044 | 98.53%  |
| hand     |       740 | 90.27%  |
| emnist   |      2288 | 99.65%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| articStaccatoAbove      | augmentationDot         |      65 |
| augmentationDot         | articStaccatoAbove      |      22 |
| noteHalf                | noteQuarter             |      16 |
| note16th                | note8th                 |      15 |
| noteWhole               | augmentationDot         |      13 |
| restQuarter             | accidentalNatural       |       8 |
| articStaccatissimoAbove | fing1                   |       7 |
| articStaccatoAbove      | noteWhole               |       6 |
| note8th                 | note16th                |       5 |
| fing1                   | articStaccatissimoAbove |       3 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 97.90%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.86%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.37%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.59%     |
| articStaccatoAbove      | articulation | 78.92%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 99.81%     |
| articStaccatissimoAbove | articulation | 97.88%     |
| articMarcatoAbove       | articulation | 99.60%     |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 98.36%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 99.72%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 99.43%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.86%     |
| dynamicSforzato         | dynamic      | 100.00%    |
| noteWhole               | note         | 95.96%     |
| noteHalf                | note         | 96.67%     |
| noteQuarter             | note         | 96.58%     |
| note8th                 | note         | 96.40%     |
| note16th                | note         | 96.26%     |
| augmentationDot         | note         | 83.52%     |
| restQuarter             | rest         | 97.19%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 99.86%     |
| coda                    | navigation   | 100.00%    |
