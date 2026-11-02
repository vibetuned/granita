# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta-fornes/dataset.npz (14198 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.30%  |
| Top-3 Accuracy | 99.97%  |
| Macro F1       | 99.25%  |
| Weighted F1    | 99.30%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.39%     |
| articulation | 98.81%     |
| clef         | 99.96%     |
| dynamic      | 99.83%     |
| fingering    | 99.72%     |
| navigation   | 99.92%     |
| note         | 96.98%     |
| ornament     | 99.92%     |
| rest         | 97.73%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.73%  |
| stroke   |      1262 | 99.45%  |
| crop     |      2044 | 99.80%  |
| hand     |       740 | 92.43%  |
| emnist   |      2288 | 99.61%  |
| fornes   |       543 | 99.45%  |

## Top Confusions

| True               | Predicted               |   Count |
|--------------------|-------------------------|---------|
| note16th           | note8th                 |      18 |
| articStaccatoAbove | augmentationDot         |      12 |
| noteHalf           | noteQuarter             |      11 |
| augmentationDot    | articStaccatoAbove      |       9 |
| note8th            | note16th                |       5 |
| restQuarter        | accidentalNatural       |       5 |
| noteWhole          | augmentationDot         |       4 |
| noteQuarter        | noteHalf                |       4 |
| fing1              | articStaccatissimoAbove |       3 |
| restQuarter        | noteQuarter             |       2 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.87%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.72%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.79%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 95.32%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.95%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.60%     |
| accidentalNatural       | accidental   | 98.98%     |
| accidentalSharp         | accidental   | 99.60%     |
| dynamicPiano            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 99.85%     |
| dynamicSforzando        | dynamic      | 99.86%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 99.86%     |
| dynamicMF               | dynamic      | 99.86%     |
| dynamicFF               | dynamic      | 99.72%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.72%     |
| dynamicSforzato         | dynamic      | 99.73%     |
| noteWhole               | note         | 98.90%     |
| noteHalf                | note         | 97.33%     |
| noteQuarter             | note         | 97.29%     |
| note8th                 | note         | 96.42%     |
| note16th                | note         | 96.27%     |
| augmentationDot         | note         | 95.68%     |
| restQuarter             | rest         | 97.73%     |
| gClef                   | clef         | 99.88%     |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 99.85%     |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 99.77%     |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.69%  |
| Top-3 Accuracy | 99.96%  |
| Macro F1       | 98.52%  |
| Weighted F1    | 98.69%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.22%     |
| articulation | 96.14%     |
| clef         | 99.87%     |
| dynamic      | 99.79%     |
| fingering    | 99.52%     |
| navigation   | 99.87%     |
| note         | 94.45%     |
| ornament     | 99.88%     |
| rest         | 97.38%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.25%  |
| stroke   |      1262 | 98.65%  |
| crop     |      2044 | 98.87%  |
| hand     |       740 | 89.86%  |
| emnist   |      2288 | 99.48%  |
| fornes   |       543 | 99.26%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      43 |
| augmentationDot         | articStaccatoAbove |      31 |
| note16th                | note8th            |      19 |
| noteHalf                | noteQuarter        |      15 |
| noteWhole               | augmentationDot    |      11 |
| articStaccatissimoAbove | fing1              |       7 |
| articStaccatoAbove      | noteWhole          |       7 |
| restQuarter             | accidentalNatural  |       6 |
| note8th                 | note16th           |       5 |
| augmentationDot         | noteWhole          |       4 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.31%     |
| fing2                   | fingering    | 99.73%     |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.58%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.58%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 83.02%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 97.88%     |
| articMarcatoAbove       | articulation | 99.80%     |
| accidentalFlat          | accidental   | 99.47%     |
| accidentalNatural       | accidental   | 98.72%     |
| accidentalSharp         | accidental   | 99.47%     |
| dynamicPiano            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 99.85%     |
| dynamicSforzando        | dynamic      | 99.59%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 99.86%     |
| dynamicMF               | dynamic      | 99.86%     |
| dynamicFF               | dynamic      | 99.58%     |
| dynamicFFF              | dynamic      | 99.86%     |
| dynamicForzando         | dynamic      | 99.72%     |
| dynamicSforzato         | dynamic      | 99.73%     |
| noteWhole               | note         | 95.99%     |
| noteHalf                | note         | 96.84%     |
| noteQuarter             | note         | 96.86%     |
| note8th                 | note         | 96.27%     |
| note16th                | note         | 95.79%     |
| augmentationDot         | note         | 84.94%     |
| restQuarter             | rest         | 97.38%     |
| gClef                   | clef         | 99.88%     |
| fClef                   | clef         | 99.73%     |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 99.85%     |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 99.86%     |
| coda                    | navigation   | 99.65%     |
