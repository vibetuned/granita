# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta-replies/dataset.npz (13655 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.27%  |
| Top-3 Accuracy | 99.97%  |
| Macro F1       | 99.23%  |
| Weighted F1    | 99.27%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.50%     |
| articulation | 98.76%     |
| clef         | 100.00%    |
| dynamic      | 99.88%     |
| fingering    | 99.86%     |
| navigation   | 100.00%    |
| note         | 96.60%     |
| ornament     | 99.85%     |
| rest         | 97.91%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.77%  |
| stroke   |      1262 | 99.13%  |
| crop     |      2044 | 99.80%  |
| hand     |       740 | 91.76%  |
| emnist   |      2288 | 99.74%  |

## Top Confusions

| True               | Predicted          |   Count |
|--------------------|--------------------|---------|
| note16th           | note8th            |      19 |
| articStaccatoAbove | augmentationDot    |      17 |
| noteHalf           | noteQuarter        |      15 |
| augmentationDot    | articStaccatoAbove |       6 |
| note8th            | note16th           |       5 |
| noteWhole          | augmentationDot    |       4 |
| restQuarter        | accidentalNatural  |       4 |
| noteQuarter        | noteHalf           |       4 |
| articStaccatoAbove | noteWhole          |       3 |
| accidentalNatural  | accidentalFlat     |       2 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 99.29%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.58%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 99.86%     |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 94.61%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 99.58%     |
| articMarcatoAbove       | articulation | 99.60%     |
| accidentalFlat          | accidental   | 99.65%     |
| accidentalNatural       | accidental   | 98.84%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 99.72%     |
| dynamicFFF              | dynamic      | 99.73%     |
| dynamicForzando         | dynamic      | 99.72%     |
| dynamicSforzato         | dynamic      | 99.73%     |
| noteWhole               | note         | 98.72%     |
| noteHalf                | note         | 96.68%     |
| noteQuarter             | note         | 96.72%     |
| note8th                 | note         | 95.99%     |
| note16th                | note         | 96.10%     |
| augmentationDot         | note         | 95.40%     |
| restQuarter             | rest         | 97.91%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |

# Same model with the context masked (all zeros)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 98.70%  |
| Top-3 Accuracy | 99.96%  |
| Macro F1       | 98.56%  |
| Weighted F1    | 98.70%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.44%     |
| articulation | 96.03%     |
| clef         | 100.00%    |
| dynamic      | 99.85%     |
| fingering    | 99.72%     |
| navigation   | 99.95%     |
| note         | 94.39%     |
| ornament     | 99.76%     |
| rest         | 97.55%     |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      7321 | 99.34%  |
| stroke   |      1262 | 97.46%  |
| crop     |      2044 | 99.36%  |
| hand     |       740 | 89.86%  |
| emnist   |      2288 | 99.61%  |

## Top Confusions

| True                    | Predicted          |   Count |
|-------------------------|--------------------|---------|
| articStaccatoAbove      | augmentationDot    |      47 |
| augmentationDot         | articStaccatoAbove |      31 |
| note16th                | note8th            |      19 |
| noteHalf                | noteQuarter        |      15 |
| noteWhole               | augmentationDot    |      12 |
| articStaccatoAbove      | noteWhole          |       7 |
| restQuarter             | accidentalNatural  |       6 |
| note8th                 | note16th           |       5 |
| articStaccatissimoAbove | fing1              |       5 |
| restQuarter             | trill              |       4 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.74%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 99.86%     |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 99.17%     |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 99.86%     |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 99.80%     |
| articStaccatoAbove      | articulation | 82.03%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 99.81%     |
| articStaccatissimoAbove | articulation | 98.73%     |
| articMarcatoAbove       | articulation | 99.60%     |
| accidentalFlat          | accidental   | 99.65%     |
| accidentalNatural       | accidental   | 98.68%     |
| accidentalSharp         | accidental   | 100.00%    |
| dynamicPiano            | dynamic      | 99.86%     |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 99.86%     |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 99.86%     |
| dynamicFF               | dynamic      | 99.72%     |
| dynamicFFF              | dynamic      | 99.73%     |
| dynamicForzando         | dynamic      | 99.72%     |
| dynamicSforzato         | dynamic      | 99.73%     |
| noteWhole               | note         | 95.97%     |
| noteHalf                | note         | 96.84%     |
| noteQuarter             | note         | 96.87%     |
| note8th                 | note         | 96.27%     |
| note16th                | note         | 95.94%     |
| augmentationDot         | note         | 84.42%     |
| restQuarter             | rest         | 97.55%     |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 99.86%     |
| coda                    | navigation   | 99.88%     |
