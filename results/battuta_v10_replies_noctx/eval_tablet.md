# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta-tablet/dataset.npz (12862 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.21%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 99.21%  |
| Weighted F1    | 99.21%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.89%     |
| articulation | 96.69%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 99.70%     |
| navigation   | 100.00%    |
| note         | 97.36%     |
| ornament     | 100.00%    |
| rest         | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      9805 | 99.41%  |
| stroke   |      3057 | 98.59%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| articStaccatoAbove      | augmentationDot         |      43 |
| augmentationDot         | articStaccatoAbove      |      39 |
| articStaccatissimoAbove | fing1                   |       5 |
| articStaccatoAbove      | noteWhole               |       5 |
| augmentationDot         | noteWhole               |       4 |
| fing1                   | articStaccatissimoAbove |       4 |
| accidentalNatural       | accidentalSharp         |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.50%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 84.97%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.50%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 99.83%     |
| accidentalSharp         | accidental   | 99.83%     |
| dynamicPiano            | dynamic      | 100.00%    |
| dynamicForte            | dynamic      | 100.00%    |
| dynamicSforzando        | dynamic      | 100.00%    |
| dynamicPP               | dynamic      | 100.00%    |
| dynamicMP               | dynamic      | 100.00%    |
| dynamicMF               | dynamic      | 100.00%    |
| dynamicFF               | dynamic      | 100.00%    |
| dynamicFFF              | dynamic      | 100.00%    |
| dynamicForzando         | dynamic      | 100.00%    |
| dynamicSforzato         | dynamic      | 100.00%    |
| noteWhole               | note         | 98.52%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| augmentationDot         | note         | 85.67%     |
| restQuarter             | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |
