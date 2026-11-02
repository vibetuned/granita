# granitanet on the test split of /home/flux/projects/musicset/data/symbols-battuta-tablet/dataset.npz (12862 samples)

## Summary

| Metric         | Value   |
|----------------|---------|
| Top-1 Accuracy | 99.00%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 98.97%  |
| Weighted F1    | 98.98%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.89%     |
| articulation | 95.15%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 99.67%     |
| navigation   | 100.00%    |
| note         | 96.99%     |
| ornament     | 100.00%    |
| rest         | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      9805 | 99.59%  |
| stroke   |      3057 | 97.09%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| articStaccatoAbove      | augmentationDot         |      91 |
| augmentationDot         | articStaccatoAbove      |      20 |
| articStaccatissimoAbove | fing1                   |       7 |
| fing1                   | articStaccatissimoAbove |       3 |
| augmentationDot         | noteWhole               |       3 |
| articStaccatoAbove      | noteWhole               |       3 |
| articStaccatissimoAbove | accidentalNatural       |       1 |
| articStaccatoAbove      | accidentalNatural       |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.34%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 77.58%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.15%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 99.67%     |
| accidentalSharp         | accidental   | 100.00%    |
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
| noteWhole               | note         | 99.01%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| augmentationDot         | note         | 82.93%     |
| restQuarter             | rest         | 100.00%    |
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
| Top-1 Accuracy | 99.00%  |
| Top-3 Accuracy | 100.00% |
| Macro F1       | 98.97%  |
| Weighted F1    | 98.98%  |

## Group Performance

| Group        | Macro F1   |
|--------------|------------|
| accidental   | 99.89%     |
| articulation | 95.15%     |
| clef         | 100.00%    |
| dynamic      | 100.00%    |
| fingering    | 99.67%     |
| navigation   | 100.00%    |
| note         | 96.99%     |
| ornament     | 100.00%    |
| rest         | 100.00%    |

## Per-Source Accuracy

| Source   |   Samples | Top-1   |
|----------|-----------|---------|
| font     |      9805 | 99.59%  |
| stroke   |      3057 | 97.09%  |

## Top Confusions

| True                    | Predicted               |   Count |
|-------------------------|-------------------------|---------|
| articStaccatoAbove      | augmentationDot         |      91 |
| augmentationDot         | articStaccatoAbove      |      20 |
| articStaccatissimoAbove | fing1                   |       7 |
| fing1                   | articStaccatissimoAbove |       3 |
| augmentationDot         | noteWhole               |       3 |
| articStaccatoAbove      | noteWhole               |       3 |
| articStaccatissimoAbove | accidentalNatural       |       1 |
| articStaccatoAbove      | accidentalNatural       |       1 |

## Per-Class Breakdown

| Class                   | Group        | F1 Score   |
|-------------------------|--------------|------------|
| fing1                   | fingering    | 98.34%     |
| fing2                   | fingering    | 100.00%    |
| fing3                   | fingering    | 100.00%    |
| fing4                   | fingering    | 100.00%    |
| fing5                   | fingering    | 100.00%    |
| trill                   | ornament     | 100.00%    |
| mordent                 | ornament     | 100.00%    |
| turn                    | ornament     | 100.00%    |
| fermata                 | ornament     | 100.00%    |
| arpeg                   | ornament     | 100.00%    |
| articStaccatoAbove      | articulation | 77.58%     |
| articTenutoAbove        | articulation | 100.00%    |
| articAccentAbove        | articulation | 100.00%    |
| articStaccatissimoAbove | articulation | 98.15%     |
| articMarcatoAbove       | articulation | 100.00%    |
| accidentalFlat          | accidental   | 100.00%    |
| accidentalNatural       | accidental   | 99.67%     |
| accidentalSharp         | accidental   | 100.00%    |
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
| noteWhole               | note         | 99.01%     |
| noteHalf                | note         | 100.00%    |
| noteQuarter             | note         | 100.00%    |
| note8th                 | note         | 100.00%    |
| note16th                | note         | 100.00%    |
| augmentationDot         | note         | 82.93%     |
| restQuarter             | rest         | 100.00%    |
| gClef                   | clef         | 100.00%    |
| fClef                   | clef         | 100.00%    |
| cClef                   | clef         | 100.00%    |
| dalSegno                | navigation   | 100.00%    |
| daCapo                  | navigation   | 100.00%    |
| fine                    | navigation   | 100.00%    |
| segno                   | navigation   | 100.00%    |
| coda                    | navigation   | 100.00%    |
