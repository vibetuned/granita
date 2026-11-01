# Context effect: /home/flux/projects/granita/checkpoints/granitanet_v6_ctx/best_model.pt on the test split

Model context dim 9; `context masked` feeds the same model all-zero vectors.

|                                                  | with context   | context masked   | context-free model   |
|--------------------------------------------------|----------------|------------------|----------------------|
| top-1                                            | 98.68%         | 97.94%           | 98.06%               |
| macro F1                                         | 98.79%         | 98.05%           | 98.17%               |
| font top-1                                       | 99.79%         | 99.51%           | 99.51%               |
| stroke top-1                                     | 99.05%         | 96.61%           | 97.29%               |
| crop top-1                                       | 99.49%         | 99.07%           | 99.28%               |
| hand top-1                                       | 88.95%         | 86.05%           | 86.25%               |
| samples with a known context                     | 12543 / 16624  |                  |                      |
| augmentationDot <-> articStaccatoAbove errors    | 23             | 69               | 59                   |
| fing1 <-> barlineSingle errors                   | 4              | 21               | 21                   |
| articStaccatissimoAbove <-> barlineSingle errors | 5              | 11               | 8                    |
| noteWhole <-> augmentationDot errors             | 1              | 5                | 7                    |
| restQuarter <-> accidentalNatural errors         | 5              | 8                | 8                    |
| noteQuarter <-> noteHalf errors                  | 22             | 27               | 29                   |
