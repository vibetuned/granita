# Context effect: /home/flux/projects/granita/checkpoints/granitanet_v7_ctx/best_model.pt on the test split

Model context dim 16; `context masked` feeds the same model all-zero vectors.

|                                                  | with context   | context masked   | context-free model   |
|--------------------------------------------------|----------------|------------------|----------------------|
| top-1                                            | 98.59%         | 97.76%           | 97.88%               |
| macro F1                                         | 98.57%         | 97.67%           | 97.81%               |
| font top-1                                       | 99.69%         | 99.25%           | 99.33%               |
| stroke top-1                                     | 98.85%         | 97.15%           | 97.97%               |
| crop top-1                                       | 99.71%         | 98.84%           | 99.21%               |
| hand top-1                                       | 86.95%         | 83.91%           | 82.84%               |
| emnist top-1                                     | 98.79%         | 98.19%           | 98.30%               |
| samples with a known context                     | 11677 / 15546  |                  |                      |
| augmentationDot <-> articStaccatoAbove errors    | 25             | 77               | 61                   |
| fing1 <-> barlineSingle errors                   | 6              | 25               | 20                   |
| articStaccatissimoAbove <-> barlineSingle errors | 10             | 16               | 7                    |
| noteWhole <-> augmentationDot errors             | 2              | 9                | 10                   |
| restQuarter <-> accidentalNatural errors         | 1              | 2                | 2                    |
| noteQuarter <-> noteHalf errors                  | 19             | 19               | 23                   |
