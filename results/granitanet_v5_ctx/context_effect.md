# Context effect: /home/flux/projects/granita/checkpoints/granitanet_v5_ctx/best_model.pt on the test split

Model context dim 9; `context masked` feeds the same model all-zero vectors.

|                                                  | with context   | context masked   | context-free model   |
|--------------------------------------------------|----------------|------------------|----------------------|
| top-1                                            | 98.54%         | 97.81%           | 97.88%               |
| macro F1                                         | 98.62%         | 97.92%           | 97.99%               |
| font top-1                                       | 99.83%         | 99.49%           | 99.44%               |
| stroke top-1                                     | 98.64%         | 97.02%           | 97.70%               |
| crop top-1                                       | 99.77%         | 99.31%           | 99.54%               |
| hand top-1                                       | 87.63%         | 84.61%           | 84.67%               |
| samples with a known context                     | 8853 / 15873   |                  |                      |
| augmentationDot <-> articStaccatoAbove errors    | 23             | 67               | 67                   |
| fing1 <-> barlineSingle errors                   | 7              | 21               | 21                   |
| articStaccatissimoAbove <-> barlineSingle errors | 2              | 10               | 7                    |
| noteWhole <-> augmentationDot errors             | 7              | 18               | 17                   |
| restQuarter <-> accidentalNatural errors         | 7              | 13               | 13                   |
| noteQuarter <-> noteHalf errors                  | 27             | 27               | 26                   |
