# Context effect: checkpoints/battuta_v10_replies_ctx/best_model.pt on the test split

Model context dim 16; `context masked` feeds the same model all-zero vectors.

|                                               | with context   | context masked   | context-free model   |
|-----------------------------------------------|----------------|------------------|----------------------|
| top-1                                         | 99.27%         | 98.70%           | 98.76%               |
| macro F1                                      | 99.23%         | 98.56%           | 98.63%               |
| font top-1                                    | 99.77%         | 99.34%           | 99.11%               |
| stroke top-1                                  | 99.13%         | 97.46%           | 99.21%               |
| crop top-1                                    | 99.80%         | 99.36%           | 99.22%               |
| hand top-1                                    | 91.76%         | 89.86%           | 90.41%               |
| emnist top-1                                  | 99.74%         | 99.61%           | 99.65%               |
| samples with a known context                  | 10217 / 13655  |                  |                      |
| augmentationDot <-> articStaccatoAbove errors | 23             | 78               | 71                   |
| noteWhole <-> augmentationDot errors          | 4              | 15               | 13                   |
| restQuarter <-> accidentalNatural errors      | 5              | 6                | 7                    |
| noteQuarter <-> noteHalf errors               | 19             | 18               | 17                   |
| fing1 <-> articStaccatissimoAbove errors      | 2              | 6                | 12                   |
| note8th <-> note16th errors                   | 24             | 24               | 25                   |
