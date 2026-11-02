# Context effect: checkpoints/battuta_v8_ctx/best_model.pt on the test split

Model context dim 16; `context masked` feeds the same model all-zero vectors.

|                                               | with context   | context masked   | context-free model   |
|-----------------------------------------------|----------------|------------------|----------------------|
| top-1                                         | 99.30%         | 98.61%           | 98.85%               |
| macro F1                                      | 99.27%         | 98.43%           | 98.73%               |
| font top-1                                    | 99.78%         | 99.43%           | 99.26%               |
| stroke top-1                                  | 99.21%         | 96.99%           | 99.05%               |
| crop top-1                                    | 99.76%         | 98.53%           | 99.17%               |
| hand top-1                                    | 92.43%         | 90.27%           | 90.68%               |
| emnist top-1                                  | 99.65%         | 99.65%           | 99.78%               |
| samples with a known context                  | 10220 / 13655  |                  |                      |
| augmentationDot <-> articStaccatoAbove errors | 20             | 87               | 64                   |
| noteWhole <-> augmentationDot errors          | 4              | 16               | 13                   |
| restQuarter <-> accidentalNatural errors      | 6              | 8                | 6                    |
| noteQuarter <-> noteHalf errors               | 19             | 18               | 20                   |
| fing1 <-> articStaccatissimoAbove errors      | 4              | 10               | 8                    |
| note8th <-> note16th errors                   | 20             | 20               | 25                   |
