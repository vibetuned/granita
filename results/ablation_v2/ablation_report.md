# Granita Model Ablation Report

## Overall Performance & Efficiency

| Model      |    Params |   Epochs | Train Time   | Batch-128 Latency   | Val F1   | Test Top-1   | Test Top-3   | Test Macro F1   |
|------------|-----------|----------|--------------|---------------------|----------|--------------|--------------|-----------------|
| tiny       |    77,793 |       13 | 44.0s        | 0.15 ms             | 98.95%   | 99.14%       | 99.87%       | 99.14%          |
| granitanet | 1,997,561 |       13 | 32.9s        | 0.76 ms             | 99.88%   | 99.78%       | 99.99%       | 99.78%          |
| regla      | 1,454,529 |       15 | 87.8s        | 2.48 ms             | 99.82%   | 99.79%       | 99.99%       | 99.79%          |
| iformer    | 2,668,225 |       12 | 79.4s        | 2.11 ms             | 99.80%   | 99.66%       | 99.98%       | 99.66%          |
| lsnet      | 2,517,465 |       15 | 147.7s       | 8.02 ms             | 99.78%   | 99.76%       | 99.98%       | 99.76%          |

## Semantic Group Macro F1 Breakdown

| Model      | Accidental   | Articulation   | Dynamic   | Fingering   | Ornament   | Pedal   |
|------------|--------------|----------------|-----------|-------------|------------|---------|
| tiny       | 99.07%       | 98.65%         | 99.72%    | 98.67%      | 98.87%     | 98.56%  |
| granitanet | 99.92%       | 99.51%         | 99.93%    | 99.53%      | 99.83%     | 99.59%  |
| regla      | 99.92%       | 99.38%         | 100.00%   | 99.45%      | 99.87%     | 100.00% |
| iformer    | 99.88%       | 99.12%         | 99.97%    | 99.15%      | 99.71%     | 99.80%  |
| lsnet      | 99.96%       | 99.43%         | 99.93%    | 99.45%      | 99.75%     | 99.80%  |

## Test Top-1 by Sample Source

`stroke` is the stylus-drawn subset, the input a score editor sends.

| Model      | Font Top-1   | Stroke Top-1   | Crop Top-1   |
|------------|--------------|----------------|--------------|
| tiny       | 99.45%       | 98.64%         | 98.58%       |
| granitanet | 99.84%       | 99.73%         | 99.63%       |
| regla      | 99.86%       | 99.53%         | 99.81%       |
| iformer    | 99.86%       | 98.92%         | 99.69%       |
| lsnet      | 99.82%       | 99.66%         | 99.63%       |
