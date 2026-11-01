# Held-out-writer accuracy vs. number of HOMUS training writers

granitanet, 15 epochs, seed 42; same synthetic + crop data in every row; 10 held-out writers, 1520 strokes, pen 0.085 at natural size.

|   Training writers |   Hand train samples | Held-out top-1   | Macro F1   | Per-writer mean   | Worst writer   |
|--------------------|----------------------|------------------|------------|-------------------|----------------|
|                  0 |                    0 | 22.6%            | 27.7%      | 22.6%             | 9.2%           |
|                  5 |                 1520 | 62.4%            | 65.6%      | 62.4%             | 39.5%          |
|                 10 |                 3040 | 69.1%            | 71.7%      | 69.1%             | 53.9%          |
|                 20 |                 6080 | 75.8%            | 78.2%      | 75.8%             | 63.2%          |
|                 40 |                12158 | 81.1%            | 83.2%      | 81.1%             | 71.7%          |
|                 80 |                24311 | 84.5%            | 86.2%      | 84.5%             | 79.6%          |

Weakest classes at 80 writers: restQuarter 55%, noteWhole 60%, note16th 66%, rest32nd 68%, note32nd 69%, rest16th 70%, timeSig3over4 70%, note64th 72%
