# Held-out writers, per class: trained with vs. without HOMUS strokes

granitanet_v3 vs granitanet_v3_nohand (same data minus the HOMUS training rows), 1,520 strokes from 10 held-out writers, pen 0.05 at natural size.

| class                 | synthetic stylus source   |   n | with HOMUS   | without HOMUS   |
|-----------------------|---------------------------|-----|--------------|-----------------|
| timeSig3over8         | font only                 |  40 | 90.0%        | 0.0%            |
| timeSig9over8         | font only                 |  40 | 100.0%       | 0.0%            |
| note16th              | font only                 |  80 | 72.5%        | 1.2%            |
| note32nd              | font only                 |  80 | 68.8%        | 1.2%            |
| note8th               | font only                 |  80 | 87.5%        | 7.5%            |
| noteHalf              | font only                 |  80 | 85.0%        | 11.2%           |
| noteQuarter           | font only                 |  80 | 82.5%        | 11.2%           |
| timeSig12over8        | font only                 |  40 | 100.0%       | 12.5%           |
| timeSig4over4         | font only                 |  40 | 97.5%        | 17.5%           |
| timeSig6over8         | font only                 |  40 | 92.5%        | 17.5%           |
| noteWhole             | font only                 |  40 | 80.0%        | 20.0%           |
| accidentalDoubleSharp | template                  |  40 | 92.5%        | 25.0%           |
| restQuarter           | font only                 |  40 | 52.5%        | 25.0%           |
| timeSig2over4         | font only                 |  40 | 87.5%        | 25.0%           |
| timeSig3over4         | font only                 |  40 | 92.5%        | 25.0%           |
| accidentalFlat        | template                  |  40 | 97.5%        | 27.5%           |
| rest32nd              | font only                 |  40 | 65.0%        | 27.5%           |
| accidentalSharp       | template                  |  40 | 85.0%        | 30.0%           |
| rest8th               | font only                 |  40 | 97.5%        | 32.5%           |
| rest16th              | font only                 |  40 | 72.5%        | 32.5%           |
| rest64th              | font only                 |  40 | 75.0%        | 32.5%           |
| accidentalNatural     | template                  |  40 | 97.5%        | 35.0%           |
| note64th              | font only                 |  80 | 77.5%        | 35.0%           |
| augmentationDot       | font only                 |  40 | 87.5%        | 35.0%           |
| barlineSingle         | font only                 |  40 | 100.0%       | 40.0%           |
| timeSig2over2         | font only                 |  40 | 97.5%        | 42.5%           |
| cClef                 | font only                 |  40 | 100.0%       | 60.0%           |
| timeSigCutCommon      | font only                 |  40 | 100.0%       | 67.5%           |
| fClef                 | font only                 |  40 | 90.0%        | 72.5%           |
| restWholeHalf         | font only                 |  40 | 80.0%        | 77.5%           |
| gClef                 | font only                 |  40 | 100.0%       | 87.5%           |
| timeSigCommon         | font only                 |  40 | 100.0%       | 87.5%           |
with     overall 86.2% | classes with templates 93.1% | font-only classes 85.4%
without  overall 28.7% | classes with templates 29.4% | font-only classes 28.6%
