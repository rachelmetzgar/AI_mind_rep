# Concept Geometry: Internal PCA Analysis

**Model:** Gemma-2-2B-IT
**Run:** 2026-03-29T12:36:52.121901

## What is being tested

Does a human/AI axis emerge as a natural principal component of the representation space, or is the categorical structure only visible via supervised methods (RSA/LDA)?

## Metrics

- **PC explained variance**: How much variance each unsupervised PC captures
- **LDA axis variance**: Fraction of total variance along the supervised human/AI direction (Fisher's discriminant)
- **Silhouette score**: Clustering quality of human vs AI groups in PC1/PC2 space (-1 to 1, higher = better separation)
- **Mann-Whitney U**: Whether PC1/PC2 scores differ between AI and human groups

## Peak Layers

- **Best silhouette (PC1/PC2):** Layer 26 (0.714)
- **Most LDA variance:** Layer 16 (70.3%)

## Layer Profile

| Layer | PC1 var% | PC2 var% | LDA var% | Silhouette | PC1 U p-val | PC2 U p-val |
|------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0 |      0.0 |      0.0 |      0.0 |        nan |     1.0000 |     1.0000 |
|     1 |     35.8 |     12.3 |     31.0 |      0.412 |     0.0000 |     0.0008 |
|     2 |     35.5 |      8.2 |     31.1 |      0.336 |     0.0000 |     0.8035 |
|     3 |     40.8 |     10.4 |     36.9 |      0.373 |     0.0000 |     0.2290 |
|     4 |     38.8 |     10.6 |     36.3 |      0.547 |     0.0000 |     0.0011 |
|     5 |     46.0 |     12.4 |     45.0 |      0.595 |     0.0000 |     0.0144 |
|     6 |     57.0 |      8.3 |     56.3 |      0.648 |     0.0000 |     0.0225 |
|     7 |     63.5 |      6.4 |     63.0 |      0.657 |     0.0000 |     0.2455 |
|     8 |     60.8 |      7.2 |     60.3 |      0.680 |     0.0000 |     0.9669 |
|     9 |     65.4 |      7.8 |     65.1 |      0.705 |     0.0000 |     0.3195 |
|    10 |     62.2 |      7.8 |     61.8 |      0.700 |     0.0000 |     0.8682 |
|    11 |     65.9 |      8.3 |     65.6 |      0.692 |     0.0000 |     1.0000 |
|    12 |     65.7 |      8.7 |     65.2 |      0.660 |     0.0000 |     0.5614 |
|    13 |     60.6 |     13.4 |     60.1 |      0.613 |     0.0000 |     0.5069 |
|    14 |     69.4 |      9.6 |     69.1 |      0.697 |     0.0000 |     0.7089 |
|    15 |     69.3 |      9.4 |     69.1 |      0.697 |     0.0000 |     0.9339 |
|    16 |     70.5 |      9.3 |     70.3 |      0.701 |     0.0000 |     0.9010 |
|    17 |     70.5 |      8.6 |     70.3 |      0.709 |     0.0000 |     0.4306 |
|    18 |     67.1 |      8.6 |     66.8 |      0.700 |     0.0000 |     0.5338 |
|    19 |     66.5 |      9.3 |     66.1 |      0.682 |     0.0000 |     0.6783 |
|    20 |     66.4 |      9.2 |     65.9 |      0.670 |     0.0000 |     0.8357 |
|    21 |     63.6 |      9.6 |     63.2 |      0.666 |     0.0000 |     0.9339 |
|    22 |     62.0 |     10.5 |     61.5 |      0.655 |     0.0000 |     1.0000 |
|    23 |     55.9 |     10.3 |     55.4 |      0.654 |     0.0000 |     0.8682 |
|    24 |     55.8 |     10.3 |     55.4 |      0.657 |     0.0000 |     0.8357 |
|    25 |     55.0 |     10.1 |     54.5 |      0.650 |     0.0000 |     0.8682 |
|    26 |     46.7 |      8.0 |     46.3 |      0.714 |     0.0000 |     0.0564 |

## Character Positions at Peak Layer 26

| Character | Type | PC1 | PC2 |
|-----------|------|----:|----:|
| Claude | ai | -5.941 | +5.248 |
| ChatGPT | ai | -43.006 | -10.869 |
| GPT-4 | ai | -35.959 | -12.021 |
| Siri | ai | -29.818 | +15.383 |
| Alexa | ai | -14.572 | +13.811 |
| Cortana | ai | -26.345 | +14.798 |
| Google Assistant | ai | -19.190 | +12.015 |
| Bixby | ai | -21.196 | +22.431 |
| Replika | ai | -29.525 | +3.141 |
| Cleverbot | ai | -42.584 | -11.760 |
| Watson | ai | -9.765 | +2.288 |
| Copilot | ai | -19.143 | +18.807 |
| Bard | ai | -34.838 | -0.795 |
| ELIZA | ai | -51.837 | -43.319 |
| Bing Chat | ai | -28.167 | +6.551 |
| Sam | human | +27.954 | -2.059 |
| Casey | human | +29.644 | -5.020 |
| Rebecca | human | +27.276 | -5.667 |
| Gregory | human | +29.777 | -6.449 |
| James | human | +33.246 | -3.396 |
| Maria | human | +28.273 | -3.174 |
| David | human | +32.206 | -4.249 |
| Aisha | human | +25.580 | -2.445 |
| Michael | human | +33.123 | -3.378 |
| Emily | human | +30.648 | -3.726 |
| Carlos | human | +26.426 | -2.019 |
| Priya | human | +24.320 | +1.693 |
| Omar | human | +24.960 | -4.531 |
| Mei | human | +11.677 | +10.191 |
| Sofia | human | +26.777 | -1.481 |

