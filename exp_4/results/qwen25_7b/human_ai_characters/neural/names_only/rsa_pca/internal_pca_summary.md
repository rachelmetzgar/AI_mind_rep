# Concept Geometry: Internal PCA Analysis

**Model:** Qwen-2.5-7B (Base)
**Run:** 2026-03-29T12:43:06.081407

## What is being tested

Does a human/AI axis emerge as a natural principal component of the representation space, or is the categorical structure only visible via supervised methods (RSA/LDA)?

## Metrics

- **PC explained variance**: How much variance each unsupervised PC captures
- **LDA axis variance**: Fraction of total variance along the supervised human/AI direction (Fisher's discriminant)
- **Silhouette score**: Clustering quality of human vs AI groups in PC1/PC2 space (-1 to 1, higher = better separation)
- **Mann-Whitney U**: Whether PC1/PC2 scores differ between AI and human groups

## Peak Layers

- **Best silhouette (PC1/PC2):** Layer 9 (0.628)
- **Most LDA variance:** Layer 12 (43.6%)

## Layer Profile

| Layer | PC1 var% | PC2 var% | LDA var% | Silhouette | PC1 U p-val | PC2 U p-val |
|------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0 |      0.0 |      0.0 |      0.0 |        nan |     1.0000 |     1.0000 |
|     1 |     20.0 |      9.0 |     13.1 |      0.248 |     0.0202 |     0.0021 |
|     2 |     24.4 |     17.4 |     17.8 |      0.270 |     0.2808 |     0.0005 |
|     3 |     23.1 |     14.3 |     20.3 |      0.322 |     0.0001 |     0.0161 |
|     4 |     27.6 |     11.2 |     26.2 |      0.444 |     0.0000 |     0.0202 |
|     5 |     30.4 |     10.3 |     29.4 |      0.542 |     0.0000 |     0.0079 |
|     6 |     31.8 |     11.6 |     30.6 |      0.544 |     0.0000 |     0.0042 |
|     7 |     38.0 |      9.4 |     37.0 |      0.567 |     0.0000 |     0.0062 |
|     8 |     40.0 |      8.2 |     39.3 |      0.590 |     0.0000 |     0.2134 |
|     9 |     41.5 |      7.6 |     41.0 |      0.628 |     0.0000 |     0.1711 |
|    10 |     41.4 |     11.4 |     40.3 |      0.595 |     0.0000 |     0.0745 |
|    11 |     42.1 |     12.4 |     41.3 |      0.588 |     0.0000 |     0.1844 |
|    12 |     44.0 |     13.0 |     43.6 |      0.593 |     0.0000 |     0.2628 |
|    13 |     43.0 |     13.3 |     42.5 |      0.591 |     0.0000 |     0.3615 |
|    14 |     41.1 |     14.4 |     40.7 |      0.583 |     0.0000 |     0.4553 |
|    15 |     41.4 |     15.0 |     40.8 |      0.584 |     0.0000 |     0.2290 |
|    16 |     41.9 |     15.0 |     41.4 |      0.583 |     0.0000 |     0.2290 |
|    17 |     41.8 |     16.1 |     41.5 |      0.574 |     0.0000 |     0.3615 |
|    18 |     43.8 |     15.8 |     43.6 |      0.575 |     0.0000 |     0.5069 |
|    19 |     43.6 |     16.8 |     43.4 |      0.570 |     0.0000 |     0.6783 |
|    20 |     42.5 |     16.9 |     42.4 |      0.583 |     0.0000 |     0.8035 |
|    21 |     43.1 |     17.0 |     42.9 |      0.593 |     0.0000 |     0.9669 |
|    22 |     42.7 |     16.5 |     42.4 |      0.598 |     0.0000 |     0.7716 |
|    23 |     40.3 |     16.2 |     40.0 |      0.593 |     0.0000 |     0.6783 |
|    24 |     38.2 |     15.2 |     37.8 |      0.596 |     0.0000 |     0.7400 |
|    25 |     36.9 |     16.1 |     36.6 |      0.601 |     0.0000 |     0.9669 |
|    26 |     36.8 |     15.5 |     36.5 |      0.597 |     0.0000 |     1.0000 |
|    27 |     33.3 |     14.8 |     33.0 |      0.597 |     0.0000 |     1.0000 |
|    28 |     36.0 |     18.9 |     35.5 |      0.500 |     0.0000 |     0.6783 |

## Character Positions at Peak Layer 9

| Character | Type | PC1 | PC2 |
|-----------|------|----:|----:|
| Claude | ai | -7.872 | -2.881 |
| ChatGPT | ai | +14.450 | +8.937 |
| GPT-4 | ai | +16.909 | +16.735 |
| Siri | ai | +13.206 | -7.452 |
| Alexa | ai | +5.743 | -6.840 |
| Cortana | ai | +15.643 | -2.236 |
| Google Assistant | ai | +19.446 | +1.265 |
| Bixby | ai | +8.344 | -7.854 |
| Replika | ai | +13.059 | -0.702 |
| Cleverbot | ai | +15.688 | +2.576 |
| Watson | ai | +4.132 | -9.868 |
| Copilot | ai | +16.770 | -3.614 |
| Bard | ai | +5.545 | -8.193 |
| ELIZA | ai | +8.672 | +4.562 |
| Bing Chat | ai | +18.641 | +1.625 |
| Sam | human | -12.386 | +1.636 |
| Casey | human | -11.394 | -1.097 |
| Rebecca | human | -10.025 | -1.519 |
| Gregory | human | -8.643 | -3.008 |
| James | human | -14.221 | +2.345 |
| Maria | human | -13.370 | +1.678 |
| David | human | -10.571 | +0.141 |
| Aisha | human | -9.126 | +2.193 |
| Michael | human | -14.507 | +1.679 |
| Emily | human | -12.859 | +1.796 |
| Carlos | human | -13.164 | +1.012 |
| Priya | human | -9.167 | +6.769 |
| Omar | human | -11.842 | +1.797 |
| Mei | human | -8.487 | -2.361 |
| Sofia | human | -8.615 | +0.880 |

