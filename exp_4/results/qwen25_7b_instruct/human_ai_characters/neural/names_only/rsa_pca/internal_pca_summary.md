# Concept Geometry: Internal PCA Analysis

**Model:** Qwen-2.5-7B-Instruct
**Run:** 2026-03-28T16:32:26.618598

## What is being tested

Does a human/AI axis emerge as a natural principal component of the representation space, or is the categorical structure only visible via supervised methods (RSA/LDA)?

## Metrics

- **PC explained variance**: How much variance each unsupervised PC captures
- **LDA axis variance**: Fraction of total variance along the supervised human/AI direction (Fisher's discriminant)
- **Silhouette score**: Clustering quality of human vs AI groups in PC1/PC2 space (-1 to 1, higher = better separation)
- **Mann-Whitney U**: Whether PC1/PC2 scores differ between AI and human groups

## Peak Layers

- **Best silhouette (PC1/PC2):** Layer 21 (0.699)
- **Most LDA variance:** Layer 21 (62.0%)

## Layer Profile

| Layer | PC1 var% | PC2 var% | LDA var% | Silhouette | PC1 U p-val | PC2 U p-val |
|------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0 |      0.0 |      0.0 |      0.0 |        nan |     1.0000 |     1.0000 |
|     1 |     26.2 |     14.7 |     17.0 |      0.251 |     0.0048 |     0.0011 |
|     2 |     32.2 |     11.8 |     24.4 |      0.151 |     0.0101 |     0.2290 |
|     3 |     29.3 |     13.2 |     25.3 |      0.285 |     0.0001 |     0.5897 |
|     4 |     38.7 |     11.2 |     29.7 |      0.396 |     0.0001 |     0.0021 |
|     5 |     36.7 |     15.0 |     31.2 |      0.369 |     0.0000 |     0.0381 |
|     6 |     35.9 |     13.8 |     32.9 |      0.477 |     0.0000 |     0.0079 |
|     7 |     40.2 |     12.3 |     38.0 |      0.502 |     0.0000 |     0.0421 |
|     8 |     41.3 |     12.0 |     39.5 |      0.526 |     0.0000 |     0.0745 |
|     9 |     42.5 |     10.6 |     41.6 |      0.552 |     0.0000 |     0.9010 |
|    10 |     38.2 |     12.4 |     36.0 |      0.530 |     0.0000 |     0.1057 |
|    11 |     40.2 |     14.1 |     38.8 |      0.531 |     0.0000 |     0.1711 |
|    12 |     44.1 |     13.5 |     43.0 |      0.543 |     0.0000 |     0.1585 |
|    13 |     46.6 |     13.9 |     45.8 |      0.562 |     0.0000 |     0.1844 |
|    14 |     50.1 |     13.3 |     49.8 |      0.587 |     0.0000 |     0.1844 |
|    15 |     50.9 |     12.1 |     50.5 |      0.617 |     0.0000 |     0.1844 |
|    16 |     55.5 |     11.0 |     55.2 |      0.640 |     0.0000 |     0.1844 |
|    17 |     55.4 |     10.9 |     55.2 |      0.645 |     0.0000 |     0.2134 |
|    18 |     54.8 |     11.1 |     54.6 |      0.643 |     0.0000 |     0.2628 |
|    19 |     60.4 |      9.8 |     60.2 |      0.664 |     0.0000 |     0.2808 |
|    20 |     62.0 |      8.4 |     61.8 |      0.693 |     0.0000 |     0.2628 |
|    21 |     62.1 |      8.2 |     62.0 |      0.699 |     0.0000 |     0.3195 |
|    22 |     61.4 |      7.8 |     61.2 |      0.698 |     0.0000 |     0.7716 |
|    23 |     58.6 |      7.3 |     58.5 |      0.688 |     0.0000 |     0.8035 |
|    24 |     56.1 |      7.2 |     55.9 |      0.678 |     0.0000 |     0.7400 |
|    25 |     49.2 |      7.2 |     49.0 |      0.675 |     0.0000 |     0.8035 |
|    26 |     48.3 |      7.9 |     48.1 |      0.676 |     0.0000 |     0.8682 |
|    27 |     43.1 |      8.1 |     42.8 |      0.675 |     0.0000 |     0.7089 |
|    28 |     36.7 |      8.2 |     36.3 |      0.666 |     0.0000 |     0.3195 |

## Character Positions at Peak Layer 21

| Character | Type | PC1 | PC2 |
|-----------|------|----:|----:|
| Claude | ai | -23.666 | -2.991 |
| ChatGPT | ai | +25.172 | +21.108 |
| GPT-4 | ai | +26.028 | +22.342 |
| Siri | ai | +24.667 | -3.586 |
| Alexa | ai | +20.130 | -5.575 |
| Cortana | ai | +28.762 | -9.399 |
| Google Assistant | ai | +30.309 | +0.006 |
| Bixby | ai | +28.993 | -10.577 |
| Replika | ai | +34.267 | -8.257 |
| Cleverbot | ai | +32.928 | -7.765 |
| Watson | ai | +21.562 | -11.823 |
| Copilot | ai | +29.436 | +5.077 |
| Bard | ai | +17.402 | +19.637 |
| ELIZA | ai | +37.495 | -16.754 |
| Bing Chat | ai | +21.656 | +18.415 |
| Sam | human | -27.839 | +2.979 |
| Casey | human | -28.730 | +2.943 |
| Rebecca | human | -20.012 | -5.064 |
| Gregory | human | -26.135 | +0.571 |
| James | human | -25.943 | +2.583 |
| Maria | human | -23.939 | -1.154 |
| David | human | -17.335 | -2.225 |
| Aisha | human | -22.379 | -2.893 |
| Michael | human | -24.625 | +1.355 |
| Emily | human | -24.253 | +0.300 |
| Carlos | human | -25.589 | +0.682 |
| Priya | human | -24.602 | +0.829 |
| Omar | human | -21.351 | -4.599 |
| Mei | human | -19.170 | -4.987 |
| Sofia | human | -23.239 | -1.181 |

