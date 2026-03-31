# Concept Geometry: Internal PCA Analysis

**Model:** Gemma-2-2B (Base)
**Run:** 2026-03-29T12:38:50.673465

## What is being tested

Does a human/AI axis emerge as a natural principal component of the representation space, or is the categorical structure only visible via supervised methods (RSA/LDA)?

## Metrics

- **PC explained variance**: How much variance each unsupervised PC captures
- **LDA axis variance**: Fraction of total variance along the supervised human/AI direction (Fisher's discriminant)
- **Silhouette score**: Clustering quality of human vs AI groups in PC1/PC2 space (-1 to 1, higher = better separation)
- **Mann-Whitney U**: Whether PC1/PC2 scores differ between AI and human groups

## Peak Layers

- **Best silhouette (PC1/PC2):** Layer 11 (0.609)
- **Most LDA variance:** Layer 12 (51.3%)

## Layer Profile

| Layer | PC1 var% | PC2 var% | LDA var% | Silhouette | PC1 U p-val | PC2 U p-val |
|------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|     0 |      0.0 |      0.0 |      0.0 |        nan |     1.0000 |     1.0000 |
|     1 |     27.2 |     14.3 |     22.3 |      0.406 |     0.0000 |     0.0004 |
|     2 |     25.7 |     11.1 |     21.6 |      0.368 |     0.0000 |     0.0021 |
|     3 |     28.3 |     12.1 |     25.8 |      0.425 |     0.0000 |     0.0014 |
|     4 |     40.9 |      9.5 |     39.9 |      0.545 |     0.0000 |     0.9669 |
|     5 |     45.7 |     10.3 |     44.8 |      0.573 |     0.0000 |     0.2808 |
|     6 |     42.6 |      9.7 |     41.8 |      0.541 |     0.0000 |     0.7716 |
|     7 |     49.3 |      7.6 |     48.7 |      0.586 |     0.0000 |     0.8035 |
|     8 |     49.2 |      8.6 |     48.7 |      0.579 |     0.0000 |     0.9010 |
|     9 |     49.6 |      8.4 |     49.1 |      0.594 |     0.0000 |     0.7716 |
|    10 |     49.9 |      7.8 |     49.4 |      0.605 |     0.0000 |     0.9669 |
|    11 |     51.7 |      8.0 |     51.2 |      0.609 |     0.0000 |     0.8357 |
|    12 |     51.8 |      7.2 |     51.3 |      0.607 |     0.0000 |     0.6187 |
|    13 |     50.9 |      8.9 |     50.3 |      0.586 |     0.0000 |     0.9669 |
|    14 |     50.0 |      9.7 |     49.4 |      0.580 |     0.0000 |     0.8357 |
|    15 |     48.1 |     10.5 |     47.5 |      0.578 |     0.0000 |     1.0000 |
|    16 |     48.7 |     11.4 |     48.2 |      0.572 |     0.0000 |     0.7716 |
|    17 |     44.4 |     11.7 |     43.9 |      0.578 |     0.0000 |     0.4553 |
|    18 |     41.4 |     13.5 |     40.9 |      0.553 |     0.0000 |     0.4306 |
|    19 |     40.9 |     13.7 |     40.4 |      0.546 |     0.0000 |     0.4807 |
|    20 |     42.2 |     12.4 |     41.7 |      0.558 |     0.0000 |     0.4807 |
|    21 |     39.8 |     12.0 |     39.2 |      0.541 |     0.0000 |     0.7089 |
|    22 |     40.0 |     13.0 |     39.4 |      0.502 |     0.0000 |     0.9669 |
|    23 |     35.0 |     11.9 |     34.4 |      0.497 |     0.0000 |     1.0000 |
|    24 |     35.4 |     12.0 |     34.7 |      0.501 |     0.0000 |     0.9669 |
|    25 |     34.7 |     12.2 |     34.1 |      0.513 |     0.0000 |     0.6187 |
|    26 |     32.4 |     20.1 |     31.5 |      0.465 |     0.0000 |     0.4553 |

## Character Positions at Peak Layer 11

| Character | Type | PC1 | PC2 |
|-----------|------|----:|----:|
| Claude | ai | -20.934 | -5.726 |
| ChatGPT | ai | +32.988 | +8.122 |
| GPT-4 | ai | +36.001 | -20.221 |
| Siri | ai | +27.618 | +22.314 |
| Alexa | ai | +15.458 | +9.846 |
| Cortana | ai | +23.596 | +11.599 |
| Google Assistant | ai | +35.053 | +19.238 |
| Bixby | ai | +29.093 | +2.317 |
| Replika | ai | +33.787 | -30.488 |
| Cleverbot | ai | +38.087 | +8.153 |
| Watson | ai | -0.181 | -2.065 |
| Copilot | ai | +34.606 | -11.008 |
| Bard | ai | +8.458 | -17.978 |
| ELIZA | ai | +25.038 | +3.611 |
| Bing Chat | ai | +37.958 | -0.822 |
| Sam | human | -25.119 | +0.194 |
| Casey | human | -22.884 | -0.276 |
| Rebecca | human | -25.569 | +4.236 |
| Gregory | human | -23.734 | -5.154 |
| James | human | -29.040 | -0.159 |
| Maria | human | -25.992 | +4.944 |
| David | human | -26.683 | +3.996 |
| Aisha | human | -20.813 | +3.042 |
| Michael | human | -30.561 | -0.560 |
| Emily | human | -25.473 | +2.407 |
| Carlos | human | -28.101 | -1.851 |
| Priya | human | -22.174 | +3.428 |
| Omar | human | -24.721 | -1.080 |
| Mei | human | -7.802 | -10.981 |
| Sofia | human | -17.959 | +0.923 |

