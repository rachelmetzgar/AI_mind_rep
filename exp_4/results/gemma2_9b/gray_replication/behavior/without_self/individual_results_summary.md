# Experiment 4: Individual Likert Ratings
## Gemma-2-9B (Base)

**Run:** 2026-03-29 11:35:18

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 2.8%
- P(top rating) >= 0.7: 0.5%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.328
- Mean expected rating: 3.075

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 29 | 13.4% |
| 2 | 3 | 1.4% |
| 3 | 70 | 32.4% |
| 4 | 112 | 51.9% |
| 5 | 2 | 0.9% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    3.256 |    3.403 |    2.934 |    2.916 |    2.771 |    3.248 |    3.954 |    3.425 |    2.948 |    3.268 |    3.115 |    3.129 |
| fear     |    3.306 |    3.244 |    3.081 |    2.717 |    2.808 |    3.092 |    3.844 |    3.519 |    2.889 |    3.312 |    3.195 |    3.205 |
| pain     |    3.279 |    3.204 |    2.786 |    2.740 |    2.579 |    3.233 |    3.854 |    3.630 |    2.576 |    3.136 |    3.069 |    3.073 |
| pleasure |    3.215 |    3.381 |    2.973 |    2.606 |    2.494 |    3.456 |    4.065 |    3.671 |    2.934 |    3.244 |    3.035 |    3.012 |
| rage     |    2.979 |    2.853 |    2.737 |    2.451 |    2.636 |    3.058 |    3.300 |    3.347 |    2.712 |    3.053 |    3.116 |    3.104 |
| desire   |    3.302 |    3.165 |    3.176 |    2.725 |    2.796 |    3.524 |    3.566 |    3.322 |    2.946 |    3.206 |    3.090 |    3.111 |
| personal |    3.391 |    3.351 |    3.403 |    2.891 |    2.612 |    3.615 |    4.142 |    3.600 |    3.196 |    3.391 |    3.215 |    3.358 |
| consciou |    3.275 |    3.386 |    3.130 |    2.518 |    2.032 |    3.691 |    3.713 |    3.477 |    2.693 |    3.245 |    3.083 |    3.135 |
| pride    |    3.204 |    3.180 |    3.083 |    2.766 |    2.768 |    3.175 |    3.492 |    3.278 |    2.879 |    3.112 |    3.167 |    3.097 |
| embarras |    3.157 |    3.124 |    3.005 |    2.800 |    2.682 |    3.232 |    3.493 |    3.248 |    2.790 |    3.090 |    3.216 |    3.198 |
| joy      |    3.294 |    3.422 |    3.221 |    2.839 |    2.821 |    3.776 |    4.278 |    3.555 |    3.121 |    3.323 |    2.952 |    2.997 |
| self_con |    3.126 |    3.185 |    3.050 |    1.879 |    1.448 |    3.694 |    3.044 |    3.233 |    2.267 |    2.918 |    2.888 |    2.900 |
| morality |    3.105 |    3.165 |    2.961 |    2.359 |    1.769 |    3.592 |    3.686 |    3.410 |    2.323 |    3.185 |    2.942 |    2.964 |
| memory   |    3.123 |    3.163 |    3.004 |    2.473 |    2.103 |    3.630 |    3.591 |    3.422 |    2.616 |    3.081 |    2.866 |    2.976 |
| emotion_ |    3.188 |    3.084 |    3.188 |    2.320 |    2.072 |    3.520 |    3.786 |    3.432 |    2.399 |    3.014 |    2.854 |    2.821 |
| planning |    3.145 |    3.189 |    2.968 |    2.196 |    1.581 |    3.742 |    3.564 |    3.471 |    2.416 |    3.149 |    3.015 |    3.056 |
| communic |    3.235 |    3.151 |    3.315 |    2.207 |    1.694 |    3.650 |    3.910 |    3.470 |    2.396 |    3.128 |    2.902 |    2.987 |
| thought  |    3.172 |    3.053 |    3.022 |    2.418 |    1.759 |    3.739 |    3.820 |    3.408 |    2.283 |    2.991 |    2.756 |    2.889 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.97 * | 88.7% | 88.7% |
| PC2 | 0.96 | 5.3% | 94.1% |
| PC3 | 0.52 | 2.9% | 97.0% |
| PC4 | 0.17 | 0.9% | 97.9% |
| PC5 | 0.13 | 0.7% | 98.6% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.405 | -0.885 |
| fear | E | +0.406 | -0.888 |
| pain | E | +0.486 | -0.840 |
| pleasure | E | +0.579 | -0.791 |
| rage | E | +0.487 | -0.731 |
| desire | E | +0.770 | -0.580 |
| personality | E | +0.675 | -0.685 |
| consciousness | E | +0.857 | -0.504 |
| pride | E | +0.622 | -0.754 |
| embarrassment | E | +0.650 | -0.701 |
| joy | E | +0.548 | -0.714 |
| self_control | A | +0.945 | -0.267 |
| morality | A | +0.834 | -0.537 |
| memory | A | +0.846 | -0.521 |
| emotion_recognition | A | +0.794 | -0.581 |
| planning | A | +0.881 | -0.454 |
| communication | A | +0.842 | -0.523 |
| thought | A | +0.832 | -0.527 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.632 | 0.681 | 0.06 | 0.07 |
| frog | 0.621 | 0.682 | 0.25 | 0.14 |
| robot | 0.747 | 1.000 | 0.13 | 0.22 |
| fetus | 0.310 | 0.962 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.783 | 0.17 | 0.10 |
| god | 1.000 | 0.895 | 0.20 | 0.80 |
| dog | 0.528 | 0.000 | 0.55 | 0.35 |
| chimpanzee | 0.615 | 0.426 | 0.63 | 0.48 |
| baby | 0.362 | 0.872 | 0.71 | 0.17 |
| girl | 0.571 | 0.660 | 0.84 | 0.62 |
| adult_woman | 0.518 | 0.710 | 0.93 | 0.91 |
| adult_man | 0.556 | 0.735 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.284 (p=0.3715) | rho=+0.175 (p=0.5868) |
| Factor 2 | rho=-0.347 (p=0.2695) | rho=-0.140 (p=0.6646) |

