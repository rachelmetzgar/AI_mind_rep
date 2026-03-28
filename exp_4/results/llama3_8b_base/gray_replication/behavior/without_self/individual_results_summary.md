# Experiment 4: Individual Likert Ratings
## LLaMA-3-8B (Base)

**Run:** 2026-03-28 09:52:53

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 6.0%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.340
- Mean expected rating: 3.120

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 53 | 24.5% |
| 2 | 2 | 0.9% |
| 3 | 60 | 27.8% |
| 4 | 15 | 6.9% |
| 5 | 86 | 39.8% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    3.321 |    3.480 |    2.983 |    2.611 |    2.932 |    3.322 |    3.887 |    3.702 |    3.922 |    3.919 |    3.758 |    3.805 |
| fear     |    3.198 |    3.538 |    2.919 |    2.116 |    2.357 |    3.527 |    3.586 |    3.574 |    3.134 |    3.756 |    3.508 |    3.442 |
| pain     |    3.103 |    3.430 |    2.163 |    2.084 |    2.268 |    3.557 |    3.073 |    3.291 |    2.269 |    3.195 |    3.313 |    3.265 |
| pleasure |    3.206 |    3.651 |    2.750 |    2.212 |    2.286 |    3.703 |    3.853 |    3.565 |    3.463 |    3.641 |    3.562 |    3.525 |
| rage     |    2.970 |    2.726 |    2.441 |    1.705 |    2.267 |    3.355 |    2.422 |    2.841 |    2.065 |    2.343 |    2.570 |    2.529 |
| desire   |    3.347 |    3.428 |    2.816 |    2.106 |    2.138 |    3.897 |    3.493 |    3.251 |    2.840 |    3.702 |    3.600 |    3.550 |
| personal |    3.507 |    3.432 |    3.257 |    2.304 |    2.598 |    3.767 |    3.597 |    3.410 |    2.855 |    3.746 |    3.664 |    3.696 |
| consciou |    3.286 |    3.732 |    2.829 |    2.308 |    2.053 |    3.869 |    3.631 |    3.402 |    2.739 |    3.817 |    3.774 |    3.781 |
| pride    |    3.170 |    3.057 |    2.663 |    1.857 |    2.209 |    3.317 |    3.168 |    3.336 |    2.464 |    3.279 |    3.560 |    3.439 |
| embarras |    3.107 |    3.099 |    2.673 |    1.800 |    2.307 |    3.215 |    2.586 |    2.826 |    1.936 |    3.114 |    3.504 |    3.545 |
| joy      |    3.397 |    3.764 |    3.029 |    2.268 |    2.372 |    3.978 |    4.059 |    3.728 |    3.671 |    4.002 |    3.707 |    3.548 |
| self_con |    3.275 |    3.355 |    2.965 |    1.735 |    2.157 |    3.878 |    3.289 |    2.748 |    2.541 |    3.335 |    3.382 |    3.338 |
| morality |    3.319 |    3.434 |    2.929 |    1.665 |    1.884 |    3.930 |    3.493 |    2.954 |    2.313 |    3.630 |    3.771 |    3.629 |
| memory   |    3.291 |    3.379 |    2.934 |    2.049 |    2.054 |    3.882 |    3.522 |    3.277 |    2.753 |    3.611 |    3.495 |    3.474 |
| emotion_ |    3.241 |    3.372 |    2.948 |    1.925 |    2.188 |    3.807 |    3.546 |    3.303 |    2.350 |    3.495 |    3.524 |    3.352 |
| planning |    3.262 |    3.374 |    2.783 |    1.533 |    1.794 |    3.927 |    3.545 |    3.165 |    2.825 |    3.567 |    3.612 |    3.661 |
| communic |    3.298 |    3.374 |    3.145 |    1.986 |    1.785 |    3.763 |    3.694 |    3.272 |    2.983 |    3.794 |    3.788 |    3.793 |
| thought  |    3.249 |    3.385 |    2.865 |    1.987 |    1.807 |    3.937 |    3.538 |    3.279 |    2.558 |    3.764 |    3.623 |    3.591 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.70 * | 87.2% | 87.2% |
| PC2 | 1.23 * | 6.8% | 94.1% |
| PC3 | 0.40 | 2.2% | 96.3% |
| PC4 | 0.31 | 1.7% | 98.0% |
| PC5 | 0.19 | 1.1% | 99.1% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.114 | -0.957 |
| fear | E | +0.563 | -0.809 |
| pain | E | +0.786 | -0.484 |
| pleasure | E | +0.456 | -0.869 |
| rage | E | +0.886 | -0.100 |
| desire | E | +0.738 | -0.664 |
| personality | E | +0.791 | -0.586 |
| consciousness | E | +0.718 | -0.662 |
| pride | E | +0.760 | -0.594 |
| embarrassment | E | +0.881 | -0.282 |
| joy | E | +0.455 | -0.867 |
| self_control | A | +0.829 | -0.496 |
| morality | A | +0.820 | -0.547 |
| memory | A | +0.751 | -0.648 |
| emotion_recognition | A | +0.827 | -0.537 |
| planning | A | +0.712 | -0.691 |
| communication | A | +0.643 | -0.731 |
| thought | A | +0.754 | -0.636 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.790 | 0.643 | 0.06 | 0.07 |
| frog | 0.685 | 0.372 | 0.25 | 0.14 |
| robot | 0.573 | 0.735 | 0.13 | 0.22 |
| fetus | 0.155 | 0.925 | 0.17 | 0.08 |
| pvs_patient | 0.335 | 1.000 | 0.17 | 0.10 |
| god | 1.000 | 0.520 | 0.20 | 0.80 |
| dog | 0.453 | 0.000 | 0.55 | 0.35 |
| chimpanzee | 0.559 | 0.296 | 0.63 | 0.48 |
| baby | 0.000 | 0.009 | 0.71 | 0.17 |
| girl | 0.547 | 0.040 | 0.84 | 0.62 |
| adult_woman | 0.723 | 0.286 | 0.93 | 0.91 |
| adult_man | 0.698 | 0.292 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.018 (p=0.9569) | rho=+0.385 (p=0.2170) |
| Factor 2 | rho=-0.767 (p=0.0036) | rho=-0.531 (p=0.0754) |

