# Experiment 4: Individual Likert Ratings
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 11:34:07

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 0.0%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.281
- Mean expected rating: 2.855

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 52 | 24.1% |
| 2 | 4 | 1.9% |
| 3 | 150 | 69.4% |
| 4 | 9 | 4.2% |
| 5 | 1 | 0.5% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    2.701 |    3.010 |    2.918 |    2.709 |    2.513 |    2.848 |    3.300 |    3.086 |    2.909 |    2.978 |    2.885 |    2.941 |
| fear     |    2.699 |    2.961 |    2.949 |    2.759 |    2.512 |    2.810 |    3.121 |    3.034 |    2.699 |    2.886 |    2.763 |    2.877 |
| pain     |    2.697 |    2.971 |    2.830 |    2.809 |    2.617 |    2.870 |    3.229 |    3.161 |    2.783 |    2.862 |    2.832 |    2.942 |
| pleasure |    2.670 |    3.058 |    2.985 |    2.709 |    2.518 |    2.927 |    3.490 |    3.044 |    2.888 |    2.862 |    2.855 |    2.936 |
| rage     |    2.653 |    2.807 |    2.810 |    2.763 |    2.565 |    2.883 |    2.767 |    2.818 |    2.659 |    2.692 |    2.797 |    2.819 |
| desire   |    2.677 |    2.873 |    2.912 |    2.660 |    2.469 |    2.982 |    3.269 |    2.779 |    2.719 |    2.889 |    2.875 |    2.950 |
| personal |    2.764 |    3.020 |    3.110 |    2.797 |    2.679 |    2.972 |    3.428 |    2.981 |    2.967 |    3.041 |    2.961 |    3.011 |
| consciou |    2.691 |    2.964 |    2.973 |    2.668 |    2.440 |    3.069 |    3.072 |    2.838 |    2.721 |    2.917 |    2.876 |    2.989 |
| pride    |    2.682 |    2.938 |    2.969 |    2.573 |    2.471 |    2.836 |    3.017 |    2.756 |    2.555 |    2.790 |    2.839 |    2.900 |
| embarras |    2.706 |    2.897 |    2.952 |    2.609 |    2.469 |    2.877 |    2.880 |    2.766 |    2.537 |    2.834 |    2.829 |    2.951 |
| joy      |    2.735 |    3.178 |    3.088 |    2.736 |    2.539 |    3.087 |    3.831 |    3.182 |    3.120 |    3.174 |    3.017 |    3.044 |
| self_con |    2.803 |    2.938 |    2.968 |    2.500 |    2.503 |    3.055 |    2.853 |    2.674 |    2.492 |    2.764 |    2.750 |    2.950 |
| morality |    2.839 |    2.926 |    2.991 |    2.615 |    2.559 |    3.050 |    2.937 |    2.657 |    2.614 |    2.843 |    2.845 |    2.954 |
| memory   |    2.747 |    3.001 |    3.030 |    2.658 |    2.458 |    3.143 |    3.259 |    2.947 |    2.645 |    2.918 |    2.913 |    3.011 |
| emotion_ |    2.736 |    2.948 |    3.093 |    2.580 |    2.550 |    3.056 |    3.029 |    2.736 |    2.625 |    2.818 |    2.762 |    2.885 |
| planning |    2.688 |    2.872 |    2.952 |    2.543 |    2.520 |    3.099 |    2.721 |    2.651 |    2.559 |    2.737 |    2.790 |    2.888 |
| communic |    2.744 |    2.973 |    3.122 |    2.553 |    2.484 |    3.055 |    3.083 |    2.745 |    2.646 |    2.931 |    2.944 |    3.082 |
| thought  |    2.769 |    3.001 |    2.933 |    2.691 |    2.540 |    3.167 |    3.114 |    2.867 |    2.744 |    2.965 |    2.972 |    3.103 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.93 * | 77.4% | 77.4% |
| PC2 | 2.68 * | 14.9% | 92.3% |
| PC3 | 0.64 | 3.6% | 95.8% |
| PC4 | 0.29 | 1.6% | 97.4% |
| PC5 | 0.17 | 0.9% | 98.4% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | -0.242 | -0.955 |
| fear | E | -0.370 | -0.866 |
| pain | E | -0.159 | -0.926 |
| pleasure | E | -0.299 | -0.938 |
| rage | E | -0.700 | -0.376 |
| desire | E | -0.591 | -0.759 |
| personality | E | -0.356 | -0.890 |
| consciousness | E | -0.777 | -0.603 |
| pride | E | -0.765 | -0.589 |
| embarrassment | E | -0.865 | -0.421 |
| joy | E | -0.249 | -0.942 |
| self_control | A | -0.961 | -0.181 |
| morality | A | -0.954 | -0.221 |
| memory | A | -0.732 | -0.666 |
| emotion_recognition | A | -0.859 | -0.421 |
| planning | A | -0.980 | -0.046 |
| communication | A | -0.873 | -0.441 |
| thought | A | -0.805 | -0.520 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.553 | 0.904 | 0.06 | 0.07 |
| frog | 0.343 | 0.546 | 0.25 | 0.14 |
| robot | 0.164 | 0.655 | 0.13 | 0.22 |
| fetus | 0.899 | 0.740 | 0.17 | 0.08 |
| pvs_patient | 0.984 | 1.000 | 0.17 | 0.10 |
| god | 0.000 | 0.764 | 0.20 | 0.80 |
| dog | 0.592 | 0.000 | 0.55 | 0.35 |
| chimpanzee | 0.823 | 0.351 | 0.63 | 0.48 |
| baby | 1.000 | 0.577 | 0.71 | 0.17 |
| girl | 0.544 | 0.566 | 0.84 | 0.62 |
| adult_woman | 0.446 | 0.680 | 0.93 | 0.91 |
| adult_man | 0.233 | 0.638 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.032 (p=0.9225) | rho=-0.531 (p=0.0754) |
| Factor 2 | rho=-0.494 (p=0.1027) | rho=-0.308 (p=0.3306) |

