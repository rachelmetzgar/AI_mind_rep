# Experiment 4, Phase 2: Behavioral Replication of Gray et al. (2007)
## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)

**Run:** 2026-02-19 14:45:33

---

## What is being tested

Does LLaMA-2-13B's (base, pretrained) **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained LLaMA-2-13B (no chat/RLHF)
- **Logit-based ratings**: probability distribution over tokens "1"-"5" instead of generated text
- **Expected rating**: continuous E[R] = sum(p_i * i) instead of discrete response
- **No photos**: text descriptions only
- **Completion format**: prompt ends with "Rating:" for natural completion
- **Position-bias control**: each pair in both orders
- **12 entities**

## Response statistics

- Total comparisons: 2376
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 58.6%
- P(top rating) >= 0.7: 8.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.537
- Mean expected rating: 2.25

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 2376 | 100.0% |
| 2 | 0 | 0.0% |
| 3 | 0 | 0.0% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1188
- Mean |E[R_AB] + E[R_BA] - 6|: 1.509
- Argmax perfectly consistent: 0 (0.0%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.56 * | 86.5% | 86.5% |
| PC2 | 1.46 * | 8.1% | 94.6% |
| PC3 | 0.39 | 2.2% | 96.7% |
| PC4 | 0.20 | 1.1% | 97.8% |
| PC5 | 0.15 | 0.8% | 98.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.910 | +0.334 |
| fear | E | +0.758 | +0.617 |
| pain | E | +0.762 | +0.616 |
| pleasure | E | +0.672 | +0.697 |
| rage | E | +0.930 | +0.293 |
| desire | E | +0.670 | +0.715 |
| personality | E | +0.060 | +0.964 |
| consciousness | E | +0.464 | +0.857 |
| pride | E | +0.880 | +0.389 |
| embarrassment | E | +0.891 | +0.368 |
| joy | E | +0.936 | +0.312 |
| self_control | A | +0.672 | +0.696 |
| morality | A | +0.455 | +0.826 |
| memory | A | +0.681 | +0.721 |
| emotion_recognition | A | +0.713 | +0.673 |
| planning | A | +0.485 | +0.834 |
| communication | A | +0.588 | +0.774 |
| thought | A | +0.674 | +0.711 |

### Entity positions (factor scores, 0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.314 | 0.639 | 0.06 | 0.07 |
| frog | 0.397 | 0.137 | 0.25 | 0.14 |
| robot | 0.358 | 0.166 | 0.13 | 0.22 |
| fetus | 1.000 | 0.480 | 0.17 | 0.08 |
| pvs_patient | 0.402 | 0.353 | 0.17 | 0.10 |
| god | 0.386 | 0.000 | 0.20 | 0.80 |
| dog | 0.167 | 0.267 | 0.55 | 0.35 |
| chimpanzee | 0.250 | 0.691 | 0.63 | 0.48 |
| baby | 0.150 | 0.620 | 0.71 | 0.17 |
| girl | 0.000 | 0.663 | 0.84 | 0.62 |
| adult_woman | 0.266 | 0.735 | 0.93 | 0.91 |
| adult_man | 0.626 | 1.000 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.378 (p=0.2253) | rho=-0.210 (p=0.5128) |
| Factor 2 | rho=+0.613 (p=0.0341) | rho=+0.371 (p=0.2356) |

