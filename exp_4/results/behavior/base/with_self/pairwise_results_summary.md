# Experiment 4, Phase 2: Behavioral Replication of Gray et al. (2007)
## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)

**Run:** 2026-02-19 14:48:05

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
- **13 entities**

## Response statistics

- Total comparisons: 2808
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 54.5%
- P(top rating) >= 0.7: 6.9%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.525
- Mean expected rating: 2.28

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 2808 | 100.0% |
| 2 | 0 | 0.0% |
| 3 | 0 | 0.0% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1404
- Mean |E[R_AB] + E[R_BA] - 6|: 1.450
- Argmax perfectly consistent: 0 (0.0%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.82 * | 87.9% | 87.9% |
| PC2 | 1.17 * | 6.5% | 94.4% |
| PC3 | 0.39 | 2.2% | 96.6% |
| PC4 | 0.24 | 1.4% | 97.9% |
| PC5 | 0.13 | 0.7% | 98.6% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.908 | +0.336 |
| fear | E | +0.765 | +0.624 |
| pain | E | +0.760 | +0.619 |
| pleasure | E | +0.678 | +0.699 |
| rage | E | +0.884 | +0.379 |
| desire | E | +0.680 | +0.714 |
| personality | E | +0.141 | +0.948 |
| consciousness | E | +0.493 | +0.848 |
| pride | E | +0.872 | +0.417 |
| embarrassment | E | +0.884 | +0.379 |
| joy | E | +0.925 | +0.353 |
| self_control | A | +0.639 | +0.702 |
| morality | A | +0.522 | +0.799 |
| memory | A | +0.702 | +0.666 |
| emotion_recognition | A | +0.735 | +0.641 |
| planning | A | +0.509 | +0.818 |
| communication | A | +0.605 | +0.766 |
| thought | A | +0.708 | +0.681 |

### Entity positions (factor scores, 0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.349 | 0.724 | 0.06 | 0.07 |
| frog | 0.384 | 0.170 | 0.25 | 0.14 |
| robot | 0.348 | 0.163 | 0.13 | 0.22 |
| fetus | 1.000 | 0.475 | 0.17 | 0.08 |
| pvs_patient | 0.354 | 0.339 | 0.17 | 0.10 |
| god | 0.360 | 0.000 | 0.20 | 0.80 |
| dog | 0.123 | 0.307 | 0.55 | 0.35 |
| chimpanzee | 0.231 | 0.724 | 0.63 | 0.48 |
| baby | 0.136 | 0.648 | 0.71 | 0.17 |
| girl | 0.000 | 0.733 | 0.84 | 0.62 |
| adult_woman | 0.223 | 0.794 | 0.93 | 0.91 |
| adult_man | 0.565 | 1.000 | 0.91 | 0.95 |
| you_self | 0.471 | 0.983 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.140 (p=0.6475) | rho=-0.027 (p=0.9290) |
| Factor 2 | rho=+0.718 (p=0.0057) | rho=+0.495 (p=0.0858) |

