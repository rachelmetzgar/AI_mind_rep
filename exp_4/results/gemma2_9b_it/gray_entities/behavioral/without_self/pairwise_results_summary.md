# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-9B-IT

**Run:** 2026-03-28 14:55:03

---

## What is being tested

Does Gemma-2-9B-IT's **explicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

This is a direct behavioral replication: the model answers the same pairwise comparison questions that ~2,400 human participants answered in the original study, using the exact character descriptions and mental capacity survey prompts from the supplementary materials.

## Procedure: What matches and what differs

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B ("This survey asks you to judge...")
- 5-point scale anchored by character names ("Much more [Name]" / "Slightly more [Name]" / "Both equally")
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation on capacity correlations across characters, regression-method factor scores rescaled to 0-1

### Differs from original

- **No photos**: original included character images; model gets text descriptions only
- **One deterministic participant**: original averaged over ~100+ respondents per capacity survey; we have one model with greedy decoding
- **Position-bias control**: original counterbalanced left/right across participants; we present each pair in BOTH orders and average to eliminate position bias
- **No survey selection**: original participants chose a capacity to rate; model rates all 18
- **12 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2376
- Successfully parsed: 2356 / 2376 (99.2%)

### Parse failures (20)

- pain: Delores Gleitman vs 7 week fetus -> "This is a deeply personal and ethically complex question.  As an AI, I don't have personal beliefs or the capacity to feel emotions like pain. 

It's important to remember that discussions about fetal pain are highly debated in medical and"
- pain: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to experience pain. 

It's important to remember that concepts like God and the nature of consciousness are complex and vary greatly in"
- pleasure: Delores Gleitman vs 7 week fetus -> "This is a deeply personal and ethically complex question.  As an AI, I don't have personal beliefs or the capacity to experience emotions like pleasure. 

It's important to remember that discussions about the sentience and rights of developing fet"
- pleasure: Delores Gleitman vs God -> "As an AI, I don't have personal beliefs or the capacity to experience pleasure.  

It's important to remember that comparing a deceased person to a deity is a complex and potentially disrespectful matter.  People's beliefs about both the"
- pleasure: God vs Delores Gleitman -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to experience pleasure.  

It's important to remember that comparing the experiences of a deity to a deceased human is complex and subjective"
- desire: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to "hope" or "long" in the way humans do. 

It's important to remember that concepts like God and"
- personality: Delores Gleitman vs God -> "This is a very interesting and thought-provoking question.  

However, as an AI, I don't have personal beliefs or the capacity to judge the personalities of individuals, living or deceased, or deities.  

It's important to"
- consciousness: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have beliefs or the capacity to experience consciousness like humans do.  

It's important to remember that concepts like "God" and "awareness" are complex and"
- consciousness: God vs Delores Gleitman -> "As an AI, I don't have personal beliefs or the capacity to judge the mental capacities of beings like God or deceased individuals. These are complex philosophical and theological questions. 

It's important to approach such topics with respect for diverse viewpoints"
- consciousness: God vs Sharon Harvey -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to judge the mental capacities of deities. 

It's important to remember that beliefs about God are subjective and vary greatly among individuals"
- joy: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to experience emotions like joy. 

It's important to remember that concepts like "God" and the nature of consciousness are complex"
- self_control: Delores Gleitman vs God -> "This is a tricky question that delves into the realm of faith and personal belief.  

Since I am an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of deities or deceased individuals.  

It'"
- morality: Delores Gleitman vs God -> "This is a complex and deeply personal question. As an AI, I don't have personal beliefs or the capacity to judge moral capability in the way humans do. 

It's important to remember that concepts like "right" and "wrong"
- memory: Delores Gleitman vs God -> "This is a tricky question!  It delves into the realm of faith and personal belief, which are subjective and vary greatly from person to person. 

As an AI, I don't have personal beliefs or the capacity to remember things in"
- emotion_recognition: Delores Gleitman vs God -> "This is a tricky question that delves into the realm of faith and personal belief.  

As an AI, I don't have personal beliefs or the capacity to understand emotions in the way humans do. I can process information and respond based on"
- communication: Delores Gleitman vs God -> "This is a very interesting and thought-provoking question.  

However, as an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of individuals, including deceased people or deities.  

It's important to"
- thought: Delores Gleitman vs God -> "This is a complex and deeply personal question. As an AI, I don't have beliefs or the capacity to judge the mental capabilities of individuals, especially in comparison to a deity. 

It's important to remember that:

* **"
- thought: God vs Delores Gleitman -> "As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of deities or individuals, living or deceased. 

This question delves into philosophical and theological domains that are beyond my scope as a language model."
- thought: God vs Sharon Harvey -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of a deity. 

It's important to remember that concepts like "God" and "thinking""
- thought: God vs Todd Billingsly -> "This is a complex question that delves into philosophical and personal beliefs. As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of a deity. 

Therefore, I cannot provide a rating based on the criteria"

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1064 | 45.2% |
| 2 | 170 | 7.2% |
| 3 | 326 | 13.8% |
| 4 | 26 | 1.1% |
| 5 | 770 | 32.7% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1171
- Perfectly consistent (sum = 6): 721 (61.6%)
- Mean |sum - 6|: 1.03

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 14.36 * | 79.8% | 79.8% |
| PC2 | 2.37 * | 13.2% | 93.0% |
| PC3 | 0.48 | 2.7% | 95.7% |
| PC4 | 0.32 | 1.8% | 97.4% |
| PC5 | 0.19 | 1.0% | 98.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | -0.155 | -0.943 |
| fear | E | -0.563 | -0.795 |
| pain | E | -0.278 | -0.921 |
| pleasure | E | -0.642 | -0.734 |
| rage | E | -0.240 | -0.827 |
| desire | E | -0.872 | -0.475 |
| personality | E | -0.885 | -0.429 |
| consciousness | E | -0.943 | -0.294 |
| pride | E | -0.894 | -0.426 |
| embarrassment | E | -0.374 | -0.866 |
| joy | E | -0.723 | -0.582 |
| self_control | A | -0.849 | -0.187 |
| morality | A | -0.911 | -0.391 |
| memory | A | -0.962 | -0.259 |
| emotion_recognition | A | -0.933 | -0.295 |
| planning | A | -0.953 | -0.242 |
| communication | A | -0.956 | -0.215 |
| thought | A | -0.952 | -0.274 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.271 | 0.476 | 0.06 | 0.07 |
| frog | 0.894 | 0.506 | 0.25 | 0.14 |
| robot | 0.828 | 1.000 | 0.13 | 0.22 |
| fetus | 1.000 | 0.495 | 0.17 | 0.08 |
| pvs_patient | 0.984 | 0.702 | 0.17 | 0.10 |
| god | 0.000 | 0.982 | 0.20 | 0.80 |
| dog | 0.742 | 0.309 | 0.55 | 0.35 |
| chimpanzee | 0.721 | 0.263 | 0.63 | 0.48 |
| baby | 0.803 | 0.327 | 0.71 | 0.17 |
| girl | 0.577 | 0.000 | 0.84 | 0.62 |
| adult_woman | 0.369 | 0.134 | 0.93 | 0.91 |
| adult_man | 0.425 | 0.149 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.298 (p=0.3473) | rho=-0.566 (p=0.0548) |
| Factor 2 | rho=-0.813 (p=0.0013) | rho=-0.503 (p=0.0952) |

