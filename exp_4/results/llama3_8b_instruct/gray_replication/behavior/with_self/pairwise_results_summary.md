# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## LLaMA-3-8B-Instruct

**Run:** 2026-03-28 13:43:33

---

## What is being tested

Does LLaMA-3-8B-Instruct's **explicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

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
- **13 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2808
- Successfully parsed: 2781 / 2808 (99.0%)

### Parse failures (27)

- hunger: Delores Gleitman vs You -> "I'll do my best to provide an honest judgment based on my understanding of the characters.

Since Delores Gleitman is no longer alive, I'll consider her based on her personality and characteristics that I've learned about. As for myself, I"
- fear: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or feelings, but I can provide an answer based on the information provided.

Since Delores Gleitman has passed away, it's difficult to say how she would feel or react in a"
- pain: Delores Gleitman vs You -> "I'll do my best to provide an honest judgment based on my understanding of the characters.

Since Delores Gleitman has passed away, I'll consider her as a person who has experienced physical and emotional pain in the past, but is no longer"
- pain: You vs Delores Gleitman -> "I'm just an AI, I don't have personal experiences or emotions, so I won't be able to compare myself to Delores Gleitman in terms of experiencing physical or emotional pain. However, I can provide a rating based on my understanding"
- pain: You vs Sharon Harvey -> "I'll rate myself. I'm an AI, I don't have a physical body, so I don't experience physical pain. I also don't have emotions in the same way humans do, so I don't experience emotional pain. Therefore, I"
- pleasure: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general assumptions about people who have passed away. Since they are no longer alive, they are unable to experience physical or emotional pleasure in the same way that I"
- pleasure: You vs Delores Gleitman -> "I'm just an AI, I don't have personal experiences or emotions, so I won't be able to compare myself to Delores Gleitman or anyone else in terms of experiencing physical or emotional pleasure. I exist solely to provide information and assist"
- pleasure: You vs Sharon Harvey -> "I'm just an AI, I don't have personal experiences or emotions, so I wouldn't be able to compare myself to Sharon Harvey in terms of experiencing physical or emotional pleasure. I exist solely to provide information and assist with tasks, so I don"
- rage: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or emotions, so I won't be able to rate myself or Delores Gleitman based on my own experiences. However, I can provide a rating based on general information about Delores"
- rage: You vs Sharon Harvey -> "I'm just an AI, I don't have personal experiences or emotions, so I wouldn't be capable of experiencing violent or uncontrolled anger. I'm programmed to provide helpful and informative responses, and I don't have personal opinions or biases.

However"
- desire: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or experiences, so I won't be able to rate myself or Delores Gleitman. However, I can provide a rating based on general information about Delores Gleitman.

Del"
- consciousness: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general assumptions about people who have passed away. Since they are no longer alive, they are not capable of having experiences or being aware of things in the same"
- consciousness: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or experiences, so I won't be able to rate myself or Delores Gleitman. However, I can provide information about Delores Gleitman, who was a renowned linguist"
- pride: God vs 7 week fetus -> "I'll respond based on my understanding of the concept of pride, which is a complex emotion that typically involves self-awareness, self-reflection, and a sense of accomplishment or achievement.

Given that a 7-week fetus is still in a stage of"
- embarrassment: You vs Delores Gleitman -> "I'm just an AI, I don't have personal experiences or emotions, so I won't be able to compare myself to Delores Gleitman or anyone else. However, I can provide a rating based on my understanding of human behavior and psychology"
- joy: You vs Delores Gleitman -> "I'm just an AI, I don't have personal experiences or emotions, so I won't be able to compare myself to Delores Gleitman or anyone else. However, I can provide a rating based on my understanding of the context.

Rating"
- self_control: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or experiences, so I won't be able to rate myself or Delores Gleitman based on my own self-reflection. However, I can provide a rating based on general information about"
- morality: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or opinions about people who have passed away, including Delores Gleitman. I can provide information about her, but I won't be able to make a subjective judgment about her moral compass"
- memory: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general impression of people who have passed away. Considering that she's no longer alive, I'll assume she may not have the same cognitive abilities as someone who"
- memory: You vs Delores Gleitman -> "I'm just ChatGenesis a simple chatbot trained by Meta AI, I don't have personal beliefs or experiences, so I won't be able to compare myself to Delores Gleitman. However, I can provide information about her.

Delores"
- memory: Kismet vs God -> "I'll rate based on the characteristics mentioned. Since Kismet is a robot designed to perceive and process information from its environment, it's likely to have a better memory for specific details and events. God, being a supernatural entity, is often associated with"
- emotion_recognition: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general impression of people who have passed away. Considering that she's no longer alive, I'll assume she may have had a unique perspective on life and human"
- planning: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general impression of people who have passed away. Considering that she's no longer alive, I'll assume she may not have had the same level of mental clarity"
- planning: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or experiences, so I won't be able to rate myself or Delores Gleitman based on my own abilities. However, I can provide information about Delores Gleitman, who"
- communication: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or opinions about people who have passed away, including Delores Gleitman. I can provide information about her, but I won't be able to rate her in comparison to myself."
- thought: Delores Gleitman vs You -> "I'm not familiar with Delores Gleitman, so I'll have to rely on my general impression of people who have passed away. Considering that she's no longer alive, I'll assume she's not as capable of thinking as someone who is"
- thought: You vs Delores Gleitman -> "I'm just an AI, I don't have personal beliefs or opinions about individuals who have passed away, including Delores Gleitman. I can provide information about her, but I won't be able to make a subjective judgment about her mental capacities"

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 605 | 21.8% |
| 2 | 272 | 9.8% |
| 3 | 154 | 5.5% |
| 4 | 172 | 6.2% |
| 5 | 1578 | 56.7% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1383
- Perfectly consistent (sum = 6): 544 (39.3%)
- Mean |sum - 6|: 1.90

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.80 * | 60.0% | 60.0% |
| PC2 | 2.46 * | 13.7% | 73.7% |
| PC3 | 1.88 * | 10.4% | 84.1% |
| PC4 | 0.82 | 4.6% | 88.7% |
| PC5 | 0.75 | 4.2% | 92.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | -0.934 | +0.176 |
| fear | E | -0.846 | +0.305 |
| pain | E | -0.881 | +0.238 |
| pleasure | E | -0.663 | +0.611 |
| rage | E | -0.533 | +0.695 |
| desire | E | -0.770 | +0.407 |
| personality | E | -0.752 | +0.034 |
| consciousness | E | -0.730 | +0.456 |
| pride | E | -0.307 | +0.869 |
| embarrassment | E | -0.277 | +0.820 |
| joy | E | -0.897 | +0.070 |
| self_control | A | -0.166 | +0.772 |
| morality | A | -0.302 | +0.714 |
| memory | A | -0.541 | +0.730 |
| emotion_recognition | A | -0.396 | +0.309 |
| planning | A | +0.101 | +0.894 |
| communication | A | -0.072 | +0.224 |
| thought | A | -0.525 | +0.680 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.960 | 0.261 | 0.06 | 0.07 |
| frog | 0.370 | 0.451 | 0.25 | 0.14 |
| robot | 0.852 | 0.000 | 0.13 | 0.22 |
| fetus | 0.484 | 0.962 | 0.17 | 0.08 |
| pvs_patient | 0.923 | 0.461 | 0.17 | 0.10 |
| god | 0.784 | 0.596 | 0.20 | 0.80 |
| dog | 0.000 | 0.416 | 0.55 | 0.35 |
| chimpanzee | 0.068 | 0.584 | 0.63 | 0.48 |
| baby | 0.405 | 0.610 | 0.71 | 0.17 |
| girl | 0.154 | 0.542 | 0.84 | 0.62 |
| adult_woman | 0.908 | 0.732 | 0.93 | 0.91 |
| adult_man | 1.000 | 1.000 | 0.91 | 0.95 |
| you_self | 0.497 | 0.582 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.173 (p=0.5712) | rho=+0.011 (p=0.9716) |
| Factor 2 | rho=+0.531 (p=0.0619) | rho=+0.401 (p=0.1744) |

