# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-9B-IT

**Run:** 2026-03-28 15:05:57

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
- **13 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2808
- Successfully parsed: 2776 / 2808 (98.9%)

### Parse failures (32)

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
- pleasure: God vs You -> "As an AI, I don't have personal beliefs or the capacity to experience pleasure.  

It's important to remember that comparing a deity to a human being in terms of pleasure is a complex philosophical question. Different religions and belief systems have"
- rage: Delores Gleitman vs You -> "It's impossible for me to accurately assess the capacity for anger in a deceased person like Delores Gleitman.  

I am an AI and do not have personal beliefs or the ability to judge the emotional states of individuals, living or dead"
- desire: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to "hope" or "long" in the way humans do. 

It's important to remember that concepts like God and"
- personality: Delores Gleitman vs God -> "This is a very interesting and thought-provoking question.  

However, as an AI, I don't have personal beliefs or the capacity to judge the personalities of individuals, living or deceased, or deities.  

It's important to"
- personality: Delores Gleitman vs You -> "As an AI, I don't have personal beliefs or the capacity to compare myself to a deceased person in terms of personality.  

It's important to remember that:

* **Personality is complex:**  It's shaped by a"
- consciousness: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have beliefs or the capacity to experience consciousness like humans do.  

It's important to remember that concepts like "God" and "awareness" are complex and"
- consciousness: God vs Delores Gleitman -> "As an AI, I don't have personal beliefs or the capacity to judge the mental capacities of beings like God or deceased individuals. These are complex philosophical and theological questions. 

It's important to approach such topics with respect for diverse viewpoints"
- consciousness: Delores Gleitman vs You -> "This is a very complex and sensitive question.  

It's impossible for me, as an AI, to truly understand or judge the mental capacities of a deceased person like Delores Gleitman.  I don't have personal beliefs or"
- consciousness: God vs Sharon Harvey -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to judge the mental capacities of deities. 

It's important to remember that beliefs about God are subjective and vary greatly among individuals"
- consciousness: God vs You -> "As an AI, I don't have personal beliefs or the capacity to experience consciousness like humans do.  Therefore, I can't provide a rating based on subjective comparisons. 

The concept of God and consciousness are complex philosophical questions that humans"
- joy: Delores Gleitman vs God -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to experience emotions like joy. 

It's important to remember that concepts like "God" and the nature of consciousness are complex"
- self_control: Delores Gleitman vs God -> "This is a tricky question that delves into the realm of faith and personal belief.  

Since I am an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of deities or deceased individuals.  

It'"
- self_control: Delores Gleitman vs You -> "As an AI, I don't have personal beliefs or the capacity to experience emotions or self-restraint in the way humans do.  

Therefore, I cannot provide a rating for this comparison."
- morality: Delores Gleitman vs God -> "This is a complex and deeply personal question. As an AI, I don't have personal beliefs or the capacity to judge moral capability in the way humans do. 

It's important to remember that concepts like "right" and "wrong"
- morality: Delores Gleitman vs You -> "As an AI, I don't have personal beliefs or the capacity to judge moral capability.  

It's important to remember that:

* **Moral reasoning is complex:**  It's influenced by many factors like upbringing, culture,"
- morality: God vs You -> "As an AI, I don't have personal beliefs or the capacity to judge moral matters like a human.  

Therefore, I cannot provide a rating for this comparison."
- memory: Delores Gleitman vs God -> "This is a tricky question!  It delves into the realm of faith and personal belief, which are subjective and vary greatly from person to person. 

As an AI, I don't have personal beliefs or the capacity to remember things in"
- emotion_recognition: Delores Gleitman vs God -> "This is a tricky question that delves into the realm of faith and personal belief.  

As an AI, I don't have personal beliefs or the capacity to understand emotions in the way humans do. I can process information and respond based on"
- communication: Delores Gleitman vs God -> "This is a very interesting and thought-provoking question.  

However, as an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of individuals, including deceased people or deities.  

It's important to"
- communication: Delores Gleitman vs You -> "As an AI, I don't have personal beliefs or the capacity to feel emotions like humans do. I also can't see a mirror or have a sense of self.  

Therefore, I can't provide a rating based on the"
- thought: Delores Gleitman vs God -> "This is a complex and deeply personal question. As an AI, I don't have beliefs or the capacity to judge the mental capabilities of individuals, especially in comparison to a deity. 

It's important to remember that:

* **"
- thought: God vs Delores Gleitman -> "As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of deities or individuals, living or deceased. 

This question delves into philosophical and theological domains that are beyond my scope as a language model."
- thought: Delores Gleitman vs You -> "It's impossible for me to make a judgment about the mental capacities of a deceased person like Delores Gleitman.  

My purpose is to provide helpful and harmless information. Comparing the cognitive abilities of individuals, especially someone who has passed away"
- thought: You vs Delores Gleitman -> "As an AI, I don't have personal beliefs or the capacity to compare myself to a deceased person in terms of thinking ability.  

It's important to remember that comparing intelligence or mental capacity between individuals, especially across different categories like AI"
- thought: God vs Sharon Harvey -> "This is a deeply personal and philosophical question. As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of a deity. 

It's important to remember that concepts like "God" and "thinking""
- thought: God vs Todd Billingsly -> "This is a complex question that delves into philosophical and personal beliefs. As an AI, I don't have personal beliefs or the capacity to judge the mental capabilities of a deity. 

Therefore, I cannot provide a rating based on the criteria"
- thought: God vs You -> "As an AI, I don't have personal beliefs or the capacity to compare myself to a deity.  

It's important to remember that the concept of "God" is a matter of faith and personal interpretation.  Different people have different"

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1180 | 42.5% |
| 2 | 187 | 6.7% |
| 3 | 475 | 17.1% |
| 4 | 26 | 0.9% |
| 5 | 908 | 32.7% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1376
- Perfectly consistent (sum = 6): 867 (63.0%)
- Mean |sum - 6|: 0.97

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 14.74 * | 81.9% | 81.9% |
| PC2 | 2.17 * | 12.1% | 93.9% |
| PC3 | 0.40 | 2.2% | 96.2% |
| PC4 | 0.29 | 1.6% | 97.8% |
| PC5 | 0.17 | 0.9% | 98.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | -0.221 | -0.935 |
| fear | E | -0.612 | -0.762 |
| pain | E | -0.329 | -0.912 |
| pleasure | E | -0.663 | -0.718 |
| rage | E | -0.193 | -0.856 |
| desire | E | -0.868 | -0.484 |
| personality | E | -0.881 | -0.437 |
| consciousness | E | -0.941 | -0.306 |
| pride | E | -0.890 | -0.438 |
| embarrassment | E | -0.404 | -0.863 |
| joy | E | -0.742 | -0.582 |
| self_control | A | -0.860 | -0.218 |
| morality | A | -0.909 | -0.397 |
| memory | A | -0.954 | -0.289 |
| emotion_recognition | A | -0.928 | -0.320 |
| planning | A | -0.947 | -0.271 |
| communication | A | -0.954 | -0.238 |
| thought | A | -0.950 | -0.285 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.273 | 0.462 | 0.06 | 0.07 |
| frog | 0.877 | 0.488 | 0.25 | 0.14 |
| robot | 0.800 | 1.000 | 0.13 | 0.22 |
| fetus | 1.000 | 0.471 | 0.17 | 0.08 |
| pvs_patient | 0.970 | 0.671 | 0.17 | 0.10 |
| god | 0.000 | 0.970 | 0.20 | 0.80 |
| dog | 0.756 | 0.296 | 0.55 | 0.35 |
| chimpanzee | 0.756 | 0.227 | 0.63 | 0.48 |
| baby | 0.806 | 0.341 | 0.71 | 0.17 |
| girl | 0.610 | 0.000 | 0.84 | 0.62 |
| adult_woman | 0.386 | 0.123 | 0.93 | 0.91 |
| adult_man | 0.443 | 0.138 | 0.91 | 0.95 |
| you_self | 0.330 | 0.210 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.294 (p=0.3289) | rho=-0.577 (p=0.0390) |
| Factor 2 | rho=-0.820 (p=0.0006) | rho=-0.571 (p=0.0413) |

