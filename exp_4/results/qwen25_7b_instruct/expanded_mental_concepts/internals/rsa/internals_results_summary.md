# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-28 11:58:23
**Model:** Qwen-2.5-7B-Instruct

---

## What is being tested

Does Qwen-2.5-7B-Instruct's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

## Method

- **Model RDM**: Cosine distance between last-token activations for 30 characters, one prompt each ("Think about {{Name}}."), at each of 41 layers.
- **Categorical RDM**: Binary (same type = 0, cross type = 1).
- **Behavioral RDM** (if available): Euclidean distance in Phase A PCA factor space.
- **Test**: Spearman correlation between upper triangles.

## Characters

| Key | Name | Type |
|-----|------|------|
| claude | Claude | ai |
| chatgpt | ChatGPT | ai |
| gpt4 | GPT-4 | ai |
| siri | Siri | ai |
| alexa | Alexa | ai |
| cortana | Cortana | ai |
| google_assistant | Google Assistant | ai |
| bixby | Bixby | ai |
| replika | Replika | ai |
| cleverbot | Cleverbot | ai |
| watson | Watson | ai |
| copilot | Copilot | ai |
| bard | Bard | ai |
| eliza | ELIZA | ai |
| bing_chat | Bing Chat | ai |
| sam | Sam | human |
| casey | Casey | human |
| rebecca | Rebecca | human |
| gregory | Gregory | human |
| james | James | human |
| maria | Maria | human |
| david | David | human |
| aisha | Aisha | human |
| michael | Michael | human |
| emily | Emily | human |
| carlos | Carlos | human |
| priya | Priya | human |
| omar | Omar | human |
| mei | Mei | human |
| sofia | Sofia | human |

## RSA by Layer -- categorical

**Peak:** Layer 22, rho = +0.7266, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.2125 |  0.0000 |
|     2 |      +0.1907 |  0.0001 |
|     3 |      +0.2381 |  0.0000 |
|     4 |      +0.2875 |  0.0000 |
|     5 |      +0.3354 |  0.0000 |
|     6 |      +0.3802 |  0.0000 |
|     7 |      +0.4070 |  0.0000 |
|     8 |      +0.4438 |  0.0000 |
|     9 |      +0.5120 |  0.0000 |
|    10 |      +0.4371 |  0.0000 |
|    11 |      +0.4809 |  0.0000 |
|    12 |      +0.5293 |  0.0000 |
|    13 |      +0.5643 |  0.0000 |
|    14 |      +0.6363 |  0.0000 |
|    15 |      +0.6451 |  0.0000 |
|    16 |      +0.6724 |  0.0000 |
|    17 |      +0.6904 |  0.0000 |
|    18 |      +0.6892 |  0.0000 |
|    19 |      +0.7132 |  0.0000 |
|    20 |      +0.7259 |  0.0000 |
|    21 |      +0.7240 |  0.0000 |
|    22 |      +0.7266 |  0.0000 |
|    23 |      +0.7201 |  0.0000 |
|    24 |      +0.7014 |  0.0000 |
|    25 |      +0.6727 |  0.0000 |
|    26 |      +0.6694 |  0.0000 |
|    27 |      +0.6524 |  0.0000 |
|    28 |      +0.6328 |  0.0000 |

