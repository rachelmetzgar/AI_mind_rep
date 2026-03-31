# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-29 10:57:53
**Model:** Gemma-2-2B (Base)

---

## What is being tested

Does Gemma-2-2B (Base)'s internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 11, rho = +0.6593, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.3345 |  0.0000 |
|     2 |      +0.3015 |  0.0000 |
|     3 |      +0.3514 |  0.0000 |
|     4 |      +0.4925 |  0.0000 |
|     5 |      +0.5242 |  0.0000 |
|     6 |      +0.6043 |  0.0000 |
|     7 |      +0.6351 |  0.0000 |
|     8 |      +0.6561 |  0.0000 |
|     9 |      +0.6433 |  0.0000 |
|    10 |      +0.6453 |  0.0000 |
|    11 |      +0.6593 |  0.0000 |
|    12 |      +0.6560 |  0.0000 |
|    13 |      +0.6534 |  0.0000 |
|    14 |      +0.6334 |  0.0000 |
|    15 |      +0.6393 |  0.0000 |
|    16 |      +0.6484 |  0.0000 |
|    17 |      +0.6197 |  0.0000 |
|    18 |      +0.5973 |  0.0000 |
|    19 |      +0.6001 |  0.0000 |
|    20 |      +0.5973 |  0.0000 |
|    21 |      +0.5888 |  0.0000 |
|    22 |      +0.5842 |  0.0000 |
|    23 |      +0.5904 |  0.0000 |
|    24 |      +0.5800 |  0.0000 |
|    25 |      +0.5504 |  0.0000 |
|    26 |      +0.4270 |  0.0000 |

