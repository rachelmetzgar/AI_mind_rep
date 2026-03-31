# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-29 10:57:41
**Model:** Gemma-2-2B-IT

---

## What is being tested

Does Gemma-2-2B-IT's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 16, rho = +0.7992, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.3024 |  0.0000 |
|     2 |      +0.2301 |  0.0000 |
|     3 |      +0.2930 |  0.0000 |
|     4 |      +0.3717 |  0.0000 |
|     5 |      +0.4746 |  0.0000 |
|     6 |      +0.5780 |  0.0000 |
|     7 |      +0.6397 |  0.0000 |
|     8 |      +0.6848 |  0.0000 |
|     9 |      +0.7414 |  0.0000 |
|    10 |      +0.7441 |  0.0000 |
|    11 |      +0.7289 |  0.0000 |
|    12 |      +0.7168 |  0.0000 |
|    13 |      +0.6868 |  0.0000 |
|    14 |      +0.7837 |  0.0000 |
|    15 |      +0.7909 |  0.0000 |
|    16 |      +0.7992 |  0.0000 |
|    17 |      +0.7943 |  0.0000 |
|    18 |      +0.7776 |  0.0000 |
|    19 |      +0.7761 |  0.0000 |
|    20 |      +0.7546 |  0.0000 |
|    21 |      +0.7506 |  0.0000 |
|    22 |      +0.7416 |  0.0000 |
|    23 |      +0.7264 |  0.0000 |
|    24 |      +0.7386 |  0.0000 |
|    25 |      +0.7354 |  0.0000 |
|    26 |      +0.7618 |  0.0000 |

