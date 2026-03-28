# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-28 09:50:59
**Model:** LLaMA-3-8B (Base)

---

## What is being tested

Does LLaMA-3-8B (Base)'s internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 22, rho = +0.6536, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1467 |  0.0022 |
|     2 |      +0.3106 |  0.0000 |
|     3 |      +0.3535 |  0.0000 |
|     4 |      +0.4589 |  0.0000 |
|     5 |      +0.4716 |  0.0000 |
|     6 |      +0.4633 |  0.0000 |
|     7 |      +0.5673 |  0.0000 |
|     8 |      +0.5722 |  0.0000 |
|     9 |      +0.5892 |  0.0000 |
|    10 |      +0.5922 |  0.0000 |
|    11 |      +0.5893 |  0.0000 |
|    12 |      +0.6045 |  0.0000 |
|    13 |      +0.6194 |  0.0000 |
|    14 |      +0.5875 |  0.0000 |
|    15 |      +0.6239 |  0.0000 |
|    16 |      +0.6273 |  0.0000 |
|    17 |      +0.6480 |  0.0000 |
|    18 |      +0.6494 |  0.0000 |
|    19 |      +0.6465 |  0.0000 |
|    20 |      +0.6445 |  0.0000 |
|    21 |      +0.6483 |  0.0000 |
|    22 |      +0.6536 |  0.0000 |
|    23 |      +0.6406 |  0.0000 |
|    24 |      +0.6321 |  0.0000 |
|    25 |      +0.6210 |  0.0000 |
|    26 |      +0.6182 |  0.0000 |
|    27 |      +0.6081 |  0.0000 |
|    28 |      +0.6184 |  0.0000 |
|    29 |      +0.6103 |  0.0000 |
|    30 |      +0.6221 |  0.0000 |
|    31 |      +0.6310 |  0.0000 |
|    32 |      +0.6342 |  0.0000 |

