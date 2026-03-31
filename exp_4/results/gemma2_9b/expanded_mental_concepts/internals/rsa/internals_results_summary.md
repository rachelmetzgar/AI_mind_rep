# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-29 10:58:21
**Model:** Gemma-2-9B (Base)

---

## What is being tested

Does Gemma-2-9B (Base)'s internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 14, rho = +0.6734, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.3549 |  0.0000 |
|     2 |      +0.3495 |  0.0000 |
|     3 |      +0.3223 |  0.0000 |
|     4 |      +0.4820 |  0.0000 |
|     5 |      +0.4343 |  0.0000 |
|     6 |      +0.4945 |  0.0000 |
|     7 |      +0.5558 |  0.0000 |
|     8 |      +0.5799 |  0.0000 |
|     9 |      +0.6342 |  0.0000 |
|    10 |      +0.6485 |  0.0000 |
|    11 |      +0.6481 |  0.0000 |
|    12 |      +0.6693 |  0.0000 |
|    13 |      +0.6687 |  0.0000 |
|    14 |      +0.6734 |  0.0000 |
|    15 |      +0.6689 |  0.0000 |
|    16 |      +0.6635 |  0.0000 |
|    17 |      +0.6590 |  0.0000 |
|    18 |      +0.6573 |  0.0000 |
|    19 |      +0.6234 |  0.0000 |
|    20 |      +0.6401 |  0.0000 |
|    21 |      +0.6250 |  0.0000 |
|    22 |      +0.6309 |  0.0000 |
|    23 |      +0.6206 |  0.0000 |
|    24 |      +0.6163 |  0.0000 |
|    25 |      +0.6155 |  0.0000 |
|    26 |      +0.6427 |  0.0000 |
|    27 |      +0.6506 |  0.0000 |
|    28 |      +0.6594 |  0.0000 |
|    29 |      +0.6680 |  0.0000 |
|    30 |      +0.6498 |  0.0000 |
|    31 |      +0.6563 |  0.0000 |
|    32 |      +0.6311 |  0.0000 |
|    33 |      +0.6066 |  0.0000 |
|    34 |      +0.5955 |  0.0000 |
|    35 |      +0.6032 |  0.0000 |
|    36 |      +0.6112 |  0.0000 |
|    37 |      +0.5911 |  0.0000 |
|    38 |      +0.6079 |  0.0000 |
|    39 |      +0.6040 |  0.0000 |
|    40 |      +0.5710 |  0.0000 |
|    41 |      +0.5774 |  0.0000 |
|    42 |      +0.5801 |  0.0000 |

