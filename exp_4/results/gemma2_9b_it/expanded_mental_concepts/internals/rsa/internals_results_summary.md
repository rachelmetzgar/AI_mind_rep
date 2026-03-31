# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-28 12:56:31
**Model:** Gemma-2-9B-IT

---

## What is being tested

Does Gemma-2-9B-IT's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 25, rho = +0.8495, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.2778 |  0.0000 |
|     2 |      +0.2068 |  0.0000 |
|     3 |      +0.2354 |  0.0000 |
|     4 |      +0.2312 |  0.0000 |
|     5 |      +0.4343 |  0.0000 |
|     6 |      +0.4670 |  0.0000 |
|     7 |      +0.4285 |  0.0000 |
|     8 |      +0.4703 |  0.0000 |
|     9 |      +0.7137 |  0.0000 |
|    10 |      +0.7310 |  0.0000 |
|    11 |      +0.7754 |  0.0000 |
|    12 |      +0.7796 |  0.0000 |
|    13 |      +0.7963 |  0.0000 |
|    14 |      +0.8001 |  0.0000 |
|    15 |      +0.8053 |  0.0000 |
|    16 |      +0.8009 |  0.0000 |
|    17 |      +0.8089 |  0.0000 |
|    18 |      +0.8146 |  0.0000 |
|    19 |      +0.8210 |  0.0000 |
|    20 |      +0.8293 |  0.0000 |
|    21 |      +0.8294 |  0.0000 |
|    22 |      +0.8352 |  0.0000 |
|    23 |      +0.8380 |  0.0000 |
|    24 |      +0.8481 |  0.0000 |
|    25 |      +0.8495 |  0.0000 |
|    26 |      +0.8495 |  0.0000 |
|    27 |      +0.8484 |  0.0000 |
|    28 |      +0.8427 |  0.0000 |
|    29 |      +0.8339 |  0.0000 |
|    30 |      +0.8285 |  0.0000 |
|    31 |      +0.8254 |  0.0000 |
|    32 |      +0.8244 |  0.0000 |
|    33 |      +0.8221 |  0.0000 |
|    34 |      +0.8117 |  0.0000 |
|    35 |      +0.8078 |  0.0000 |
|    36 |      +0.8056 |  0.0000 |
|    37 |      +0.8023 |  0.0000 |
|    38 |      +0.8006 |  0.0000 |
|    39 |      +0.8000 |  0.0000 |
|    40 |      +0.7723 |  0.0000 |
|    41 |      +0.7544 |  0.0000 |
|    42 |      +0.7101 |  0.0000 |

