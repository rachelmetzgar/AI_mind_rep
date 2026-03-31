# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-08 15:16:15
**Model:** LLaMA-2-13B-Chat

---

## What is being tested

Does LLaMA-2-13B-Chat's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 16, rho = +0.5638, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1332 |  0.0054 |
|     2 |      +0.1187 |  0.0133 |
|     3 |      +0.2079 |  0.0000 |
|     4 |      +0.2300 |  0.0000 |
|     5 |      +0.2968 |  0.0000 |
|     6 |      +0.3341 |  0.0000 |
|     7 |      +0.3640 |  0.0000 |
|     8 |      +0.3918 |  0.0000 |
|     9 |      +0.4108 |  0.0000 |
|    10 |      +0.3961 |  0.0000 |
|    11 |      +0.4071 |  0.0000 |
|    12 |      +0.4482 |  0.0000 |
|    13 |      +0.4716 |  0.0000 |
|    14 |      +0.4716 |  0.0000 |
|    15 |      +0.4833 |  0.0000 |
|    16 |      +0.5638 |  0.0000 |
|    17 |      +0.5406 |  0.0000 |
|    18 |      +0.5268 |  0.0000 |
|    19 |      +0.5260 |  0.0000 |
|    20 |      +0.5250 |  0.0000 |
|    21 |      +0.5092 |  0.0000 |
|    22 |      +0.5104 |  0.0000 |
|    23 |      +0.5141 |  0.0000 |
|    24 |      +0.5154 |  0.0000 |
|    25 |      +0.5151 |  0.0000 |
|    26 |      +0.5102 |  0.0000 |
|    27 |      +0.5101 |  0.0000 |
|    28 |      +0.5088 |  0.0000 |
|    29 |      +0.5129 |  0.0000 |
|    30 |      +0.5138 |  0.0000 |
|    31 |      +0.5099 |  0.0000 |
|    32 |      +0.5116 |  0.0000 |
|    33 |      +0.5137 |  0.0000 |
|    34 |      +0.5074 |  0.0000 |
|    35 |      +0.5038 |  0.0000 |
|    36 |      +0.5084 |  0.0000 |
|    37 |      +0.5094 |  0.0000 |
|    38 |      +0.5100 |  0.0000 |
|    39 |      +0.5090 |  0.0000 |
|    40 |      +0.5369 |  0.0000 |

