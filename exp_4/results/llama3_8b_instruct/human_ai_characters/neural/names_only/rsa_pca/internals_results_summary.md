# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-28 09:50:23
**Model:** LLaMA-3-8B-Instruct

---

## What is being tested

Does LLaMA-3-8B-Instruct's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 18, rho = +0.7618, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.2284 |  0.0000 |
|     2 |      +0.1906 |  0.0001 |
|     3 |      +0.2474 |  0.0000 |
|     4 |      +0.3899 |  0.0000 |
|     5 |      +0.4061 |  0.0000 |
|     6 |      +0.4183 |  0.0000 |
|     7 |      +0.4253 |  0.0000 |
|     8 |      +0.5558 |  0.0000 |
|     9 |      +0.6147 |  0.0000 |
|    10 |      +0.6271 |  0.0000 |
|    11 |      +0.6599 |  0.0000 |
|    12 |      +0.6891 |  0.0000 |
|    13 |      +0.6940 |  0.0000 |
|    14 |      +0.7263 |  0.0000 |
|    15 |      +0.7353 |  0.0000 |
|    16 |      +0.7555 |  0.0000 |
|    17 |      +0.7550 |  0.0000 |
|    18 |      +0.7618 |  0.0000 |
|    19 |      +0.7562 |  0.0000 |
|    20 |      +0.7586 |  0.0000 |
|    21 |      +0.7611 |  0.0000 |
|    22 |      +0.7566 |  0.0000 |
|    23 |      +0.7482 |  0.0000 |
|    24 |      +0.6894 |  0.0000 |
|    25 |      +0.6603 |  0.0000 |
|    26 |      +0.6536 |  0.0000 |
|    27 |      +0.6456 |  0.0000 |
|    28 |      +0.6361 |  0.0000 |
|    29 |      +0.6390 |  0.0000 |
|    30 |      +0.6234 |  0.0000 |
|    31 |      +0.6135 |  0.0000 |
|    32 |      +0.6085 |  0.0000 |

