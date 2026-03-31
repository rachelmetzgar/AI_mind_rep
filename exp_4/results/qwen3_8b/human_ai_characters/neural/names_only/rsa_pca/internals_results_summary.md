# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-28 12:56:49
**Model:** Qwen3-8B

---

## What is being tested

Does Qwen3-8B's internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 21, rho = +0.8390, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1747 |  0.0003 |
|     2 |      +0.2144 |  0.0000 |
|     3 |      +0.2670 |  0.0000 |
|     4 |      +0.4090 |  0.0000 |
|     5 |      +0.4207 |  0.0000 |
|     6 |      +0.3912 |  0.0000 |
|     7 |      +0.3583 |  0.0000 |
|     8 |      +0.3779 |  0.0000 |
|     9 |      +0.4565 |  0.0000 |
|    10 |      +0.5391 |  0.0000 |
|    11 |      +0.5702 |  0.0000 |
|    12 |      +0.5858 |  0.0000 |
|    13 |      +0.6797 |  0.0000 |
|    14 |      +0.7183 |  0.0000 |
|    15 |      +0.7454 |  0.0000 |
|    16 |      +0.7502 |  0.0000 |
|    17 |      +0.7429 |  0.0000 |
|    18 |      +0.7390 |  0.0000 |
|    19 |      +0.7821 |  0.0000 |
|    20 |      +0.8244 |  0.0000 |
|    21 |      +0.8390 |  0.0000 |
|    22 |      +0.8353 |  0.0000 |
|    23 |      +0.8352 |  0.0000 |
|    24 |      +0.8203 |  0.0000 |
|    25 |      +0.7986 |  0.0000 |
|    26 |      +0.7954 |  0.0000 |
|    27 |      +0.8029 |  0.0000 |
|    28 |      +0.8033 |  0.0000 |
|    29 |      +0.8072 |  0.0000 |
|    30 |      +0.8037 |  0.0000 |
|    31 |      +0.7861 |  0.0000 |
|    32 |      +0.7819 |  0.0000 |
|    33 |      +0.7889 |  0.0000 |
|    34 |      +0.7906 |  0.0000 |
|    35 |      +0.7607 |  0.0000 |
|    36 |      +0.7388 |  0.0000 |

