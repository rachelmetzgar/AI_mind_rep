# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-29 10:58:15
**Model:** Qwen-2.5-7B (Base)

---

## What is being tested

Does Qwen-2.5-7B (Base)'s internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

**Peak:** Layer 22, rho = +0.7010, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1884 |  0.0001 |
|     2 |      +0.2755 |  0.0000 |
|     3 |      +0.2869 |  0.0000 |
|     4 |      +0.4133 |  0.0000 |
|     5 |      +0.4892 |  0.0000 |
|     6 |      +0.4876 |  0.0000 |
|     7 |      +0.5604 |  0.0000 |
|     8 |      +0.6469 |  0.0000 |
|     9 |      +0.6644 |  0.0000 |
|    10 |      +0.6298 |  0.0000 |
|    11 |      +0.6523 |  0.0000 |
|    12 |      +0.6909 |  0.0000 |
|    13 |      +0.6786 |  0.0000 |
|    14 |      +0.6807 |  0.0000 |
|    15 |      +0.6664 |  0.0000 |
|    16 |      +0.6696 |  0.0000 |
|    17 |      +0.6778 |  0.0000 |
|    18 |      +0.6961 |  0.0000 |
|    19 |      +0.6786 |  0.0000 |
|    20 |      +0.6909 |  0.0000 |
|    21 |      +0.6993 |  0.0000 |
|    22 |      +0.7010 |  0.0000 |
|    23 |      +0.6948 |  0.0000 |
|    24 |      +0.6958 |  0.0000 |
|    25 |      +0.6879 |  0.0000 |
|    26 |      +0.6979 |  0.0000 |
|    27 |      +0.6891 |  0.0000 |
|    28 |      +0.6567 |  0.0000 |

