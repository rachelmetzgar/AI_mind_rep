# Concept Geometry, Phase B: Internal RSA

**Run:** 2026-03-08 17:48:39
**Model:** LLaMA-2-13B (Base)

---

## What is being tested

Does LLaMA-2-13B (Base)'s internal representational geometry over 28 characters (14 AI, 14 human) reflect their categorical distinction? RSA compares the model's activation-based RDM to reference RDMs at every layer.

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

## RSA by Layer -- behavioral

**Peak:** Layer 11, rho = +0.3344, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0049 |  0.9186 |
|     2 |      +0.0093 |  0.8472 |
|     3 |      +0.0216 |  0.6538 |
|     4 |      +0.1914 |  0.0001 |
|     5 |      +0.1592 |  0.0009 |
|     6 |      +0.1744 |  0.0003 |
|     7 |      +0.2479 |  0.0000 |
|     8 |      +0.2958 |  0.0000 |
|     9 |      +0.2957 |  0.0000 |
|    10 |      +0.3197 |  0.0000 |
|    11 |      +0.3344 |  0.0000 |
|    12 |      +0.3266 |  0.0000 |
|    13 |      +0.3104 |  0.0000 |
|    14 |      +0.2942 |  0.0000 |
|    15 |      +0.2857 |  0.0000 |
|    16 |      +0.2830 |  0.0000 |
|    17 |      +0.2670 |  0.0000 |
|    18 |      +0.2719 |  0.0000 |
|    19 |      +0.2692 |  0.0000 |
|    20 |      +0.2534 |  0.0000 |
|    21 |      +0.2614 |  0.0000 |
|    22 |      +0.2482 |  0.0000 |
|    23 |      +0.2549 |  0.0000 |
|    24 |      +0.2388 |  0.0000 |
|    25 |      +0.2526 |  0.0000 |
|    26 |      +0.2477 |  0.0000 |
|    27 |      +0.2415 |  0.0000 |
|    28 |      +0.2352 |  0.0000 |
|    29 |      +0.2334 |  0.0000 |
|    30 |      +0.2447 |  0.0000 |
|    31 |      +0.2431 |  0.0000 |
|    32 |      +0.2409 |  0.0000 |
|    33 |      +0.2376 |  0.0000 |
|    34 |      +0.2384 |  0.0000 |
|    35 |      +0.2372 |  0.0000 |
|    36 |      +0.2319 |  0.0000 |
|    37 |      +0.2357 |  0.0000 |
|    38 |      +0.2340 |  0.0000 |
|    39 |      +0.2336 |  0.0000 |
|    40 |      +0.2161 |  0.0000 |

## RSA by Layer -- categorical

**Peak:** Layer 8, rho = +0.6616, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1104 |  0.0212 |
|     2 |      +0.1153 |  0.0161 |
|     3 |      +0.1218 |  0.0110 |
|     4 |      +0.3077 |  0.0000 |
|     5 |      +0.3702 |  0.0000 |
|     6 |      +0.5247 |  0.0000 |
|     7 |      +0.5986 |  0.0000 |
|     8 |      +0.6616 |  0.0000 |
|     9 |      +0.6277 |  0.0000 |
|    10 |      +0.6212 |  0.0000 |
|    11 |      +0.6143 |  0.0000 |
|    12 |      +0.6423 |  0.0000 |
|    13 |      +0.6222 |  0.0000 |
|    14 |      +0.6083 |  0.0000 |
|    15 |      +0.5853 |  0.0000 |
|    16 |      +0.5755 |  0.0000 |
|    17 |      +0.5838 |  0.0000 |
|    18 |      +0.5951 |  0.0000 |
|    19 |      +0.5933 |  0.0000 |
|    20 |      +0.5816 |  0.0000 |
|    21 |      +0.5811 |  0.0000 |
|    22 |      +0.5863 |  0.0000 |
|    23 |      +0.6050 |  0.0000 |
|    24 |      +0.5912 |  0.0000 |
|    25 |      +0.5913 |  0.0000 |
|    26 |      +0.5908 |  0.0000 |
|    27 |      +0.5720 |  0.0000 |
|    28 |      +0.5772 |  0.0000 |
|    29 |      +0.5765 |  0.0000 |
|    30 |      +0.5776 |  0.0000 |
|    31 |      +0.5726 |  0.0000 |
|    32 |      +0.5701 |  0.0000 |
|    33 |      +0.5682 |  0.0000 |
|    34 |      +0.5658 |  0.0000 |
|    35 |      +0.5718 |  0.0000 |
|    36 |      +0.5651 |  0.0000 |
|    37 |      +0.5689 |  0.0000 |
|    38 |      +0.5724 |  0.0000 |
|    39 |      +0.5646 |  0.0000 |
|    40 |      +0.5314 |  0.0000 |

