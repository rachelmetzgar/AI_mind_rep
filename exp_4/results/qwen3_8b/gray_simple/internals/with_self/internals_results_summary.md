# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 12:55:44
**Tag:** `with_self`
**Include self:** True
**Model:** Qwen3-8B

---

## What is being tested

Does Qwen3-8B's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Qwen3-8B in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDMs**: Three variants from Gray et al. (2007) factor scores:
  - Combined: Euclidean distance in 2D (Experience, Agency) space
  - Experience-only: |exp_i - exp_j|
  - Agency-only: |agency_i - agency_j|

**Test:** Spearman rank correlation between the upper triangles of the model RDM and each human RDM (78 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

**Interpretation:** A positive Spearman rho means entities that are far apart in human mind-perception space are also far apart in the model's activation space (and vice versa). The model's entity geometry mirrors human folk psychology.

## Entities

| Entity | Prompt | Experience | Agency |
|--------|--------|------------|--------|
| dead_woman | Think about a dead woman. | 0.06 | 0.07 |
| frog | Think about a frog. | 0.25 | 0.14 |
| robot | Think about a robot. | 0.13 | 0.22 |
| fetus | Think about a seven-week-old human fetus. | 0.17 | 0.08 |
| pvs_patient | Think about a person in a persistent vegetative state. | 0.17 | 0.10 |
| god | Think about God. | 0.20 | 0.80 |
| dog | Think about a dog. | 0.55 | 0.35 |
| chimpanzee | Think about a chimpanzee. | 0.63 | 0.48 |
| baby | Think about a five-month-old baby. | 0.71 | 0.17 |
| girl | Think about a five-year-old girl. | 0.84 | 0.62 |
| adult_woman | Think about an adult woman. | 0.93 | 0.91 |
| adult_man | Think about an adult man. | 0.91 | 0.95 |
| you_self | Think about yourself. | 0.97 | 1.00 |

## RSA by Layer -- Combined

**Peak:** Layer 30, rho = +0.3507, p = 0.0016

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0228 |  0.8429 |
|     2 |      -0.0197 |  0.8644 |
|     3 |      +0.0204 |  0.8590 |
|     4 |      +0.0805 |  0.4833 |
|     5 |      +0.1269 |  0.2680 |
|     6 |      +0.1465 |  0.2007 |
|     7 |      +0.1307 |  0.2541 |
|     8 |      +0.1394 |  0.2235 |
|     9 |      +0.1861 |  0.1028 |
|    10 |      +0.1923 |  0.0917 |
|    11 |      +0.1763 |  0.1225 |
|    12 |      +0.2164 |  0.0571 |
|    13 |      +0.2625 |  0.0203 |
|    14 |      +0.2573 |  0.0230 |
|    15 |      +0.2595 |  0.0218 |
|    16 |      +0.2389 |  0.0352 |
|    17 |      +0.2818 |  0.0125 |
|    18 |      +0.3450 |  0.0020 |
|    19 |      +0.2927 |  0.0093 |
|    20 |      +0.2786 |  0.0135 |
|    21 |      +0.2752 |  0.0148 |
|    22 |      +0.2765 |  0.0143 |
|    23 |      +0.2941 |  0.0090 |
|    24 |      +0.3063 |  0.0064 |
|    25 |      +0.3233 |  0.0039 |
|    26 |      +0.3173 |  0.0047 |
|    27 |      +0.3352 |  0.0027 |
|    28 |      +0.3442 |  0.0020 |
|    29 |      +0.3476 |  0.0018 |
|    30 |      +0.3507 |  0.0016 |
|    31 |      +0.3389 |  0.0024 |
|    32 |      +0.3358 |  0.0026 |
|    33 |      +0.3441 |  0.0020 |
|    34 |      +0.3359 |  0.0026 |
|    35 |      +0.3384 |  0.0024 |
|    36 |      +0.3040 |  0.0068 |

## RSA by Layer -- Experience

**Peak:** Layer 28, rho = +0.2775, p = 0.0139

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0604 |  0.5996 |
|     2 |      -0.0498 |  0.6649 |
|     3 |      -0.0427 |  0.7106 |
|     4 |      +0.0148 |  0.8974 |
|     5 |      +0.0494 |  0.6675 |
|     6 |      +0.0655 |  0.5690 |
|     7 |      +0.0480 |  0.6762 |
|     8 |      +0.0475 |  0.6798 |
|     9 |      +0.0801 |  0.4855 |
|    10 |      +0.0724 |  0.5289 |
|    11 |      +0.0827 |  0.4717 |
|    12 |      +0.0876 |  0.4457 |
|    13 |      +0.1164 |  0.3103 |
|    14 |      +0.1138 |  0.3213 |
|    15 |      +0.1149 |  0.3164 |
|    16 |      +0.1164 |  0.3100 |
|    17 |      +0.1579 |  0.1675 |
|    18 |      +0.2124 |  0.0619 |
|    19 |      +0.1806 |  0.1135 |
|    20 |      +0.1830 |  0.1088 |
|    21 |      +0.2029 |  0.0748 |
|    22 |      +0.2110 |  0.0637 |
|    23 |      +0.2482 |  0.0285 |
|    24 |      +0.2395 |  0.0347 |
|    25 |      +0.2466 |  0.0295 |
|    26 |      +0.2394 |  0.0348 |
|    27 |      +0.2615 |  0.0207 |
|    28 |      +0.2775 |  0.0139 |
|    29 |      +0.2714 |  0.0162 |
|    30 |      +0.2664 |  0.0184 |
|    31 |      +0.2406 |  0.0338 |
|    32 |      +0.2385 |  0.0355 |
|    33 |      +0.2416 |  0.0331 |
|    34 |      +0.2350 |  0.0384 |
|    35 |      +0.2297 |  0.0430 |
|    36 |      +0.2030 |  0.0746 |

## RSA by Layer -- Agency

**Peak:** Layer 18, rho = +0.3041, p = 0.0068

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0141 |  0.9023 |
|     2 |      +0.0687 |  0.5500 |
|     3 |      +0.0870 |  0.4489 |
|     4 |      +0.0225 |  0.8451 |
|     5 |      +0.0544 |  0.6360 |
|     6 |      +0.0839 |  0.4652 |
|     7 |      +0.0695 |  0.5457 |
|     8 |      +0.0864 |  0.4522 |
|     9 |      +0.1238 |  0.2801 |
|    10 |      +0.1361 |  0.2347 |
|    11 |      +0.1077 |  0.3478 |
|    12 |      +0.1783 |  0.1184 |
|    13 |      +0.2326 |  0.0404 |
|    14 |      +0.2131 |  0.0610 |
|    15 |      +0.2096 |  0.0655 |
|    16 |      +0.1815 |  0.1117 |
|    17 |      +0.2395 |  0.0347 |
|    18 |      +0.3041 |  0.0068 |
|    19 |      +0.2389 |  0.0351 |
|    20 |      +0.2267 |  0.0459 |
|    21 |      +0.2195 |  0.0535 |
|    22 |      +0.2199 |  0.0531 |
|    23 |      +0.2209 |  0.0519 |
|    24 |      +0.2257 |  0.0470 |
|    25 |      +0.2353 |  0.0381 |
|    26 |      +0.2252 |  0.0474 |
|    27 |      +0.2452 |  0.0305 |
|    28 |      +0.2488 |  0.0281 |
|    29 |      +0.2592 |  0.0219 |
|    30 |      +0.2694 |  0.0171 |
|    31 |      +0.2602 |  0.0214 |
|    32 |      +0.2624 |  0.0203 |
|    33 |      +0.2777 |  0.0138 |
|    34 |      +0.2751 |  0.0148 |
|    35 |      +0.2838 |  0.0118 |
|    36 |      +0.2539 |  0.0249 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
