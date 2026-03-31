# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:54:41
**Tag:** `without_self`
**Include self:** False
**Model:** Qwen-2.5-7B (Base)

---

## What is being tested

Does Qwen-2.5-7B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Qwen-2.5-7B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDMs**: Three variants from Gray et al. (2007) factor scores:
  - Combined: Euclidean distance in 2D (Experience, Agency) space
  - Experience-only: |exp_i - exp_j|
  - Agency-only: |agency_i - agency_j|

**Test:** Spearman rank correlation between the upper triangles of the model RDM and each human RDM (66 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

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

## RSA by Layer -- Combined

**Peak:** Layer 19, rho = +0.3103, p = 0.0112

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0337 |  0.7881 |
|     2 |      -0.0549 |  0.6616 |
|     3 |      -0.0944 |  0.4510 |
|     4 |      +0.0030 |  0.9810 |
|     5 |      +0.0464 |  0.7116 |
|     6 |      +0.0083 |  0.9473 |
|     7 |      +0.0348 |  0.7812 |
|     8 |      +0.1576 |  0.2064 |
|     9 |      +0.1532 |  0.2194 |
|    10 |      +0.1445 |  0.2471 |
|    11 |      +0.1436 |  0.2499 |
|    12 |      +0.1644 |  0.1872 |
|    13 |      +0.1606 |  0.1977 |
|    14 |      +0.1554 |  0.2128 |
|    15 |      +0.2371 |  0.0553 |
|    16 |      +0.2208 |  0.0748 |
|    17 |      +0.2671 |  0.0302 |
|    18 |      +0.2656 |  0.0311 |
|    19 |      +0.3103 |  0.0112 |
|    20 |      +0.2322 |  0.0606 |
|    21 |      +0.1747 |  0.1607 |
|    22 |      +0.1233 |  0.3239 |
|    23 |      +0.0779 |  0.5342 |
|    24 |      +0.0358 |  0.7756 |
|    25 |      +0.0279 |  0.8242 |
|    26 |      +0.0062 |  0.9606 |
|    27 |      +0.0253 |  0.8403 |
|    28 |      +0.0596 |  0.6343 |

## RSA by Layer -- Experience

**Peak:** Layer 19, rho = +0.1002, p = 0.4236

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1784 |  0.1519 |
|     2 |      -0.1320 |  0.2907 |
|     3 |      -0.1446 |  0.2467 |
|     4 |      -0.0858 |  0.4934 |
|     5 |      -0.0695 |  0.5793 |
|     6 |      -0.0920 |  0.4625 |
|     7 |      -0.0773 |  0.5375 |
|     8 |      +0.0419 |  0.7383 |
|     9 |      +0.0428 |  0.7330 |
|    10 |      +0.0338 |  0.7877 |
|    11 |      +0.0118 |  0.9254 |
|    12 |      +0.0182 |  0.8848 |
|    13 |      +0.0146 |  0.9073 |
|    14 |      +0.0084 |  0.9468 |
|    15 |      +0.0571 |  0.6488 |
|    16 |      +0.0395 |  0.7529 |
|    17 |      +0.0744 |  0.5526 |
|    18 |      +0.0724 |  0.5633 |
|    19 |      +0.1002 |  0.4236 |
|    20 |      +0.0520 |  0.6784 |
|    21 |      +0.0214 |  0.8646 |
|    22 |      -0.0045 |  0.9716 |
|    23 |      -0.0224 |  0.8582 |
|    24 |      -0.0318 |  0.7999 |
|    25 |      -0.0437 |  0.7276 |
|    26 |      -0.0733 |  0.5586 |
|    27 |      -0.0964 |  0.4414 |
|    28 |      -0.0457 |  0.7158 |

## RSA by Layer -- Agency

**Peak:** Layer 19, rho = +0.2521, p = 0.0411

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1120 |  0.3706 |
|     2 |      -0.0292 |  0.8159 |
|     3 |      -0.1003 |  0.4229 |
|     4 |      -0.0459 |  0.7141 |
|     5 |      -0.0385 |  0.7592 |
|     6 |      -0.0659 |  0.5988 |
|     7 |      -0.0229 |  0.8554 |
|     8 |      +0.0490 |  0.6959 |
|     9 |      +0.0380 |  0.7620 |
|    10 |      +0.0295 |  0.8143 |
|    11 |      +0.0256 |  0.8385 |
|    12 |      +0.0570 |  0.6493 |
|    13 |      +0.0609 |  0.6272 |
|    14 |      +0.0578 |  0.6449 |
|    15 |      +0.1640 |  0.1884 |
|    16 |      +0.1607 |  0.1974 |
|    17 |      +0.2165 |  0.0808 |
|    18 |      +0.2098 |  0.0909 |
|    19 |      +0.2521 |  0.0411 |
|    20 |      +0.1651 |  0.1853 |
|    21 |      +0.0935 |  0.4551 |
|    22 |      +0.0492 |  0.6948 |
|    23 |      +0.0267 |  0.8315 |
|    24 |      -0.0062 |  0.9603 |
|    25 |      +0.0071 |  0.9552 |
|    26 |      -0.0149 |  0.9052 |
|    27 |      +0.0039 |  0.9749 |
|    28 |      +0.0193 |  0.8778 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
