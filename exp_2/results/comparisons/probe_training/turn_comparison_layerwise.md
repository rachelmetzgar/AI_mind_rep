# Probe Accuracy by Layer and Conversation Turn


Experiment 2 — Llama-2-13B-chat  |  6 dataset variants
 x 5 turns x 2 probe types  |  41 layers


## 0. Peak Accuracy Summary


Peak test accuracy across all layers for each variant and turn. Green = high accuracy, red = near chance.


*[Figure: Peak accuracy heatmap — see HTML report]*


## 1. Layer Profiles by Variant


Stars mark peak accuracy layer for each turn. Dashed gray line = chance (0.5).


### Balanced GPT


*[Figure: balanced_gpt layer profiles — see HTML report]*



| Turn | Metacognitive Probe | Operational Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 5 | 1.0000 | 0.9662 | 5 | 1.0000 | 0.9721 |
| Turn 2 | 9 | 1.0000 | 0.9399 | 9 | 1.0000 | 0.9473 |
| Turn 3 | 21 | 1.0000 | 0.9285 | 8 | 1.0000 | 0.9401 |
| Turn 4 | 26 | 0.9550 | 0.8811 | 9 | 0.9575 | 0.8961 |
| Turn 5 | 40 | 0.7775 | 0.7162 | 39 | 0.7950 | 0.7365 |


### Nonsense Codeword (Control)


*[Figure: nonsense_codeword layer profiles — see HTML report]*



| Turn | Metacognitive Probe | Operational Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 7 | 1.0000 | 0.9610 | 6 | 1.0000 | 0.9634 |
| Turn 2 | 39 | 0.9850 | 0.8104 | 17 | 1.0000 | 0.8609 |
| Turn 3 | 39 | 0.8725 | 0.6759 | 37 | 0.8875 | 0.7357 |
| Turn 4 | 39 | 0.6100 | 0.5487 | 38 | 0.6225 | 0.5513 |
| Turn 5 | 30 | 0.5650 | 0.5222 | 35 | 0.5500 | 0.5170 |


---


## 2. Peak Layer Migration Across Turns


Solid lines = metacognitive probes, dashed = operational probes. Shows how the most informative layer shifts as the conversation progresses.


*[Figure: Peak layer shift — see HTML report]*


---


## 3. Cross-Variant Comparison (All Turns)


Overlays all 6 dataset variants at each conversation turn (Turn 1 through Turn 5). Stars mark peak layers.


*[Figure: Cross-variant comparison — see HTML report]*


---


Generated 2026-02-22  |
Data: exp_2/data/{variant}/probe_checkpoints/turn_{N}/{probe}/accuracy_summary.pkl
