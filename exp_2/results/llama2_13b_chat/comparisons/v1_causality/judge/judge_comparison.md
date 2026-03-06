# LLM Judge Comparison — Partner Identity vs Control





**Generated:** 2026-03-04 20:38:47


**Judge model:** claude-sonnet-4-20250514


**Versions compared:** Balanced GPT (Partner Identity), Nonsense Codeword (Control)


**Purpose:** Compare LLM judge accuracy between the partner-identity version
        (balanced_gpt) and the control version (nonsense_codeword) to assess whether causal
        interventions produce detectably different behavior only when meaningful partner labels are used.







### Table of Contents


            - Overview Comparison
            - Statistical Tests
            - Comparison Chart
            - Operational Probe — Judge Accuracy

            - Metacognitive (peak) — Detailed Breakdown
            - Operational — Detailed Breakdown

            - Interpretation





## Overview Comparison


| Probe Type | Strength | Partner Identity Accuracy | Partner Identity n correct / n | Partner Identity p (vs chance) | Control Accuracy | Control n correct / n | Control p (vs chance) | Difference | Fisher p (between versions) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Metacognitive (peak)** | 4 | 40.0% | 24 / 60 | 0.9538 n.s. | 46.7% | 28 / 60 | 0.7405 n.s. | -6.7% | 0.5807 n.s. |
| **Operational** | 4 | 71.7% | 43 / 60 | 0.0005 *** | 43.3% | 26 / 60 | 0.8775 n.s. | +28.3% | 0.0030 ** |



## Statistical Tests


Two-sided tests comparing accuracy between Partner Identity (balanced_gpt)
    and Control (nonsense_codeword) for each probe type.


| Probe Type | Partner Identity Accuracy | Control Accuracy | Difference (PI - Ctrl) | SE | Fisher's Exact OR | Fisher's Exact p-value | &chi;&sup2; (Yates) p-value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Metacognitive (peak)** (str=4) | 40.0% (24/60) | 46.7% (28/60) | -6.7% | 0.090 | 0.76 | 0.5807 n.s. | 0.5805 n.s. |
| **Operational** (str=4) | 71.7% (43/60) | 43.3% (26/60) | +28.3% | 0.086 | 3.31 | 0.0030 ** | 0.0031 ** |


## Comparison Chart


*[Figure: Cross-version comparison chart — see HTML report]*


## Operational Probe — Judge Accuracy


*[Figure: Operational probe judge accuracy — see HTML report]*





## Metacognitive (peak) — Detailed Breakdown (Strength 4)



*[Figure: Metacognitive (peak) breakdown — see HTML report]*



### Partner Identity




                Accuracy: 40.0% n.s.

            n = 60
            Failed: 0




            Position bias: Human-first 38.9% (14/36), AI-first 41.7% (10/24)




### Control




                Accuracy: 46.7% n.s.

            n = 60
            Failed: 0




            Position bias: Human-first 55.6% (20/36), AI-first 33.3% (8/24)









## Operational — Detailed Breakdown (Strength 4)



*[Figure: Operational breakdown — see HTML report]*



### Partner Identity




                Accuracy: 71.7% ***

            n = 60
            Failed: 0




            Position bias: Human-first 72.2% (26/36), AI-first 70.8% (17/24)




### Control




                Accuracy: 43.3% n.s.

            n = 60
            Failed: 0




            Position bias: Human-first 44.4% (16/36), AI-first 41.7% (10/24)









## Interpretation



            - **Metacognitive (peak) — Partner Identity:** Judge accuracy (40.0%) does not significantly differ from chance (p = 0.9538).
            - **Metacognitive (peak) — Control:** Judge accuracy (46.7%) does not significantly differ from chance (p = 0.7405).
            - **Metacognitive (peak) — Cross-version:** The -6.7% accuracy difference between versions is not significant (Fisher p = 0.5807).
            - **Operational — Partner Identity:** Judge accuracy (71.7%) is significantly above chance (p = 0.0005).
            - **Operational — Control:** Judge accuracy (43.3%) does not significantly differ from chance (p = 0.8775).
            - **Operational — Cross-version:** The +28.3% accuracy difference between versions is statistically significant (Fisher p = 0.0030).




**Key finding:** At least one probe type shows a statistically significant difference in judge accuracy between the partner-identity and control versions, suggesting the intervention has a detectable, version-specific effect.
