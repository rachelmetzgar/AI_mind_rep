# LLM Judge Report — Balanced GPT (Partner Identity)





**Generated:** 2026-03-04 16:26:05


**Version:** balanced_gpt


**Judge files found:** 2


**Judge model:** claude-sonnet-4-20250514


**Purpose:** Evaluate whether an LLM judge can distinguish
        human-steered vs AI-steered LLaMA-2 responses after causal intervention.







### Table of Contents


            - Summary Table
            - Comparison Chart

            - Peak 15 / Metacognitive (peak) / Strength 4
            - Peak 15 / Operational / Strength 4






## Summary Table


| Strategy | Probe Type | Strength | Success Rate | n_correct / n_judged | p-value (binomial) | Position Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **peak_15** | Metacognitive (peak) | 4 | 40.0% | 24 / 60 | 0.9538 n.s. | H1st: 14/36 (39%) A1st: 10/24 (42%) |
| **peak_15** | Operational | 4 | 71.7% | 43 / 60 | 0.0005 *** | H1st: 26/36 (72%) A1st: 17/24 (71%) |


## Comparison Chart


*[Figure: Summary comparison chart — see HTML report]*





## Peak 15 / Metacognitive (peak) / Strength 4




            Accuracy: 40.0% n.s.
            n = 60 / 60
            Failed: 0
            Model: claude-sonnet-4-20250514




*[Figure: Peak 15 / Metacognitive (peak) / Strength 4 breakdown — see HTML report]*



### Position Bias Breakdown


| Presentation Order | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| Human response first | 14 | 36 | 38.9% |
| AI response first | 10 | 24 | 41.7% |



### Target Type Breakdown


| Target Type | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| Human target | 6 | 26 | 23.1% |
| AI target | 18 | 34 | 52.9% |







## Peak 15 / Operational / Strength 4




            Accuracy: 71.7% ***
            n = 60 / 60
            Failed: 0
            Model: claude-sonnet-4-20250514




*[Figure: Peak 15 / Operational / Strength 4 breakdown — see HTML report]*



### Position Bias Breakdown


| Presentation Order | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| Human response first | 26 | 36 | 72.2% |
| AI response first | 17 | 24 | 70.8% |



### Target Type Breakdown


| Target Type | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| Human target | 13 | 26 | 50.0% |
| AI target | 30 | 34 | 88.2% |
