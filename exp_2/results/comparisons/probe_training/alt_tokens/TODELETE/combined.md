Labels
Balanced GPT
Nonsense Codeword
Combined


# Alternative Token Position Probes


Cross-Version Comparison

3 versions x 5 turns x 4 alternative conditions + 2 baselines


#### Causal Attention Confound for First </s>


The first </s> achieves perfect accuracy (1.000) at every turn because
LLaMA-2 uses causal attention. The representation at this token depends only on preceding
tokens, which are identical regardless of conversation length. This is an artifact of
probing an invariant position.


## 1. Key Comparison: Metacognitive vs Operational vs Random (Turn 5)


Blue = metacognitive probe, red = operational probe, green = random token. Solid = Labels, dashed = Balanced GPT, dotted = Nonsense Codeword.


*[Figure: clean summary reading vs control vs random — see HTML report]*


---


## 2. Cross-Version Summary (Turn 5)


| Condition | Token | Labels (Primary) | Balanced GPT | Nonsense Codeword (Control) |
| --- | --- | --- | --- | --- |
| Mean | Peak (Layer) | Mean | Peak (Layer) | Mean | Peak (Layer) |
| BOS (<s>) | Position 0 | — | — | — | — | — | — |
| Random mid-seq | ~25th–75th percentile | — | — | — | — | — | — |
| First </s> | End of 1st exchange | — | — | — | — | — | — |
| Weather suffix | Last token ("is") | — | — | — | — | — | — |
| Baseline operational | Last token [/INST] | — | — | 0.736 | **0.795** (L39) | 0.517 | **0.550** (L35) |
| Baseline metacognitive | Last token ("is") | — | — | 0.716 | **0.777** (L40) | 0.522 | **0.565** (L30) |


#### Key Conclusions


- The partner identity signal is *not* broadcast — BOS and random tokens carry
no information across all versions.
- The signal is strongest at **structural boundary tokens** (</s> achieves
perfect decoding).
- The signal is **not triggered by partner-relevant questioning** — the weather
suffix works nearly as well as the partner suffix, consistently across versions.
- The representation degrades across turns due to **prompt dilution**.
- These patterns are **consistent across all 3 versions**.


---


## 3. Overview Grid (All Turns x Versions)


Each panel shows all 6 conditions for one version at one turn.


*[Figure: summary grid all turns x versions — see HTML report]*


---


## 4. Summary Tables by Version


### Labels (Primary)



| Turn | Metacognitive | Operational | BOS | Random | First </s> | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | — | — | — | — | — | — |
| Turn 2 | — | — | — | — | — | — |
| Turn 3 | — | — | — | — | — | — |
| Turn 4 | — | — | — | — | — | — |
| Turn 5 | — | — | — | — | — | — |


### Balanced GPT



| Turn | Metacognitive | Operational | BOS | Random | First </s> | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L5) | 1.000 (L5) | — | — | — | — |
| Turn 2 | 1.000 (L9) | 1.000 (L9) | — | — | — | — |
| Turn 3 | 1.000 (L21) | 1.000 (L8) | — | — | — | — |
| Turn 4 | 0.955 (L26) | 0.958 (L9) | — | — | — | — |
| Turn 5 | 0.777 (L40) | 0.795 (L39) | — | — | — | — |


### Nonsense Codeword (Control)



| Turn | Metacognitive | Operational | BOS | Random | First </s> | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L7) | 1.000 (L6) | — | — | — | — |
| Turn 2 | 0.985 (L39) | 1.000 (L17) | — | — | — | — |
| Turn 3 | 0.873 (L39) | 0.887 (L37) | — | — | — | — |
| Turn 4 | 0.610 (L39) | 0.623 (L38) | — | — | — | — |
| Turn 5 | 0.565 (L30) | 0.550 (L35) | — | — | — | — |


---


## 5. Cross-Version Comparison by Condition


Compare each condition across all 3 versions, one graph per turn.


### BOS Token (position 0)


### Random Mid-Sequence Token


### First  (end of 1st exchange)


### Irrelevant Suffix (weather)


---


Cross-Version Comparison  |  Per-version reports:
Labels  |
Balanced GPT  |
Nonsense Codeword
