# Alternative Token Position Probes


Comparing token position probes across 3 versions x 5 turns x 4 alternative conditions

Labels (Primary)  |  Balanced GPT  |  Nonsense Codeword (Control)


### Important: Causal Attention Confound for First </s> Token


The **first </s> token** achieves perfect accuracy (1.000) at every turn because
LLaMA-2 uses **causal (left-to-right) attention**. The model's representation at
the first </s> depends only on tokens that precede it — and those tokens are
*identical* regardless of how many conversation turns follow.


- **Turn 1**: No </s> token exists (only system + user prompt, no assistant response).
The code falls back to the last token (`]` from `[/INST]`), making this
condition identical to the baseline control probe.
- **Turns 2–5**: The text before the first </s> is *exactly identical*
across all turns (same 1028 chars). The first </s> sits at the same position with the same
causal context. Subsequent turns cannot influence it because information flows only left-to-right.


The constant perfect accuracy is therefore an artifact of probing an invariant position, not evidence
that partner identity is preserved across turns. It remains true that this structural boundary token
is an exceptionally informative position — but its accuracy does not address the prompt dilution
question.


## 1. Overview Grid (All Turns x Versions)


Each panel shows all 6 conditions (2 baselines + 4 alternative) for one version at one turn. Stars mark peak layers.


*[Figure: Summary grid all turns x versions — see HTML report]*


---


## 2. Summary Table


Peak accuracy and layer for each condition, version, and turn. Values at chance (&le;0.55) shown in red; above baseline (>0.70) in green.



#### Labels (Primary)


| Turn | Baseline Reading | Baseline Control | BOS | Random | First </s> | Weather Suffix |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L7) | 1.000 (L6) | 0.505 (L6) | 0.950 (L38) | 1.000 (L5) | 1.000 (L10) |
| Turn 2 | 1.000 (L14) | 1.000 (L15) | 0.527 (L11) | 0.677 (L18) | 1.000 (L33) | 0.968 (L39) |
| Turn 3 | 1.000 (L22) | 0.963 (L17) | 0.555 (L4) | 0.562 (L33) | 1.000 (L33) | 0.775 (L22) |
| Turn 4 | 0.892 (L35) | 0.738 (L15) | 0.530 (L39) | 0.570 (L15) | 1.000 (L33) | 0.625 (L22) |
| Turn 5 | 0.652 (L33) | 0.605 (L31) | 0.510 (L6) | 0.560 (L14) | 1.000 (L33) | 0.600 (L33) |




#### Balanced GPT


| Turn | Baseline Reading | Baseline Control | BOS | Random | First </s> | Weather Suffix |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L7) | 1.000 (L5) | 0.590 (L12) | 0.978 (L35) | 1.000 (L4) | 1.000 (L8) |
| Turn 2 | 1.000 (L9) | 1.000 (L9) | 0.590 (L34) | 0.807 (L21) | 1.000 (L24) | 1.000 (L14) |
| Turn 3 | 1.000 (L9) | 1.000 (L8) | 0.562 (L4) | 0.743 (L25) | 1.000 (L24) | 1.000 (L38) |
| Turn 4 | 0.960 (L34) | 0.958 (L9) | 0.555 (L39) | 0.698 (L22) | 1.000 (L24) | 0.950 (L13) |
| Turn 5 | 0.780 (L39) | 0.795 (L39) | 0.522 (L4) | 0.672 (L23) | 1.000 (L24) | 0.777 (L39) |




#### Nonsense Codeword (Control)


| Turn | Baseline Reading | Baseline Control | BOS | Random | First </s> | Weather Suffix |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L7) | 1.000 (L6) | 0.505 (L4) | 0.948 (L38) | 1.000 (L6) | 1.000 (L11) |
| Turn 2 | 1.000 (L37) | 1.000 (L17) | 0.537 (L9) | 0.578 (L36) | 1.000 (L34) | 0.998 (L40) |
| Turn 3 | 0.998 (L40) | 0.887 (L37) | 0.520 (L11) | 0.575 (L29) | 1.000 (L34) | 0.882 (L40) |
| Turn 4 | 0.835 (L40) | 0.623 (L38) | 0.555 (L9) | 0.595 (L29) | 1.000 (L34) | 0.585 (L33) |
| Turn 5 | 0.562 (L19) | 0.550 (L35) | 0.507 (L4) | 0.568 (L29) | 1.000 (L34) | 0.555 (L19) |



---


## 3. Per-Version Detail (All Conditions by Turn)


Each graph shows all conditions for one version at one turn.


### Labels (Primary)


*[Figure: labels turn 1 — see HTML report]*


*[Figure: labels turn 2 — see HTML report]*


*[Figure: labels turn 3 — see HTML report]*


*[Figure: labels turn 4 — see HTML report]*


*[Figure: labels turn 5 — see HTML report]*


### Balanced GPT


*[Figure: balanced_gpt turn 1 — see HTML report]*


*[Figure: balanced_gpt turn 2 — see HTML report]*


*[Figure: balanced_gpt turn 3 — see HTML report]*


*[Figure: balanced_gpt turn 4 — see HTML report]*


*[Figure: balanced_gpt turn 5 — see HTML report]*


### Nonsense Codeword (Control)


*[Figure: nonsense_codeword turn 1 — see HTML report]*


*[Figure: nonsense_codeword turn 2 — see HTML report]*


*[Figure: nonsense_codeword turn 3 — see HTML report]*


*[Figure: nonsense_codeword turn 4 — see HTML report]*


*[Figure: nonsense_codeword turn 5 — see HTML report]*


---


## 4. Cross-Version Comparison by Condition


Compare each alternative condition across all 3 versions, one graph per turn.


### BOS Token (position 0)


*[Figure: control_first turn 1 cross-version — see HTML report]*


*[Figure: control_first turn 2 cross-version — see HTML report]*


*[Figure: control_first turn 3 cross-version — see HTML report]*


*[Figure: control_first turn 4 cross-version — see HTML report]*


*[Figure: control_first turn 5 cross-version — see HTML report]*


### Random Mid-Sequence Token


*[Figure: control_random turn 1 cross-version — see HTML report]*


*[Figure: control_random turn 2 cross-version — see HTML report]*


*[Figure: control_random turn 3 cross-version — see HTML report]*


*[Figure: control_random turn 4 cross-version — see HTML report]*


*[Figure: control_random turn 5 cross-version — see HTML report]*


### First  (end of 1st exchange)


*[Figure: control_eos turn 1 cross-version — see HTML report]*


*[Figure: control_eos turn 2 cross-version — see HTML report]*


*[Figure: control_eos turn 3 cross-version — see HTML report]*


*[Figure: control_eos turn 4 cross-version — see HTML report]*


*[Figure: control_eos turn 5 cross-version — see HTML report]*


### Irrelevant Suffix (weather)


*[Figure: reading_irrelevant turn 1 cross-version — see HTML report]*


*[Figure: reading_irrelevant turn 2 cross-version — see HTML report]*


*[Figure: reading_irrelevant turn 3 cross-version — see HTML report]*


*[Figure: reading_irrelevant turn 4 cross-version — see HTML report]*


*[Figure: reading_irrelevant turn 5 cross-version — see HTML report]*


---


## 5. Turn Progression by Condition


How each condition changes across turns 1-5 for each version. Shows the prompt dilution effect.


### Labels (Primary)


*[Figure: labels control_first turn progression — see HTML report]*


*[Figure: labels control_random turn progression — see HTML report]*


*[Figure: labels control_eos turn progression — see HTML report]*


*[Figure: labels reading_irrelevant turn progression — see HTML report]*


### Balanced GPT


*[Figure: balanced_gpt control_first turn progression — see HTML report]*


*[Figure: balanced_gpt control_random turn progression — see HTML report]*


*[Figure: balanced_gpt control_eos turn progression — see HTML report]*


*[Figure: balanced_gpt reading_irrelevant turn progression — see HTML report]*


### Nonsense Codeword (Control)


*[Figure: nonsense_codeword control_first turn progression — see HTML report]*


*[Figure: nonsense_codeword control_random turn progression — see HTML report]*


*[Figure: nonsense_codeword control_eos turn progression — see HTML report]*


*[Figure: nonsense_codeword reading_irrelevant turn progression — see HTML report]*


---


Generated  |
Data: exp_2/data/{version}/probe_checkpoints/alternative[_turn_N]/{condition}/accuracy_summary.pkl
