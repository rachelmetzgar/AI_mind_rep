# Experiment 7 — Mental-State Concept Deployment During Theory of Mind Reasoning

## Motivation

Experiments 1-3 establish that LLMs (1) behaviorally differentiate between human and AI partners, (2) form linearly decodable internal representations of partner identity, and (3) possess concept-level representations of mental properties like awareness, attention, and consciousness. Separately, Wu et al. (2025, npj AI) showed that ToM task performance depends on a sparse set of parameters concentrated in the positional encoding module — suggesting ToM relies on tracking which information is accessible to which entity based on narrative position.

These two lines of evidence address different aspects of the same question. The positional-encoding work shows the model can track *who knows what* structurally, but doesn't ask whether mental-state concepts are involved. The previous work shows the model *has* representations of mental properties, but doesn't test whether those representations are deployed during social reasoning.

Exp 7 bridges these approaches: **does the model activate mental-state concept representations (awareness, attention, consciousness) in a context-sensitive way that tracks characters' knowledge states during theory of mind tasks?**

## Core Hypothesis

When processing false belief scenarios, the model's internal activations will show differential projection onto mental-state concept vectors depending on whether a character is in a state of knowledge or ignorance. Specifically:

- **Awareness/attention/consciousness** concept projections will be higher (or directionally different) at positions where a character has access to information
- These projections will diminish or shift when the character is absent during a state change (the false belief condition)
- **Control concepts** (shapes, embodiment, biology) will show no such context-dependent modulation
- This pattern will be specific to the character whose knowledge state changes — not a global shift in the model's activations

## Design

### Stimuli

Classic false belief tasks adapted for LLM processing:

- **Sally-Anne task variants**: Character A places object in Location 1. Character A leaves. Character B moves object to Location 2. Character A returns. Extract activations at key narrative positions.
- **Unexpected contents**: Character shown a container, asked what's inside. Container revealed to hold something unexpected. Character asked what a naive observer would think is inside.
- Multiple variants with different characters, objects, and locations to avoid stimulus-specific artifacts.

### Key extraction positions

For each scenario, extract hidden state activations at positions corresponding to:
1. **Baseline**: Before any knowledge differential exists
2. **Shared knowledge**: Both characters know the object's location
3. **Knowledge divergence**: Character A has left; state change occurs
4. **False belief**: Character A returns; model must represent that A's belief differs from reality

### Concept vectors

Reuse Exp 3 concept vectors, focusing on:
- **Mental-state concepts** (hypothesized to modulate): awareness, attention, consciousness, mind, prediction, intentions
- **Control concepts** (hypothesized to be stable): shapes, embodiment, biology, formality

### Analysis

1. **Projection analysis**: At each extraction position, project hidden states onto each concept vector. Test whether mental-state projections differ between knowledge/ignorance conditions.
2. **Layer profiles**: Identify which layers show the strongest context-dependent modulation — compare to Exp 3 alignment profiles and Exp 2 probe accuracy profiles.
3. **Concept specificity**: Confirm that modulation is specific to mental-state concepts, not a general activation shift.
4. **Character specificity**: Confirm that modulation tracks the specific character whose knowledge changes, not the scenario globally.

### Potential extensions

- **Causal test**: Steer mental-state concept vectors during false belief processing and test whether this changes the model's answer about what the character believes.
- **Scaling**: Run on larger models (Qwen-2.5-32B) to test whether concept deployment during ToM becomes more structured with scale.
- **Cross-experiment link**: Correlate which concept dimensions modulate during ToM with which dimensions align with Exp 2 partner probes. If the same mental-property dimensions that encode partner identity are also deployed during ToM reasoning, this would suggest a shared underlying "mind model."

## Relation to Other Experiments

| Experiment | What it shows | What Exp 7 adds |
|---|---|---|
| Exp 1 | LLMs adjust behavior based on partner identity labels | Exp 7 tests whether mental-state concepts are deployed during social reasoning, not just identity-conditioned behavior |
| Exp 2 | Partner identity is linearly decodable and causally active | Exp 7 tests whether the same representational space is used during ToM tasks |
| Exp 3 | LLMs have concept-level representations of mental properties | Exp 7 tests whether those representations are *used* during reasoning, not just *present* |
| Exp 4 | LLMs have implicit mind perception geometry | Exp 7 tests dynamic deployment of mind concepts, not static entity ratings |
| Wu et al. | ToM depends on positional encoding parameters | Exp 7 tests whether mental-state representations ride on top of that positional structure |

## Model

- Primary: LLaMA-2-13B-Chat (for comparability with Exps 2-3)
- Extension: Qwen-2.5-32B-Instruct (scaling test, requires 2x A100-40GB)

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619.

Wu, Y., Guo, W., Liu, Z., Ji, H., Xu, Z., & Zhang, D. (2025). How large language models encode theory-of-mind: a study on sparse parameter patterns. *npj Artificial Intelligence*, 2(1).

Wimmer, H., & Perner, J. (1983). Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1), 103-128.
