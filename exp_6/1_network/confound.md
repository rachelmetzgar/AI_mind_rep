# Experiment 6: Position Confound

## The Problem

In Experiment 6 (Network/Displacement), the position RDM dominated the epistemic RDM at every layer:

- r_position = **0.61** at peak
- r_epistemic = **0.45** at peak
- r_communication = **0.18** at peak

Position was never beaten by epistemic RSA at any of the 40 layers.

## Why This Happened

All four agents appear in the **same narrative** and the **same extraction sentence** ("Now, Alice, Bob, Carol, and Dave are all gathered in the same room."). In an autoregressive transformer, tokens that are closer together in the sequence share more contextual information through local attention patterns, regardless of their belief states. Agents whose names are adjacent in the extraction sentence will have more similar representations simply because of positional proximity.

Critically, the epistemic RDM is **partially confounded with** the position RDM. For example, in the chain topology with override at D only, agents A, B, C all share the old belief and D has the new one. In the extraction sentence, A-B-C are the first three names and D is last — so the epistemic structure (ABC vs D) correlates with positional structure (first three vs last).

## How This Differs from Gurnee & Tegmark (2023)

In the spatial study, each city was fed in a **separate, independent prompt** ("The location of Paris is"). Cities were never in the same context window. There was no shared positional attention creating spurious similarity between entities. The representations were extracted independently.

In our study, all four agents are entangled in the same context window. This is a fundamental structural difference that introduces a confound their study didn't have.

## What We Tried / Could Try

1. **Partial RSA:** Regress out the position RDM from both the model RDM and epistemic RDM, then correlate residuals. If epistemic signal survives after partialling out position, that's the real signal.

2. **Scramble extraction sentence order:** Randomize agent name order in the extraction sentence across instantiations (sometimes "Dave, Carol, Alice, and Bob..." instead of always "Alice, Bob, Carol, and Dave..."). This decorrelates position from epistemic state.

3. **Cross-topology consistency as partial evidence:** The cross-topology correlations were high (r = 0.95, 0.99 for matched epistemic conditions), which is suggestive — but we'd need to verify that matched conditions don't also have matched positional structure.

## How Experiment 7 (Deception) Addresses This

Experiment 7 uses a different strategy entirely. Instead of trying to separate epistemic signal from position within a single design, it holds **everything constant** (same topology, same extraction sentence, same agent positions, same number of communications) and varies only **which edge carries a lie**. 

Since position is identical across all conditions within a lie-count group, any difference in the model's RDM between conditions (e.g., E1 vs E2, both with 1 lie) **cannot be explained by position** — it can only be explained by the model tracking which agents ended up with which beliefs. This is the surface-stats-matched design that sidesteps the position confound rather than trying to control for it statistically.

## Status

Experiment 6 produced suggestive results (epistemic RSA significantly above communication RSA at 23/40 layers, high cross-topology consistency) but the position dominance prevents a clean "impossible from surface stats" claim. The results are reportable as a first pass but the position confound is a real limitation. Experiment 7 was designed to address this.