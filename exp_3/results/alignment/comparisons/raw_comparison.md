# Exp 3: Raw Alignment Comparison Across Data Versions

*Generated: 2026-03-02 14:48*

## Summary

| Version | Reading R² (Mental, ×10⁻³) | Control R² (Mental, ×10⁻³) | Description |
|---------|---------------------------|---------------------------|-------------|
| Labels | 0.251 | 0.325 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.454 | 1.415 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.631 | 2.810 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.435 | 1.263 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.190 | 0.313 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.177 | 0.411 | Nonsense labels with instruction to ignore them |
| Labels + Turnwise | 0.533 | 1.665 | Labels + turn-level 'Human:'/'AI:' prefix each turn |
| You-Are Labels | 0.243 | 0.379 | 'You are talking to a Human/an AI' framing |
| You-Are Bal. GPT | 0.456 | 2.271 | 'You are talking to' + named partners (Gregory/Rebecca, ChatGPT/GPT-4) |
| You-Are Lab. Turn. | 0.570 | 1.038 | 'You are talking to' framing + turn-level prefix |

## Per-Dimension Data (Reading Probe R² ×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|-----------|----------|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.229 | 0.219 | 0.400 | 0.264 | 0.201 | 0.202 | 0.598 | 0.210 | 0.308 | 0.592 |
| Emotions | Mental | 0.210 | 0.219 | 0.422 | 0.305 | 0.167 | 0.137 | 0.306 | 0.191 | 0.256 | 0.381 |
| Agency | Mental | 0.189 | 0.470 | 0.558 | 0.347 | 0.143 | 0.126 | 0.536 | 0.223 | 0.332 | 0.550 |
| Intentions | Mental | 0.255 | 0.571 | 0.673 | 0.613 | 0.172 | 0.158 | 0.612 | 0.305 | 0.429 | 0.596 |
| Prediction | Mental | 0.197 | 0.577 | 0.671 | 0.501 | 0.189 | 0.187 | 0.474 | 0.210 | 0.384 | 0.643 |
| Cognitive | Mental | 0.274 | 0.438 | 0.693 | 0.484 | 0.251 | 0.207 | 0.578 | 0.297 | 0.627 | 0.637 |
| Social | Mental | 0.350 | 0.644 | 0.773 | 0.493 | 0.164 | 0.178 | 0.437 | 0.199 | 0.586 | 0.394 |
| Attention | Mental | 0.305 | 0.494 | 0.862 | 0.469 | 0.230 | 0.222 | 0.723 | 0.305 | 0.722 | 0.772 |
| Embodiment | Physical | 0.216 | 0.183 | 0.404 | 0.338 | 0.207 | 0.205 | 0.407 | 0.261 | 0.212 | 0.547 |
| Roles | Physical | 0.261 | 0.393 | 0.611 | 0.410 | 0.173 | 0.189 | 0.453 | 0.259 | 0.363 | 0.416 |
| Animacy | Physical | 0.197 | 0.202 | 0.398 | 0.208 | 0.254 | 0.274 | 0.292 | 0.271 | 0.218 | 0.391 |
| Formality | Pragmatic | 0.314 | 0.254 | 0.244 | 0.305 | 0.245 | 0.316 | 0.274 | 0.212 | 0.354 | 0.444 |
| Expertise | Pragmatic | 0.349 | 0.614 | 0.372 | 0.529 | 0.272 | 0.212 | 0.409 | 0.394 | 0.493 | 0.467 |
| Helpfulness | Pragmatic | 0.236 | 0.559 | 0.220 | 0.593 | 0.290 | 0.216 | 0.394 | 0.374 | 0.732 | 0.301 |
| Baseline | Control | 0.216 | 0.331 | 0.596 | 0.250 | 0.187 | 0.187 | 0.622 | 0.225 | 0.527 | 0.579 |
| Biological | Control | 0.182 | 0.139 | 0.293 | 0.177 | 0.207 | 0.219 | 0.372 | 0.207 | 0.315 | 0.468 |
| Shapes | Control | 0.266 | 0.198 | 0.234 | 0.182 | 0.215 | 0.275 | 0.142 | 0.159 | 0.502 | 0.310 |
| SysPrompt | SysPrompt | 0.237 | 0.674 | 0.936 | 0.385 | 0.183 | 0.221 | 0.430 | 0.187 | 0.542 | 0.280 |

## Per-Dimension Data (Control Probe R² ×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|-----------|----------|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.263 | 0.962 | 2.105 | 0.996 | 0.258 | 0.441 | 1.812 | 0.280 | 1.874 | 1.173 |
| Emotions | Mental | 0.355 | 0.930 | 1.988 | 1.093 | 0.333 | 0.376 | 1.157 | 0.369 | 1.678 | 0.582 |
| Agency | Mental | 0.355 | 1.386 | 2.565 | 1.142 | 0.377 | 0.449 | 1.710 | 0.447 | 2.221 | 1.107 |
| Intentions | Mental | 0.408 | 1.503 | 2.820 | 1.540 | 0.378 | 0.476 | 1.650 | 0.530 | 2.077 | 0.982 |
| Prediction | Mental | 0.302 | 1.751 | 2.982 | 1.324 | 0.312 | 0.446 | 1.754 | 0.369 | 2.293 | 1.026 |
| Cognitive | Mental | 0.265 | 1.350 | 3.138 | 1.260 | 0.276 | 0.335 | 1.483 | 0.354 | 2.477 | 0.961 |
| Social | Mental | 0.341 | 1.737 | 3.341 | 1.219 | 0.292 | 0.314 | 1.954 | 0.305 | 2.560 | 1.276 |
| Attention | Mental | 0.314 | 1.704 | 3.541 | 1.532 | 0.275 | 0.451 | 1.796 | 0.381 | 2.992 | 1.192 |
| Embodiment | Physical | 0.255 | 0.810 | 1.757 | 0.865 | 0.269 | 0.471 | 1.609 | 0.299 | 1.284 | 0.838 |
| Roles | Physical | 0.274 | 1.235 | 2.504 | 1.071 | 0.253 | 0.387 | 1.746 | 0.341 | 2.187 | 0.905 |
| Animacy | Physical | 0.121 | 0.626 | 1.386 | 0.576 | 0.179 | 0.354 | 1.438 | 0.194 | 1.328 | 0.691 |
| Formality | Pragmatic | 0.298 | 0.248 | 0.353 | 0.308 | 0.232 | 0.301 | 0.256 | 0.374 | 0.361 | 0.335 |
| Expertise | Pragmatic | 0.289 | 0.874 | 1.100 | 0.784 | 0.214 | 0.213 | 0.306 | 0.347 | 0.772 | 0.333 |
| Helpfulness | Pragmatic | 0.253 | 0.732 | 0.269 | 0.636 | 0.203 | 0.174 | 0.247 | 0.302 | 0.559 | 0.240 |
| Baseline | Control | 0.165 | 1.538 | 2.920 | 0.792 | 0.143 | 0.293 | 2.499 | 0.147 | 2.745 | 1.542 |
| Biological | Control | 0.191 | 0.393 | 1.032 | 0.727 | 0.217 | 0.398 | 1.069 | 0.256 | 1.117 | 0.553 |
| Shapes | Control | 0.264 | 0.259 | 0.248 | 0.177 | 0.191 | 0.208 | 0.159 | 0.243 | 0.611 | 0.275 |
| SysPrompt | SysPrompt | 0.189 | 2.751 | 2.959 | 1.683 | 0.077 | 0.200 | 1.460 | 0.087 | 2.208 | 0.646 |

## Key Findings

1. **Alignment scales with identity cue specificity**: balanced_gpt > names > balanced_names >> labels ≈ nonsense_codeword ≈ nonsense_ignore
2. **Control probes show higher alignment** than reading probes for name-based versions
3. **Nonsense and label versions show near-floor alignment**, comparable to the shapes negative control
4. **This is raw alignment** — residual analysis (projecting out entity baseline) needed to assess concept-specific contribution

## Methods

- **Metric**: Mean R² (cosine similarity squared) between concept direction vectors and probe weight vectors, averaged across 41 layers
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Model**: LLaMA-2-13B-Chat
- **Concept vectors**: 19 dimensions, ~80 prompts each (contrasts mode: human vs AI)
- **Probes**: From Exp 2, turn 5, reading and control probes (logistic, per-layer)
