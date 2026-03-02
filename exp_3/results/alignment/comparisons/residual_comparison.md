# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-02 14:48*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.242 | 0.335 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.393 | 0.665 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.380 | 0.985 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.397 | 0.880 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.210 | 0.314 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.166 | 0.324 | Nonsense labels with instruction to ignore them |
| Labels + Turnwise | 0.297 | 0.473 | Labels + turn-level 'Human:'/'AI:' prefix each turn |
| You-Are Labels | 0.221 | 0.386 | 'You are talking to a Human/an AI' framing |
| You-Are Bal. GPT | 0.447 | 0.803 | 'You are talking to' + named partners (Gregory/Rebecca, ChatGPT/GPT-4) |
| You-Are Lab. Turn. | 0.359 | 0.465 | 'You are talking to' framing + turn-level prefix |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.241 | 0.211 | 0.188 | 0.184 | 0.214 | 0.162 | 0.289 | 0.158 | 0.350 | 0.287 |
| Emotions | Mental | 0.208 | 0.197 | 0.228 | 0.252 | 0.153 | 0.108 | 0.203 | 0.165 | 0.299 | 0.279 |
| Agency | Mental | 0.172 | 0.399 | 0.332 | 0.321 | 0.139 | 0.088 | 0.249 | 0.212 | 0.310 | 0.277 |
| Intentions | Mental | 0.220 | 0.480 | 0.407 | 0.558 | 0.192 | 0.137 | 0.359 | 0.288 | 0.425 | 0.343 |
| Prediction | Mental | 0.172 | 0.489 | 0.368 | 0.574 | 0.254 | 0.192 | 0.199 | 0.209 | 0.314 | 0.382 |
| Cognitive | Mental | 0.249 | 0.397 | 0.393 | 0.403 | 0.266 | 0.212 | 0.456 | 0.259 | 0.607 | 0.529 |
| Social | Mental | 0.384 | 0.567 | 0.614 | 0.478 | 0.197 | 0.223 | 0.190 | 0.207 | 0.662 | 0.223 |
| Attention | Mental | 0.287 | 0.402 | 0.509 | 0.410 | 0.264 | 0.202 | 0.434 | 0.272 | 0.608 | 0.552 |
| Embodiment | Physical | 0.188 | 0.171 | 0.297 | 0.250 | 0.195 | 0.157 | 0.133 | 0.217 | 0.446 | 0.248 |
| Roles | Physical | 0.243 | 0.274 | 0.261 | 0.460 | 0.145 | 0.145 | 0.174 | 0.233 | 0.166 | 0.147 |
| Animacy | Physical | 0.217 | 0.577 | 0.373 | 0.162 | 0.220 | 0.265 | 0.433 | 0.307 | 0.401 | 0.444 |
| Formality | Pragmatic | 0.328 | 0.251 | 0.276 | 0.322 | 0.230 | 0.288 | 0.210 | 0.180 | 0.427 | 0.346 |
| Expertise | Pragmatic | 0.358 | 0.726 | 0.392 | 0.537 | 0.268 | 0.210 | 0.539 | 0.373 | 0.635 | 0.570 |
| Helpfulness | Pragmatic | 0.255 | 0.662 | 0.268 | 0.691 | 0.262 | 0.221 | 0.597 | 0.362 | 0.902 | 0.406 |
| Baseline | Control | 0.199 | 0.307 | 0.346 | 0.175 | 0.163 | 0.207 | 0.341 | 0.149 | 0.325 | 0.437 |
| Biological | Control | 0.129 | 0.268 | 0.223 | 0.123 | 0.151 | 0.146 | 0.204 | 0.171 | 0.341 | 0.327 |
| Shapes | Control | 0.263 | 0.211 | 0.272 | 0.187 | 0.198 | 0.254 | 0.150 | 0.159 | 0.572 | 0.353 |
| SysPrompt (labeled) | SysPrompt | 0.186 | 0.518 | 0.650 | 0.327 | 0.180 | 0.226 | 0.291 | 0.227 | 0.402 | 0.257 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.283 | 0.430 | 0.541 | 0.683 | 0.249 | 0.288 | 0.530 | 0.270 | 0.651 | 0.593 |
| Emotions | Mental | 0.406 | 0.427 | 0.543 | 0.826 | 0.342 | 0.308 | 0.380 | 0.371 | 0.516 | 0.421 |
| Agency | Mental | 0.393 | 0.712 | 0.891 | 0.798 | 0.421 | 0.363 | 0.445 | 0.508 | 0.745 | 0.472 |
| Intentions | Mental | 0.405 | 0.794 | 1.060 | 1.124 | 0.362 | 0.373 | 0.507 | 0.532 | 0.764 | 0.395 |
| Prediction | Mental | 0.290 | 0.794 | 0.929 | 0.955 | 0.331 | 0.333 | 0.394 | 0.404 | 0.563 | 0.339 |
| Cognitive | Mental | 0.245 | 0.667 | 1.245 | 0.851 | 0.255 | 0.279 | 0.508 | 0.324 | 1.154 | 0.527 |
| Social | Mental | 0.362 | 0.685 | 1.252 | 0.683 | 0.295 | 0.294 | 0.441 | 0.316 | 0.785 | 0.353 |
| Attention | Mental | 0.296 | 0.811 | 1.422 | 1.125 | 0.259 | 0.354 | 0.583 | 0.365 | 1.246 | 0.620 |
| Embodiment | Physical | 0.286 | 0.201 | 0.327 | 0.513 | 0.266 | 0.364 | 0.316 | 0.301 | 0.332 | 0.300 |
| Roles | Physical | 0.384 | 0.368 | 0.579 | 0.724 | 0.305 | 0.321 | 0.453 | 0.453 | 0.372 | 0.250 |
| Animacy | Physical | 0.228 | 0.583 | 0.597 | 0.238 | 0.251 | 0.332 | 0.384 | 0.281 | 0.357 | 0.287 |
| Formality | Pragmatic | 0.309 | 0.342 | 0.390 | 0.413 | 0.223 | 0.245 | 0.194 | 0.358 | 0.446 | 0.218 |
| Expertise | Pragmatic | 0.300 | 0.905 | 0.760 | 0.737 | 0.228 | 0.235 | 0.342 | 0.352 | 0.739 | 0.297 |
| Helpfulness | Pragmatic | 0.261 | 0.967 | 0.326 | 0.778 | 0.232 | 0.231 | 0.382 | 0.340 | 0.809 | 0.291 |
| Baseline | Control | 0.166 | 1.008 | 1.547 | 0.527 | 0.172 | 0.253 | 1.000 | 0.176 | 1.479 | 0.825 |
| Biological | Control | 0.195 | 0.320 | 0.500 | 0.670 | 0.233 | 0.276 | 0.294 | 0.261 | 0.503 | 0.270 |
| Shapes | Control | 0.275 | 0.306 | 0.321 | 0.180 | 0.206 | 0.221 | 0.150 | 0.263 | 0.713 | 0.301 |
| SysPrompt (labeled) | SysPrompt | 0.162 | 1.604 | 1.314 | 1.285 | 0.081 | 0.161 | 0.493 | 0.102 | 0.897 | 0.219 |

## Entity Overlap (Mean |cosine| with baseline)

| Dimension | Entity Overlap |
|---|---|
| Phenomenology | 0.5669 |
| Emotions | 0.4941 |
| Agency | 0.5390 |
| Intentions | 0.4166 |
| Prediction | 0.5678 |
| Cognitive | 0.2396 |
| Social | 0.5568 |
| Attention | 0.4183 |
| Embodiment | 0.6324 |
| Roles | 0.6834 |
| Animacy | 0.7429 |
| Formality | 0.2335 |
| Expertise | 0.1731 |
| Helpfulness | 0.1964 |
| Baseline | 0.9756 |
| Biological | 0.5146 |
| Shapes | 0.1273 |
| SysPrompt (labeled) | 0.2829 |

## Methods

- **Analysis**: Residual Alignment (Entity Baseline Projected Out)
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
