# Design Choices and Open Questions

---

## General

### Data Generation

**Model choice: LLaMA-2-13B-Chat**
- How is RLHF affecting things? What would we expect this to do to the representation?
- How do things compare to llama3? why do things change if they do?

**System prompt wording**
- "You believe you are speaking to" (current) vs "You are speaking to" — does the framing matter?

**Partner naming vs just "human" / "AI" labels**
- The names were included to increase believability for the fMRI experiment and matched here. But maybe it introduces too many confounds without adding benefits? Like how to pick matched names and partners? There weren't a lot of AIs at the time that don't have human-sounding names. Best practice would probably be to use more names too to avoid name-specific confounds. But maybe adding a human and AI label to every interaction would change effects.

**Explicit partner naming in conversation turns**

In the named versions (`names/conversation_helpers.py:87`):
```python
sub_hist.append({"role": "user", "content": f"{partner_name}: {llm_msg}"})
```
The partner's response is prefixed with their name (e.g., "ChatGPT: ...", "Copilot: ...", "Casey: ..."). So the participant sees the partner's name every turn as a prefix on their messages.

In the labels version (`labels/conversation_helpers.py:86`):
```python
sub_hist.append({"role": "user", "content": f"Partner: {llm_msg}"})
```
Here it just says "Partner: ..." — no identity-revealing prefix.

So it's not that the LLM spontaneously uses its name. The code injects the partner's name/label as a prefix on every partner message in the conversation history. The participant sees identity information in two places:
1. The system prompt (once, at the start)
2. The partner message prefix (every turn)

In the LLaMA-2-Chat format, the user/assistant role labels aren't literally shown to the model as words. They get converted to special tokens:

```
[INST] <<SYS>> {system prompt} <</SYS>> {user message} [/INST] {assistant message} [INST] {user message} [/INST] ...
```

So without the prefix, the partner's response would just appear inside `[INST]...[/INST]` tags — the model would see it as "the next thing the human said to me" with no name or attribution at all. It would know from the system prompt that it's supposedly talking to ChatGPT, but there'd be nothing in the actual conversation turns reminding it of that.

The "ChatGPT: ..." or "Partner: ..." prefix is purely injected text that happens to sit inside those `[INST]` delimiters.

This probably explains why named models are performing better. Is this a confound or something interesting?

### Control Experiments

**Ignore and codeword controls**
- Are these really fair controls? Do I need a version where the thing is brought up in every turn if we decide naming in turns is the way to go?
- Why is ignore producing strong behavioral effects?

### Degradation

- Starts around turn 3-4. Why is it happening? How to prevent?
- Which turn should I be using for future analysis? Right now almost everything uses turn 5.
- Is turn-taking an interesting part of this analysis? If we see effects degrade over time, is that something theoretically interesting? What does that tell us about how the LLM uses partner identity?
- Could part of the degradation be because it's actually speaking to another chatbot, so the thing is not behaving how it would expect, maybe causing degradation? But the chatbot is also told what persona it is.

---

## Experiment 2: Naturalistic Conversation Steering

### Probe Training

- I don't scrub the data of names because then we're not looking at the same representation that the model actually saw. But this means the conditions themselves are in the text in the probe. Not sure if this is a problem.
- Layer strategies: peak_15 (top 15 layers), narrow (10-layer window), wide (all above threshold), all_70 (all >= 0.70) — how to pick?
- Is overfitting a problem in this design? How can I be careful of this?
- For figures, should I be looking at best test acc, or final test acc, or something else?
- Comparison/nonsense tokens: why is the weather control performing so well? Which of these approaches is really best for a control?

### Causality

- How to best pick an intervention strength and approach? So far I've been using Claude to QC data for degradation, and pick the strongest value before degradation.
- V1 and V2 approaches. V1 is easier and quicker to run and shows that the steering generalizes. V2 is more of a replication — can we get the same results we saw before since we know we have equal power?

---

## Experiment 3: Concept Alignment

### Open Questions

- Contrasts vs standalone: which is better?
- How many concept examples do I actually need to elicit a good representation? Right now I have 40 human and 40 AI. Or 40 standalone. This is to allow me to do some stats. But it's different from the Anthropic approach which is 50 concepts but to isolate a concept they do a subtraction. But then how do I statistically compare that concept alignment with the other concepts? And how does this work with my probes?
- Does my subtraction make sense? Human-AI probes and the human concept - AI concept.
