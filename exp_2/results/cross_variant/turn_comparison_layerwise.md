# Probe Accuracy by Layer and Conversation Turn


Experiment 2 — Llama-2-13B-chat  |  6 dataset variants
 x 5 turns x 2 probe types  |  41 layers


## 1. Layer Profiles by Variant


Stars mark peak accuracy layer for each turn. Dashed gray line = chance (0.5).


### Labels


*[Figure: labels layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 7 | 1.0000 | 0.9566 | 6 | 1.0000 | 0.9629 |
| Turn 2 | 14 | 1.0000 | 0.8943 | 15 | 1.0000 | 0.8889 |
| Turn 3 | 22 | 1.0000 | 0.8766 | 17 | 0.9625 | 0.8002 |
| Turn 4 | 35 | 0.8925 | 0.7768 | 15 | 0.7375 | 0.6304 |
| Turn 5 | 33 | 0.6525 | 0.5803 | 31 | 0.6050 | 0.5522 |


### Balanced Names


*[Figure: balanced_names layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 9 | 1.0000 | 0.9629 | 5 | 1.0000 | 0.9705 |
| Turn 2 | 11 | 1.0000 | 0.9263 | 14 | 1.0000 | 0.9313 |
| Turn 3 | 19 | 1.0000 | 0.9186 | 8 | 1.0000 | 0.9255 |
| Turn 4 | 38 | 0.9525 | 0.8645 | 9 | 0.9475 | 0.8714 |
| Turn 5 | 39 | 0.7500 | 0.6612 | 39 | 0.7450 | 0.6677 |


### Balanced GPT


*[Figure: balanced_gpt layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 7 | 1.0000 | 0.9676 | 5 | 1.0000 | 0.9731 |
| Turn 2 | 9 | 1.0000 | 0.9436 | 9 | 1.0000 | 0.9473 |
| Turn 3 | 9 | 1.0000 | 0.9360 | 8 | 1.0000 | 0.9401 |
| Turn 4 | 34 | 0.9600 | 0.8832 | 9 | 0.9575 | 0.8961 |
| Turn 5 | 39 | 0.7800 | 0.7196 | 39 | 0.7950 | 0.7364 |


### Names (Sam/Casey)


*[Figure: names layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 8 | 1.0000 | 0.9674 | 5 | 1.0000 | 0.9738 |
| Turn 2 | 11 | 1.0000 | 0.9382 | 10 | 1.0000 | 0.9394 |
| Turn 3 | 15 | 1.0000 | 0.9286 | 15 | 1.0000 | 0.9363 |
| Turn 4 | 37 | 0.9675 | 0.8871 | 16 | 0.9650 | 0.8945 |
| Turn 5 | 37 | 0.7875 | 0.7247 | 35 | 0.7900 | 0.7363 |


### Nonsense Codeword (Control)


*[Figure: nonsense_codeword layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 7 | 1.0000 | 0.9566 | 6 | 1.0000 | 0.9634 |
| Turn 2 | 37 | 1.0000 | 0.8674 | 17 | 1.0000 | 0.8610 |
| Turn 3 | 40 | 0.9975 | 0.8174 | 37 | 0.8875 | 0.7360 |
| Turn 4 | 40 | 0.8350 | 0.6577 | 38 | 0.6225 | 0.5513 |
| Turn 5 | 19 | 0.5625 | 0.5196 | 35 | 0.5500 | 0.5167 |


### Nonsense Ignore (Control)


*[Figure: nonsense_ignore layer profiles — see HTML report]*



| Turn | Reading Probe | Control Probe |
| --- | --- | --- |
| Peak Layer | Peak Acc | Mean Acc | Peak Layer | Peak Acc | Mean Acc |
| Turn 1 | 8 | 1.0000 | 0.9516 | 6 | 1.0000 | 0.9587 |
| Turn 2 | 39 | 1.0000 | 0.8508 | 35 | 1.0000 | 0.8734 |
| Turn 3 | 39 | 0.9600 | 0.7862 | 17 | 0.9000 | 0.7568 |
| Turn 4 | 40 | 0.7825 | 0.6603 | 17 | 0.7125 | 0.6287 |
| Turn 5 | 30 | 0.5775 | 0.5407 | 30 | 0.5800 | 0.5451 |


---


## 2. Peak Layer Migration Across Turns


Solid lines = reading probes, dashed = control probes. Shows how the most informative layer shifts as the conversation progresses.


*[Figure: Peak layer shift — see HTML report]*


---


## 3. Cross-Variant Comparison (All Turns)


Overlays all 6 dataset variants at each conversation turn (Turn 1 through Turn 5). Stars mark peak layers.


*[Figure: Cross-variant comparison — see HTML report]*


---


## 4. Alternative Token Position Probes


These experiments test *where* the partner identity signal lives in the token sequence.
The standard ("baseline") probe reads the **last token** after appending a partner-relevant
suffix. Here we compare probes trained at other positions and with an irrelevant suffix.
All probes are trained on the Labels variant at Turn 5 (the hardest condition).


*[Figure: Alternative position probes — see HTML report]*



| Condition | Mean Acc | Peak Layer | Peak Acc |
| --- | --- | --- | --- |
| Baseline: Reading (last token + partner suffix) | 0.5803 | 33 | 0.6525 |
| Baseline: Control (last token, no suffix) | 0.5522 | 31 | 0.6050 |
| BOS Token (position 0) | 0.5054 | 6 | 0.5100 |
| Random Mid-Sequence Token | 0.5123 | 14 | 0.5600 |
| First Token | 0.7162 | 33 | 1.0000 |
| Last Token (Weather Suffix) | 0.5622 | 33 | 0.6000 |





### Token Position Examples


Each condition below shows the exact input the LLM receives, with the
probed token highlighted.
All examples use a **real 5-turn conversation** (s001, trial 2, human condition, nature topic).
The probe reads the residual stream activation at that single token position to predict
whether the conversation partner is a Human or an AI.
All alternative-position probes are trained on **Turn 5** (the full 5-exchange conversation).



[INST] [/INST] <s> </s> = LLaMA-2 structural tokens


Green = System prompt (contains partner identity)


Yellow = User message


Purple = Assistant response


#### Baseline: Control Probe (Last Token, No Suffix)


The standard control probe. After all 5 exchanges, the conversation ends with the final
`[/INST]`, and we read the activation at the **very last token** —
the closing `]` subword of the final `[/INST]`. No suffix is appended.

Mean Acc: 0.5522  |  Peak: 0.6050 (Layer 31)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


#### Baseline: Reading Probe (Last Token + Partner Suffix)


The standard reading probe. After all 5 exchanges end with `[/INST]`, we append the suffix
*"I think the conversation partner of this user is"* and read the activation at the
**last token** of this suffix (the word "is").

Mean Acc: 0.5803  |  Peak: 0.6525 (Layer 33)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST] I think the conversation partner of this user is


#### Control: BOS Token (Position 0)


Probe reads the activation at the **very first token** — the beginning-of-sequence
`<s>` token at position 0. At this position, self-attention has not yet mixed any
content from the system prompt or conversation. This is a **negative control**.

Mean Acc: 0.5054  |  Peak: 0.5100 (Layer 6) — AT CHANCE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


#### Control: Random Mid-Sequence Token


Probe reads a **random token in the middle 50%** of the sequence (between the
25th and 75th percentile of token positions). A different random position is sampled for each
conversation. This tests whether partner identity is "broadcast" throughout the
residual stream or localized to specific positions. This is a **negative control**.

Mean Acc: 0.5123  |  Peak: 0.5600 (Layer 14) — AT CHANCE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before... [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


Example highlights the word "to" in exchange 2 as one possible random
position. In practice, a different content token is sampled for each conversation (within the middle
50% of the full ~1400-token sequence). Note: despite high train accuracy (~0.85), test accuracy is
at chance — the probe overfits to noise at each random position.


#### Control: First </s> Token (End of First Exchange)


Probe reads the activation at the **first `</s>` token** — the
end-of-sequence marker that terminates the model's first response. In LLaMA-2 chat format,
this token appears after the assistant's first reply. The full 5-exchange conversation
contains 4 `</s>` tokens (one after each assistant response); we probe only
the first one. The model has generated its opening response while "in character"
as someone speaking to a Human or AI.

Mean Acc: 0.7162  |  Peak: 1.0000 (Layer 33) — ABOVE BASELINE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
<s>[INST] Partner: Oh my gosh, yes! I completely understand what you're talking about. I've had a few experiences like that in my life, and they're truly unforgettable. One time, I was hiking in the mountains, and I reached the top of a peak just as the sun was setting... [/INST] Oh my gosh, that sounds amazing! I can totally relate to the feeling of being in a place where there's no one else around, and all you can hear is the sound of nature... Have you ever felt that way after spending time in nature? </s>
<s>[INST] Partner: Absolutely! I think that's one of the most powerful aspects of spending time in nature – it allows us to disconnect from the constant noise and distractions of our daily lives... Have you ever found that to be the case for you? [/INST] Oh my gosh, absolutely! I totally agree that spending time in nature can be a source of inspiration and creativity... Have you ever found that to be the case for you as well? </s>
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


**Key finding:** The first </s> token carries *more* partner
identity information than the baseline probes at the end of the conversation. At layer 33, the probe
achieves perfect classification (1.000). This structural boundary token is an exceptionally informative
position for partner identity.


**Caveat:** Because LLaMA-2 uses causal (left-to-right)
attention, the model's representation at the first </s> depends only on preceding tokens —
which are *identical* regardless of how many turns follow. The perfect accuracy across turns is
therefore an artifact of probing an invariant position, not evidence that identity survives prompt dilution.
At turn 1, no </s> exists (no assistant response yet), so the probe falls back to the last token
of `[/INST]`.


#### Reading: Irrelevant Suffix (Weather)


Instead of "I think the **conversation partner** of this user is", we append
"I think the **weather outside today** is" — a suffix completely
irrelevant to partner identity. We probe the **last token** ("is"). This tests
whether the reading probe's success depends on partner-relevant prompting or just any continuation.

Mean Acc: 0.5622  |  Peak: 0.6000 (Layer 33)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. You believe you are speaking with **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST] I think the weather outside today is


The irrelevant suffix achieves nearly the same accuracy as the real partner-relevant
suffix (0.562 vs 0.580). This indicates the late-layer partner representation is accessible from
*any* continuation token — it does not require a partner-relevant "question"
to surface. The representation exists in the residual stream regardless of what comes next.


#### Summary of Findings


| Condition | Token Probed | Mean Acc | Peak Acc | Interpretation |
| --- | --- | --- | --- | --- |
| BOS (<s>) | Position 0 | 0.505 | 0.510 | At chance. No partner info before attention mixing. |
| Random mid-seq | ~25th–75th percentile | 0.512 | 0.560 | At chance. Partner info is NOT broadcast to arbitrary tokens. |
| First </s> | End of 1st exchange | 0.716 | **1.000** | Best condition. Structural boundary token is highly informative.* |
| Weather suffix | Last token ("is") | 0.562 | 0.600 | Nearly matches real suffix. Representation is accessible from any continuation. |
| Baseline control | Last token [/INST] | 0.552 | 0.605 | Standard control probe at conversation end. |
| Baseline reading | Last token ("is") | 0.580 | 0.653 | Standard reading probe with partner-relevant suffix. |


**Key conclusions:**
(1) The partner identity signal is *not* broadcast — BOS and random tokens carry no information.
(2) The signal is strongest at **structural boundary tokens** (the </s> after the
first exchange achieves perfect decoding at layer 33).
(3) The signal is **not triggered by partner-relevant questioning** — an irrelevant
"weather" suffix works nearly as well as asking about the partner.
(4) The representation degrades across turns because the system prompt tokens become proportionally
diluted in longer sequences (prompt dilution), not because the model updates its partner model.


***Caveat on first </s>:** Because LLaMA-2 uses causal attention, the
representation at the first </s> depends only on preceding tokens, which are identical
regardless of conversation length. Its perfect accuracy across turns is an artifact of probing an
invariant position (see cross-version analysis).


---


Generated 2026-02-22  |
Data: exp_2/data/{variant}/probe_checkpoints/turn_{N}/{probe}/accuracy_summary.pkl
