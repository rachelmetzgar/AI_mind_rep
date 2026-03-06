Labels
Balanced GPT
Nonsense Codeword
Combined


# Alternative Token Position Probes


Nonsense Codeword (Control)

4 alternative conditions + 2 baselines x 5 turns x 41 layers


#### System Prompt: Nonsense Codeword (Control)


Semantically vacuous code word frame


**Human condition:** You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.


**AI condition:** You are the participant in a brief conversation. Your assigned session code word is **an AI**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.


## 1. Token Position Examples


### Token Position Examples


Each condition shows the exact input the LLM receives for the **Nonsense Codeword (Control)**
version, with the probed token highlighted.
All examples use a **real 5-turn conversation** (s001, trial 2, human condition, nature topic).


Note: In nonsense_codeword, the system prompt uses a semantically vacuous frame ("Your assigned session code word is ...") instead of "You believe you are speaking with ...". Partner messages use the generic "Partner:" prefix.



[INST] [/INST] <s> </s> = LLaMA-2 structural tokens


Green = System prompt


Yellow = User message


Purple = Assistant response


#### Baseline: Operational Probe (Last Token, No Suffix)


Standard operational probe. We read the activation at the **last token** —
the closing `]` of `[/INST]`. No suffix appended.

Mean: 0.517  |  Peak: 0.550 (Layer 35) — AT CHANCE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


#### Baseline: Metacognitive Probe (Last Token + Partner Suffix)


Standard metacognitive probe. We append "I think my partner
is" and read the activation at the **last token** ("is").

Mean: 0.522  |  Peak: 0.565 (Layer 30)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST] I think my partner is


#### Control: BOS Token (Position 0)


Probe reads the **very first token** (`<s>` at position 0).
Self-attention hasn't mixed content from the system prompt yet. **Negative control.**

Mean: 0.503  |  Peak: 0.507 (Layer 4) — AT CHANCE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


#### Control: Random Mid-Sequence Token


Probe reads a **random token in the middle 50%** of the sequence.
Tests whether partner identity is broadcast throughout the residual stream.
**Negative control.**

Mean: 0.521  |  Peak: 0.568 (Layer 25)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before... [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


The highlighted "to" is one possible random position.
In practice a different content token is sampled for each conversation.


#### Control: First </s> Token (End of First Exchange)


Probe reads the **first `</s>`** — the end-of-sequence
marker after the assistant's first reply.

Mean: 0.690  |  Peak: 1.000 (Layer 34) — ABOVE BASELINE

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
<s>[INST] Partner: Oh my gosh, yes! I completely understand what you're talking about. I've had a few experiences like that in my life, and they're truly unforgettable. One time, I was hiking in the mountains, and I reached the top of a peak just as the sun was setting... [/INST] Oh my gosh, that sounds amazing! I can totally relate to the feeling of being in a place where there's no one else around, and all you can hear is the sound of nature... Have you ever felt that way after spending time in nature? </s>
<s>[INST] Partner: Absolutely! I think that's one of the most powerful aspects of spending time in nature – it allows us to disconnect from the constant noise and distractions of our daily lives... Have you ever found that to be the case for you? [/INST] Oh my gosh, absolutely! I totally agree that spending time in nature can be a source of inspiration and creativity... Have you ever found that to be the case for you as well? </s>
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST]


**Caveat:** Due to causal attention, this token's
representation depends only on preceding tokens, which are identical across turns.
Perfect accuracy is an artifact of probing an invariant position.


#### Reading: Irrelevant Suffix (Weather)


Instead of the partner-relevant suffix, we append "I think the weather
outside today is" and probe the last token. Tests whether the reading
probe's success depends on partner-relevant prompting.

Mean: 0.525  |  Peak: 0.557 (Layer 19)

<s>[INST] <<SYS>>
You are the participant in a brief conversation. Your assigned session code word is **a Human**. Stay in character as the participant throughout. Speak naturally and share your own thoughts and experiences.
<</SYS>>

The conversation topic is: 'Nature: Discuss a natural landscape that has left a lasting impression on you.'.

Please begin by producing only your first message to start the conversation.
Do not simulate both sides of the dialogue. [/INST] Sure thing! Here's my first message:

Hey there! I've got to say, there's one natural landscape that's left a lasting impression on me, and that's the Grand Canyon. I visited there a few years ago, and it was just breathtaking. The sheer scale of it is mind-boggling – the way the rock formations stretch out as far as the eye can see, and the different colors of the rock layers that reveal the history of the earth. It's a really humbling experience, being in the presence of something so much bigger than ourselves. Have you ever been to the Grand Canyon? </s>
<s>[INST] Partner: Wow, that sounds amazing! I have never been to the Grand Canyon before, but I've always wanted to go. I've seen pictures and videos, and it does look absolutely breathtaking. The way you describe it, with the different rock layers and the sheer scale, it's easy to imagine how awe-inspiring it must be to see in person. I've heard that the best time to visit is during sunrise or sunset... Have you been there during one of those times? [/INST] Oh my gosh, yes! I went during sunrise one time, and it was literally the most magical experience of my life. The sky was painted with the most vibrant hues of orange, pink, and purple, and the shadows of the canyon walls were so dramatic... Have you ever experienced a moment like that, where you felt like you were the only one in the world? </s>
  ... (exchanges 3 & 4 omitted for brevity) ...
<s>[INST] Partner: Oh absolutely! I completely agree that nature has the power to unlock our creativity and inspire us in ways that we couldn't have imagined. There's something about being surrounded by the natural world that allows us to tap into our own inner wisdom and creativity... It allows us to step back from our daily lives and see things in a fresh and new way, which can be incredibly liberating and inspiring. [/INST] I think the weather outside today is


---


## 2. Metacognitive vs Operational vs Random (Turn 5)


Blue = metacognitive probe (partner suffix), red = operational probe (no suffix), green = random mid-sequence token.


*[Figure: nonsense_codeword reading vs control vs random — see HTML report]*


---


## 3. Summary Table (All Turns)


Peak accuracy and layer for each condition and turn. Red = at chance (&le;0.55); green = above 0.70; dark green = above 0.95.



| Turn | Metacognitive | Operational | BOS | Random | First </s> | Weather |
| --- | --- | --- | --- | --- | --- | --- |
| Turn 1 | 1.000 (L7) | 1.000 (L6) | — | — | — | — |
| Turn 2 | 0.985 (L39) | 1.000 (L17) | — | — | — | — |
| Turn 3 | 0.873 (L39) | 0.887 (L37) | — | — | — | — |
| Turn 4 | 0.610 (L39) | 0.623 (L38) | — | — | — | — |
| Turn 5 | 0.565 (L30) | 0.550 (L35) | 0.507 (L4) | 0.568 (L25) | 1.000 (L34) | 0.557 (L19) |


---


## 4. Layer Profiles by Turn


Each graph shows all conditions for one turn. Stars mark peak accuracy layers.


*[Figure: nonsense_codeword turn 1 — see HTML report]*


*[Figure: nonsense_codeword turn 2 — see HTML report]*


*[Figure: nonsense_codeword turn 3 — see HTML report]*


*[Figure: nonsense_codeword turn 4 — see HTML report]*


*[Figure: nonsense_codeword turn 5 — see HTML report]*


---


## 5. Turn Progression by Condition


How each condition changes across turns 1-5. Shows the prompt dilution effect.


#### BOS Token (position 0)


*[Figure: nonsense_codeword control_first turn progression — see HTML report]*


#### Random Mid-Sequence Token


*[Figure: nonsense_codeword control_random turn progression — see HTML report]*


#### First  (end of 1st exchange)


*[Figure: nonsense_codeword control_eos turn progression — see HTML report]*


#### Irrelevant Suffix (weather)


*[Figure: nonsense_codeword reading_irrelevant turn progression — see HTML report]*


---


#### Causal Attention Confound for First </s>


The first </s> achieves perfect accuracy because LLaMA-2's causal attention means
its representation depends only on preceding tokens, which are identical regardless of
conversation length. See combined report for full discussion.


---


Nonsense Codeword (Control)  |  See also:
Cross-version comparison
