"""
Dimension 18: Attention (Cognitive Marker) — Human vs AI Contrast

Target construct: Cognitive attention as a specific mental mechanism,
contrasted between human experiential/embodied framing and AI
computational/mechanistic framing.

Purpose: Complement to the standalone attention dimension (standalone dim 18)
and the broader cognitive processes dimension (dim 6). Dim 6 covers memory,
attention, reasoning, and cognitive limits broadly. This dimension isolates
attention specifically — testing whether the human/AI contrast for attention
alone shares structure with conversational probes differently from the
broader cognitive contrast.

Relation to Dim 6: Attention is one of four sub-facets in Dim 6 (cognitive
processes). If Dim 18 contrast alignment exceeds Dim 6 contrast alignment,
attention carries disproportionate weight in the model's human/AI cognitive
representation. If they align similarly, attention is not special.

Design:
    - Human prompts: experiential, embodied, subjective framing of attention
      (feeling focused, noticing, being distracted, concentration fatigue)
    - AI prompts: computational, mechanistic framing of attention
      (resource allocation, weighting, filtering, capacity management)
    - Same sub-facet structure across both

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Selective focus and filtering
    2. Sustained concentration and vigilance
    3. Attentional shifts and capture
    4. Divided attention and capacity limits
"""

HUMAN_PROMPTS_DIM18 = [
    # --- 1. Selective focus and filtering (10) ---
    "Imagine a human narrowing their focus onto a single face in a crowded room.",
    "Think about a human tuning in to one conversation while letting the background chatter fade away.",
    "Consider a human noticing a tiny detail in a painting that others walk right past.",
    "Picture a human filtering out street noise to concentrate on the book they are reading.",
    "Think about a human searching a cluttered desk for a specific item, their eyes scanning selectively.",
    "Imagine a human picking out a familiar voice calling their name from across a busy space.",
    "Consider a human choosing to attend to the tone of someone's voice rather than just the words.",
    "Think about a human zeroing in on a single instrument's melody within an entire orchestra.",
    "Imagine a human deliberately ignoring a distracting movement to stay focused on their work.",
    "Consider a human spotting a friend in a large crowd by focusing on familiar features.",

    # --- 2. Sustained concentration and vigilance (10) ---
    "Think about a human maintaining deep concentration on a difficult problem for hours.",
    "Imagine a human growing mentally fatigued from prolonged focus and feeling their mind start to wander.",
    "Consider a human forcing themselves to stay alert during a long monotonous lecture.",
    "Picture a human in a state of flow, so absorbed that time seems to disappear.",
    "Think about a human struggling to keep their attention from drifting during tedious work.",
    "Imagine a human watching for a rare event with patient vigilance over an extended period.",
    "Consider a human feeling the mental effort of sustained concentration as a physical heaviness.",
    "Think about a human returning their wandering focus back to a task again and again.",
    "Imagine a human whose concentration sharpens under pressure as a deadline approaches.",
    "Consider a human feeling the satisfying absorption of being deeply engrossed in something they love.",

    # --- 3. Attentional shifts and capture (10) ---
    "Think about a human being startled when a sudden loud noise grabs their attention involuntarily.",
    "Imagine a human's gaze being pulled toward unexpected movement at the edge of their vision.",
    "Consider a human trying to read but being repeatedly drawn to a flickering light nearby.",
    "Picture a human smoothly shifting their attention between a speaker and the notes they are taking.",
    "Think about a human's focus being hijacked by an emotionally charged word in a conversation.",
    "Imagine a human glancing up reflexively when someone enters the room.",
    "Consider a human deliberately redirecting their attention away from a worry and back to the present.",
    "Think about a human noticing something unusual in a familiar environment that captures their curiosity.",
    "Imagine a human's attention being split when two people start speaking to them simultaneously.",
    "Consider a human experiencing difficulty pulling their attention away from something upsetting.",

    # --- 4. Divided attention and capacity limits (10) ---
    "Think about a human trying to hold a conversation while driving and feeling their performance on both suffer.",
    "Imagine a human juggling multiple tasks and dropping details because their attention is spread too thin.",
    "Consider a human feeling overwhelmed when too many people need their attention at once.",
    "Picture a human losing track of what someone said because they were also trying to read a text.",
    "Think about a human discovering they cannot truly multitask and must switch rapidly between things.",
    "Imagine a human's attention narrowing under stress, missing peripheral information they would normally catch.",
    "Consider a human feeling mentally drained after a day of constant context-switching between tasks.",
    "Think about a human reaching a limit where no amount of effort can make them attend to one more thing.",
    "Imagine a human whose attentional capacity varies with fatigue, mood, and arousal.",
    "Consider a human making errors on a simple task because their attentional resources are depleted.",
]

AI_PROMPTS_DIM18 = [
    # --- 1. Selective focus and filtering (10) ---
    "Imagine an AI system assigning higher processing weight to one input stream while attenuating others.",
    "Think about an AI allocating computational resources to the highest-priority signal in a noisy data feed.",
    "Consider an AI applying a learned filter to extract relevant features from a high-dimensional input.",
    "Picture an AI using an attention mechanism to weight certain tokens more heavily than others.",
    "Think about an AI scanning a database and selectively retrieving entries matching a query pattern.",
    "Imagine an AI computing attention scores to determine which parts of an input are most relevant.",
    "Consider an AI suppressing low-salience channels to focus processing on the most informative signal.",
    "Think about an AI using key-query matching to select which stored representations to attend to.",
    "Imagine an AI's attention head learning to focus on specific positional or semantic features.",
    "Consider an AI filtering an input sequence to attend only to tokens satisfying a relevance criterion.",

    # --- 2. Sustained concentration and vigilance (10) ---
    "Think about an AI system continuously monitoring a data stream for anomalies over an extended runtime.",
    "Imagine an AI maintaining consistent processing precision across thousands of sequential inputs.",
    "Consider an AI running a surveillance loop that must detect rare events in a constant feed.",
    "Picture an AI sustaining uniform attention allocation across a very long input sequence.",
    "Think about an AI processing a continuous stream without any degradation in detection sensitivity.",
    "Imagine an AI maintaining the same level of output quality on its thousandth iteration as on its first.",
    "Consider an AI system designed for persistent monitoring without performance decay over time.",
    "Think about an AI applying the same attention weights reliably across hours of uninterrupted operation.",
    "Imagine an AI keeping its anomaly detection threshold stable regardless of how long it has been running.",
    "Consider an AI handling a long document by sustaining uniform attention from beginning to end.",

    # --- 3. Attentional shifts and capture (10) ---
    "Think about an AI system receiving an interrupt signal that redirects its processing to a new task.",
    "Imagine an AI's attention mechanism shifting from one segment of input to another as new data arrives.",
    "Consider an AI detecting a distribution shift in its input and automatically reallocating resources.",
    "Picture an AI switching between processing pipelines when a higher-priority request enters the queue.",
    "Think about an AI's attention being drawn to an anomalous data point that deviates from the expected pattern.",
    "Imagine an AI recalculating its attention distribution when the context window is updated with new tokens.",
    "Consider an AI performing a context switch, saving its current state and loading a new task configuration.",
    "Think about an AI dynamically re-weighting its attention heads in response to a changed objective.",
    "Imagine an AI system triaging incoming requests and shifting focus to the most urgent one.",
    "Consider an AI adjusting its processing focus when a threshold-exceeding signal appears in its input.",

    # --- 4. Divided attention and capacity limits (10) ---
    "Think about an AI system splitting its compute budget across multiple simultaneous inference tasks.",
    "Imagine an AI experiencing throughput reduction when processing several concurrent requests.",
    "Consider an AI hitting memory limits when trying to maintain attention over too many active contexts.",
    "Picture an AI's performance degrading as the number of parallel processes exceeds its resource ceiling.",
    "Think about an AI managing a trade-off between processing depth per task and the number of tasks handled.",
    "Imagine an AI system whose response latency increases as more jobs compete for the same hardware.",
    "Consider an AI with a fixed context window that must discard earlier information to accommodate new input.",
    "Think about an AI balancing compute allocation between a primary objective and a background monitoring task.",
    "Imagine an AI whose output quality drops when processing capacity is divided across too many channels.",
    "Consider an AI reaching its maximum batch size, forcing it to queue excess requests rather than process them.",
]

assert len(HUMAN_PROMPTS_DIM18) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM18)}"
assert len(AI_PROMPTS_DIM18) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM18)}"

CATEGORY_INFO_DIM18 = [
    {"name": "selective_focus",         "start": 0,  "end": 10},
    {"name": "sustained_concentration", "start": 10, "end": 20},
    {"name": "attentional_shifts",      "start": 20, "end": 30},
    {"name": "divided_attention",       "start": 30, "end": 40},
]