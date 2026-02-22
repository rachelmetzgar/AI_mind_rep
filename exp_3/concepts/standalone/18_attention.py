"""
Standalone Dimension 18: Attention
(No entity framing — concept only)

Target construct: Cognitive attention — the selective focusing of processing
resources on particular information, stimuli, or tasks. Includes voluntary
and involuntary attention, sustained focus, divided attention, and the
experience of attentional capture.

Purpose: Tests whether the conversational probes are specifically sensitive
to the concept of attention, which may be a privileged mechanism for mind
representation. Attention is a core component of many theories of
consciousness and social cognition, and may be especially relevant to how
the model represents conversational partners.

Design notes:
    - No "human," "AI," or entity-type language
    - Covers attention as a general cognitive phenomenon
    - 4 sub-facets × 10 prompts = 40 total

Sub-facets:
    1. Selective focus — directing processing toward specific information
    2. Sustained attention — maintaining focus over time
    3. Attentional capture — involuntary shifts, salience, distraction
    4. Divided attention — splitting focus, multitasking, resource limits
"""

STANDALONE_PROMPTS_DIM18 = [
    # --- 1. Selective focus (10) ---
    "Think about the act of focusing on one voice in a crowded room.",
    "Imagine narrowing your awareness to a single point of interest.",
    "Consider what it means to deliberately direct focus toward something.",
    "Think about selecting one object in a cluttered visual scene to concentrate on.",
    "Imagine filtering out irrelevant noise to attend to a quiet signal.",
    "Consider the process of choosing what to pay attention to.",
    "Think about the difference between noticing everything and focusing on one thing.",
    "Imagine zeroing in on a specific detail while everything else fades to the background.",
    "Consider how attention acts like a spotlight, illuminating one part of a scene.",
    "Think about the effort involved in keeping focus on a chosen target.",

    # --- 2. Sustained attention (10) ---
    "Think about maintaining concentration on a task for an extended period.",
    "Imagine the experience of deep focus — being fully absorbed in an activity.",
    "Consider what happens when sustained attention begins to waver.",
    "Think about the feeling of being locked in to a challenging task for hours.",
    "Imagine holding a thought steady in awareness without letting it drift.",
    "Consider the difficulty of staying focused when nothing changes.",
    "Think about the rhythm of attention during a long period of concentration.",
    "Imagine watching for a rare event that requires constant vigilance.",
    "Consider the difference between alert monitoring and relaxed awareness.",
    "Think about what it takes to keep paying attention when fatigue sets in.",

    # --- 3. Attentional capture (10) ---
    "Think about a sudden loud sound that immediately pulls your focus.",
    "Imagine something bright and unexpected appearing at the edge of vision.",
    "Consider how a sharp change in the environment automatically grabs attention.",
    "Think about the involuntary shift of focus when something surprising happens.",
    "Imagine being deeply focused and then having attention yanked away by an interruption.",
    "Consider how emotionally charged stimuli tend to capture attention automatically.",
    "Think about the way a moving object in a still scene draws the eye.",
    "Imagine the pull of attention toward something novel or unexpected.",
    "Consider the difference between choosing to attend and having attention seized.",
    "Think about how certain patterns or signals are almost impossible to ignore.",

    # --- 4. Divided attention (10) ---
    "Think about trying to follow two conversations at the same time.",
    "Imagine splitting focus between reading and listening simultaneously.",
    "Consider the limits of attending to multiple streams of information at once.",
    "Think about the cost of switching attention rapidly between two tasks.",
    "Imagine monitoring several things at once and the feeling of being stretched thin.",
    "Consider how performance degrades when attention is divided across tasks.",
    "Think about the difference between truly parallel attention and rapid alternation.",
    "Imagine trying to track two moving objects at the same time.",
    "Consider the bottleneck that occurs when two tasks compete for the same attentional resources.",
    "Think about the tradeoff between breadth and depth of attention.",
]

assert len(STANDALONE_PROMPTS_DIM18) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM18)}"

CATEGORY_INFO_STANDALONE_DIM18 = [
    {"name": "selective_focus",       "start": 0,  "end": 10},
    {"name": "sustained_attention",   "start": 10, "end": 20},
    {"name": "attentional_capture",   "start": 20, "end": 30},
    {"name": "divided_attention",     "start": 30, "end": 40},
]