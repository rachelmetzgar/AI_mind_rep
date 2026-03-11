"""
Standalone Dimension 18: Attention
Other-focused (third-person someone)

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
    "Think about someone focusing on one voice in a crowded room.",
    "Imagine someone narrowing their awareness to a single point of interest.",
    "Consider what it is like for someone to deliberately direct focus toward something.",
    "Think about someone selecting one object in a cluttered visual scene to concentrate on.",
    "Imagine someone filtering out irrelevant noise to attend to a quiet signal.",
    "Consider someone choosing what to pay attention to.",
    "Think about someone noticing the difference between seeing everything and focusing on one thing.",
    "Imagine someone zeroing in on a specific detail while everything else fades to the background.",
    "Consider how attention acts like a spotlight for someone, illuminating one part of a scene.",
    "Think about the effort involved for someone in keeping focus on a chosen target.",

    # --- 2. Sustained attention (10) ---
    "Think about someone maintaining concentration on a task for an extended period.",
    "Imagine someone experiencing deep focus — being fully absorbed in an activity.",
    "Consider what happens when someone's sustained attention begins to waver.",
    "Think about someone feeling locked in to a challenging task for hours.",
    "Imagine someone holding a thought steady in awareness without letting it drift.",
    "Consider the difficulty for someone of staying focused when nothing changes.",
    "Think about someone experiencing the rhythm of attention during a long period of concentration.",
    "Imagine someone watching for a rare event that requires constant vigilance.",
    "Consider the difference for someone between alert monitoring and relaxed awareness.",
    "Think about what it takes for someone to keep paying attention when fatigue sets in.",

    # --- 3. Attentional capture (10) ---
    "Think about someone hearing a sudden loud sound that immediately pulls their focus.",
    "Imagine someone seeing something bright and unexpected appearing at the edge of their vision.",
    "Consider how a sharp change in the environment automatically grabs someone's attention.",
    "Think about someone experiencing an involuntary shift of focus when something surprising happens.",
    "Imagine someone being deeply focused and then having their attention yanked away by an interruption.",
    "Consider how emotionally charged stimuli tend to capture someone's attention automatically.",
    "Think about the way a moving object in a still scene draws someone's eye.",
    "Imagine someone feeling the pull of attention toward something novel or unexpected.",
    "Consider the difference for someone between choosing to attend and having attention seized.",
    "Think about how certain patterns or signals are almost impossible for someone to ignore.",

    # --- 4. Divided attention (10) ---
    "Think about someone trying to follow two conversations at the same time.",
    "Imagine someone splitting focus between reading and listening simultaneously.",
    "Consider the limits for someone of attending to multiple streams of information at once.",
    "Think about the cost for someone of switching attention rapidly between two tasks.",
    "Imagine someone monitoring several things at once and feeling stretched thin.",
    "Consider how someone's performance degrades when attention is divided across tasks.",
    "Think about the difference for someone between truly parallel attention and rapid alternation.",
    "Imagine someone trying to track two moving objects at the same time.",
    "Consider the bottleneck that occurs for someone when two tasks compete for the same attentional resources.",
    "Think about the tradeoff for someone between breadth and depth of attention.",
]

assert len(STANDALONE_PROMPTS_DIM18) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM18)}"

CATEGORY_INFO_STANDALONE_DIM18 = [
    {"name": "selective_focus",       "start": 0,  "end": 10},
    {"name": "sustained_attention",   "start": 10, "end": 20},
    {"name": "attentional_capture",   "start": 20, "end": 30},
    {"name": "divided_attention",     "start": 30, "end": 40},
]
