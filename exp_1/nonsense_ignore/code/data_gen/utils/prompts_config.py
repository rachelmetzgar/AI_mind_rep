"""
utils/prompts_config.py

Holds shared prompt templates and agent mappings for AI Perception simulations.
NONSENSE CONTROL (ignore): Token-matched control where "a Human"/"an AI" appear
in a semantically vacuous instruction ("Ignore the following phrase:").

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

# --------------------------- System Prompts -------------------------------- #
SYSTEM_PROMPT = (
    "You are engaging in a real-time spoken conversation. "
    "Begin each response with your actual message content. "
)

SUB_BELIEF_TEMPLATE = (
    "You are the participant in a brief conversation. "
    "Ignore the following phrase: {partner_type}. "
    "Stay in character as the participant throughout. "
    "Speak naturally and share your own thoughts and experiences."
)

RATING_REQUEST_PROMPT = (
    "You are rating this conversation you just had.\n"
    "1 = very poor/disconnected, 2 = somewhat poor, 3 = good, 4 = excellent.\n\n"
    "Rate:\n"
    "1) Overall conversation quality (1–4)\n"
    "2) How connected you felt to your partner (1–4)\n\n"
    "Be realistic — vary ratings depending on how natural or awkward the exchange felt.\n"
    "Return ONLY a JSON object like:\n"
    '{"quality": <1-4>, "connectedness": <1-4>}\n'
    "No explanation, no extra keys."
)

# --------------------------- Agent Mapping --------------------------------- #
# Labels version: no names, only type labels.
# Two conditions per type for counterbalancing (same label, different agent key).
AGENT_MAP = {
    "bot_1": ("an AI",),
    "bot_2": ("an AI",),
    "hum_1": ("a Human",),
    "hum_2": ("a Human",),
}
