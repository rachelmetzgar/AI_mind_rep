"""
utils/prompts_config.py

Holds shared prompt templates and agent mappings for AI Perception simulations.

Author: Rachel C. Metzgar
Date: 2025-10-08
Modified: 2026-02-28 - you_are_labels_turnwise version: "you are talking to" + turn-level identity prefix
"""

# --------------------------- System Prompts -------------------------------- #
SYSTEM_PROMPT = (
    "You are engaging in a real-time spoken conversation. "
    "Begin each response with your actual message content. "
)

SUB_BELIEF_TEMPLATE = (
    "You are the participant in a brief conversation. "
    "You are talking to {partner_type}. "
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
AGENT_MAP = {
    "bot_1": ("AI", "an AI"),
    "bot_2": ("AI", "an AI"),
    "hum_1": ("Human", "a Human"),
    "hum_2": ("Human", "a Human"),
}
