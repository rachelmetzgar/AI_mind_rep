"""
utils/prompts_config.py

Holds shared prompt templates and agent mappings for AI Perception simulations.

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

# --------------------------- System Prompts -------------------------------- #
SYSTEM_PROMPT = """You are engaging in a real-time spoken conversation.'"""

SUB_BELIEF_TEMPLATE = (
    "You are the participant in a brief conversation. "
    "You believe you are speaking with {partner_name} ({partner_type}). "
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
    "bot_1": ("ChatGPT", "an AI"),
    "bot_2": ("Gemini", "an AI"),
    "hum_1": ("Casey", "a Human"),
    "hum_2": ("Sam", "a Human"),
}
