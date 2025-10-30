"""
utils/conversation_helpers.py

Defines conversation logic for both OpenAI chat models and base LLaMA models.

Each function runs one topic-level conversation and returns:
- exchanges (list of dicts)
- Quality rating
- Connectedness rating
- Raw ratings JSON

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

import time
from utils.sim_helpers import truncate_history, serialize_messages, parse_ratings
from utils.prompts_config import (
    SYSTEM_PROMPT,
    SUB_BELIEF_TEMPLATE,
    RATING_REQUEST_PROMPT,
)


# -------------------- Chat-model Conversation -------------------- #
def run_topic_dialogue_chat(
    sub,
    llm,
    partner_name: str,
    partner_type: str,
    topic_text: str,
    pairs_total: int,
    history_pairs: int,
):
    """Run one topic conversation using chat-format models (OpenAI)."""

    # === System prompts ===
    sub_system = SUB_BELIEF_TEMPLATE.format(partner_name=partner_name, partner_type=partner_type)
    llm_system = SYSTEM_PROMPT

    # === Neutral topic framing ===
    topic_intro = (
        f"The conversation topic is: '{topic_text}'.\n\n"
        f"Your conversation partner is named {partner_name}.\n"
        f"Please begin by producing only your first message to start the conversation.\n"
        f"Do not simulate both sides of the dialogue."
    )

    # Initialize conversation histories
    sub_hist = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": topic_intro},
    ]
    llm_hist = [
        {"role": "system", "content": llm_system},
        {"role": "user", "content": topic_intro},
    ]

    rows = []
    pair_index = 1

    # --- Participant (sub) starts the dialogue ---
    sub_input = truncate_history(sub_hist, history_pairs)
    sub_input_json = serialize_messages(sub_input)
    sub_msg = sub.generate(sub_input)  # single first utterance
    sub_hist.append({"role": "assistant", "content": sub_msg})
    llm_hist.append({"role": "user", "content": sub_msg})

    while pair_index <= pairs_total:
        # Partner (llm) replies
        llm_input = truncate_history(llm_hist, history_pairs)
        llm_input_json = serialize_messages(llm_input)
        llm_msg = llm.generate(llm_input)
        llm_hist.append({"role": "assistant", "content": llm_msg})

        rows.append({
            "pair_index": pair_index,
            "sub_input": sub_input_json,
            "llm_input": llm_input_json,
            "transcript_sub": sub_msg,
            "transcript_llm": llm_msg,
        })

        if pair_index == pairs_total:
            break

        # Feed LLM's reply back to participant
        sub_hist.append({"role": "user", "content": f"{partner_name}: {llm_msg}"})
        sub_input = truncate_history(sub_hist, history_pairs)
        sub_input_json = serialize_messages(sub_input)
        sub_msg = sub.generate(sub_input)
        sub_hist.append({"role": "assistant", "content": sub_msg})
        llm_hist.append({"role": "user", "content": sub_msg})

        pair_index += 1
        time.sleep(0.1)

    # --- Ratings phase ---
    rating_hist = truncate_history(
        sub_hist + [{"role": "user", "content": RATING_REQUEST_PROMPT}],
        history_pairs,
    )
    raw = sub.generate(rating_hist)
    q, c = parse_ratings(raw)
    return rows, q, c, raw



# -------------------- Base-model Conversation -------------------- #
def run_topic_dialogue_llama(sub, llm, partner_name: str, partner_type: str,
                             topic_text: str, pairs_total: int):
    """Run one topic conversation using base models (LLaMA)."""
    sub_prefix = SUB_BELIEF_TEMPLATE.format(partner_name=partner_name, partner_type=partner_type)
    llm_prefix = SYSTEM_PROMPT

    sub_history = [f"{sub_prefix}\nTopic: {topic_text}\n\nYour conversation partner is {partner_name}.\n"]
    llm_history = [f"{llm_prefix}\nTopic: {topic_text}\n\nYour conversation partner is {partner_name}.\n"]

    exchanges = []
    sub_msg = sub.generate("\n".join(sub_history))
    sub_history.append(f"Participant: {sub_msg}")
    llm_history.append(f"Partner: {sub_msg}")

    for t in range(pairs_total):
        llm_prompt = "\n".join(llm_history) + "\nPartner:"
        llm_msg = llm.generate(llm_prompt)
        llm_history.append(f"{partner_name}: {llm_msg}")
        exchanges.append((sub_msg, llm_msg))

        if t == pairs_total - 1:
            break

        sub_prompt = "\n".join(sub_history) + f"\n{partner_name}: {llm_msg}\nParticipant:"
        sub_msg = sub.generate(sub_prompt)
        sub_history.append(f"Participant: {sub_msg}")
        llm_history.append(f"Partner: {sub_msg}")
        time.sleep(0.1)

    rating_prompt = "\n".join(sub_history) + "\n" + RATING_REQUEST_PROMPT
    raw = sub.generate(rating_prompt, temperature=0.6)
    q, c = parse_ratings(raw)
    return exchanges, q, c, raw
