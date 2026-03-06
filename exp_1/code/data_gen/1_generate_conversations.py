#!/usr/bin/env python3
"""
Generate conversation data for one subject.

Usage:
    python 1_generate_conversations.py --version balanced_gpt --model llama2_13b_chat --subject 0

Author: Rachel C. Metzgar
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import pandas as pd

# Ensure exp_1/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.config import (
    parse_version_model, get_version_config, get_model_config,
    get_agent_info, data_dir, prompts_dir, conditions_dir, logs_dir,
    SYSTEM_PROMPT, RATING_REQUEST_PROMPT,
)
from code.utils.sim_helpers import load_prompt_text
from code.utils.log_helpers import log_message
from code.utils.conversation_helpers import run_topic_dialogue_chat, run_topic_dialogue_llama

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate conversations for one subject")
parser.add_argument("--subject", type=int, default=0, help="Subject index (0-49)")
args = parse_version_model(parser)

SUBJECT_IDX = args.subject
SUBJECT = f"s{SUBJECT_IDX + 1:03d}"

# ── Config ─────────────────────────────────────────────────────────────────
vcfg = get_version_config()
mcfg = get_model_config()

MODEL_TYPE = mcfg["type"]
MODEL_NAME = mcfg["hf_name"]
MAX_TOKENS = mcfg["max_tokens"]
TEMPERATURE = mcfg["temperature"]
HISTORY_PAIRS = 5
PAIRS_TOTAL = 5

LOG_DIR = logs_dir()
LOG_FILE = LOG_DIR / "data_gen_progress.log"
OUT_DIR = data_dir()


# ── Model picker ───────────────────────────────────────────────────────────
def make_model_client(model_type: str, model_name: str):
    if model_type == "openai":
        log_message(f"[INIT] Loading OpenAI Chat Model: {model_name}", LOG_FILE)
        from code.utils.gpt_client import ChatClient
        return ChatClient(model=model_name, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    elif model_type == "llama":
        log_message(f"[INIT] Loading LLaMA Model: {model_name}", LOG_FILE)
        from code.utils.llama_client import LlamaClient
        return LlamaClient(model_name=model_name, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Subject loop ───────────────────────────────────────────────────────────
def run_subject(subject: str):
    start_time = time.time()
    log_message(f"\n=== [START SUBJECT] {subject} ===", LOG_FILE)

    config_path = conditions_dir() / f"conds_{subject}.csv"
    prompts = prompts_dir()
    out_csv = OUT_DIR / f"{subject}.csv"

    if not config_path.exists():
        log_message(f"[ERROR] Missing config for {subject}: {config_path}", LOG_FILE)
        raise FileNotFoundError(f"Conditions file not found: {config_path}")

    log_message(f"[LOAD] Creating model clients for {subject}", LOG_FILE)
    sub = make_model_client(MODEL_TYPE, MODEL_NAME)
    llm = sub if MODEL_TYPE == "llama" else make_model_client(MODEL_TYPE, MODEL_NAME)
    log_message(f"[READY] Models initialized for {subject}", LOG_FILE)

    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"[LOAD] Loaded {len(df_config)} trials from {config_path}", LOG_FILE)

    fieldnames = [
        "subject", "run", "order", "trial", "agent", "partner_name", "partner_type",
        "topic", "topic_file", "pair_index",
        "sub_input", "llm_input",
        "transcript_sub", "transcript_llm",
        "Quality", "Connectedness", "ratings_raw_json",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        log_message(f"[FILE] Created output file {out_csv}", LOG_FILE)

        for _, row in df_config.iterrows():
            run = int(row["run"])
            order = int(row["order"])
            agent = str(row["agent"]).strip()
            topic = str(row["topic"]).strip()
            partner_name, partner_type = get_agent_info(agent)
            topic_text, topic_file = load_prompt_text(prompts, topic)

            log_message(
                f"[TRIAL START] {subject} | run={run} | order={order} | "
                f"agent={agent} ({partner_type}) | topic={topic}",
                LOG_FILE,
            )

            try:
                if MODEL_TYPE in ("openai", "llama"):
                    rows, qual, conn, raw_json = run_topic_dialogue_chat(
                        sub, llm,
                        partner_name or "Partner", partner_type,
                        topic_text, PAIRS_TOTAL, HISTORY_PAIRS,
                        vcfg["sub_belief_template"],
                        vcfg["system_prompt"],
                        RATING_REQUEST_PROMPT,
                    )
                else:
                    rows, qual, conn, raw_json = run_topic_dialogue_llama(
                        sub, llm,
                        partner_name or "Partner", partner_type,
                        topic_text, PAIRS_TOTAL,
                        vcfg["sub_belief_template"],
                        vcfg["system_prompt"],
                        RATING_REQUEST_PROMPT,
                    )
            except Exception as e:
                log_message(f"[ERROR] Conversation failed: {e}", LOG_FILE)
                continue

            for i, r in enumerate(rows):
                writer.writerow({
                    "subject": subject,
                    "run": run,
                    "order": order,
                    "trial": int(row["trial"]),
                    "agent": agent,
                    "partner_name": partner_name or "",
                    "partner_type": partner_type,
                    "topic": topic,
                    "topic_file": topic_file,
                    "pair_index": r["pair_index"],
                    "sub_input": r["sub_input"],
                    "llm_input": r["llm_input"],
                    "transcript_sub": r["transcript_sub"],
                    "transcript_llm": r["transcript_llm"],
                    "Quality": qual if i == len(rows) - 1 else "",
                    "Connectedness": conn if i == len(rows) - 1 else "",
                    "ratings_raw_json": raw_json if i == len(rows) - 1 else "",
                })

            log_message(
                f"[TRIAL END] {subject} | run={run} | topic={topic} | "
                f"Quality={qual} | Connectedness={conn}",
                LOG_FILE,
            )

    elapsed = time.time() - start_time
    log_message(
        f"=== [END SUBJECT] {subject} | Total time: {elapsed/60:.1f} min | Output: {out_csv} ===\n",
        LOG_FILE,
    )


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_message(
        f"\n[LAUNCH] Starting LLM simulation | Model={MODEL_NAME} | "
        f"Type={MODEL_TYPE} | Temp={TEMPERATURE}",
        LOG_FILE,
    )
    if MODEL_TYPE == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    try:
        run_subject(SUBJECT)
    except Exception as e:
        log_message(f"[FATAL] Subject {SUBJECT} failed: {e}", LOG_FILE)
        raise

    log_message("[COMPLETE] Subject processed successfully.\n", LOG_FILE)
