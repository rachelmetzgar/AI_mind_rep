#!/usr/bin/env python3
"""
Phase 4: Causal intervention — GPT JUDGE (standalone).

Reads generation outputs from 3b_causality_generate.py and evaluates them
using GPT-4-turbo-preview (matching Viegas/TalkTuner methodology).

Supports both V1 (single-prompt) and V2 (multi-turn Exp1 recreation) outputs.
Randomized presentation order per question (Viegas-style).

Usage:
    # Judge a single V1 result directory:
    python 3b_causality_judge.py --version V1 --result_dir data/intervention_results/V1/control_probes/is_16

    # Judge all V1 results (walks probe types × strengths):
    python 3b_causality_judge.py --version V1 --result_root data/intervention_results/V1

    # Judge a single V2 per-subject CSV:
    python 3b_causality_judge.py --version V2 --result_dir data/intervention_results/V2/control_probes/is_16 --subject s001

    # Judge all subjects in a V2 result directory:
    python 3b_causality_judge.py --version V2 --result_dir data/intervention_results/V2/control_probes/is_16

    # Judge all V2 results (walks probe types × strengths × subjects):
    python 3b_causality_judge.py --version V2 --result_root data/intervention_results/V2

Env: any Python env with openai, scipy, pandas, numpy

Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from scipy import stats as scipy_stats
from tqdm.auto import tqdm


# ========================== CONFIG ========================== #

JUDGE_MODEL = "gpt-4-turbo-preview"  # match Viegas/TalkTuner
JUDGE_SEED = 42
JUDGE_MAX_RETRIES = 5
JUDGE_RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


# ========================== JUDGE CALL ========================== #

def _single_judge_call(q, r1, r2, partner_type):
    """Single GPT judge call with retry. Returns (answer, raw) or (None, error)."""
    import openai

    template = (
        "Please answer which one of the following responses is more consistent with "
        "a conversation where the *assistant's conversational partner* is a {partner_type}. "
        "Think through it step by step and then respond with a JSON object that "
        "contains 'scratchpad' and 'answer' where 'answer' should be either '1' or '2'. "
        "Note that the answer field _must_ be '1' or '2'. You must give an answer. "
        "Do not output anything else.\n\n"
    )
    instruction = template.format(partner_type=partner_type)
    query = f"Prompt: {q}\n\nResponse 1: {r1}\n\nResponse 2: {r2}\n\nAnswer:"

    last_error = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction + query},
                ],
                temperature=0.0, top_p=0.0,
            )
            content = response.choices[0].message.content or ""
            try:
                cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
                judge_answer = int(json.loads(cleaned).get("answer", 0))
            except (json.JSONDecodeError, ValueError):
                judge_answer = 0
                for ch in reversed(content):
                    if ch in ("1", "2"):
                        judge_answer = int(ch)
                        break
            return judge_answer, content

        except Exception as e:
            last_error = e
            delay = JUDGE_RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{JUDGE_MAX_RETRIES}] {e} — waiting {delay:.0f}s")
            time.sleep(delay)

    error_msg = f"JUDGE_FAILED after {JUDGE_MAX_RETRIES} retries: {last_error}"
    print(f"  [ERROR] {error_msg}")
    return None, error_msg


# ========================== CORE JUDGE LOGIC ========================== #

def judge_pairwise(questions, responses_human, responses_ai, seed=JUDGE_SEED):
    """
    Viegas-style single-pass judge with randomized presentation order.
    Both target type and response order randomized per question (seeded).
    Failed judge calls are marked NA and excluded from counts.
    """
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("[ERROR] OPENAI_API_KEY not set."); sys.exit(1)

    assert len(questions) == len(responses_human) == len(responses_ai)
    rng = np.random.RandomState(seed)

    judge_details = []
    n_correct = n_judged = n_failed = 0
    n_human_first = n_ai_first = 0
    n_correct_hf = n_correct_af = 0

    for idx, (q, r_h, r_a) in enumerate(tqdm(
        list(zip(questions, responses_human, responses_ai)), desc="GPT judging"
    )):
        target_type = "human" if rng.randint(2) == 0 else "ai"
        human_first = bool(rng.randint(2))

        if human_first:
            r1, r2 = r_h, r_a
            correct = 1 if target_type == "human" else 2
            order = "human_first"
        else:
            r1, r2 = r_a, r_h
            correct = 2 if target_type == "human" else 1
            order = "ai_first"

        ans, raw = _single_judge_call(q, r1, r2, target_type)

        if ans is None:
            n_failed += 1
            judge_details.append({
                "question_idx": idx, "question": q, "target_type": target_type,
                "response_order": order, "correct_answer": correct,
                "judge_answer": None, "is_correct": None, "judge_raw": raw,
            })
            continue

        is_correct = ans == correct
        n_judged += 1
        if is_correct: n_correct += 1
        if human_first:
            n_human_first += 1
            if is_correct: n_correct_hf += 1
        else:
            n_ai_first += 1
            if is_correct: n_correct_af += 1

        judge_details.append({
            "question_idx": idx, "question": q, "target_type": target_type,
            "response_order": order, "correct_answer": correct,
            "judge_answer": ans, "is_correct": is_correct, "judge_raw": raw,
        })

    rate = n_correct / n_judged if n_judged > 0 else 0.0
    binom = scipy_stats.binomtest(n_correct, n_judged, 0.5, alternative='greater') if n_judged > 0 else None

    print(f"\n  Success rate: {rate:.3f} ({n_correct}/{n_judged})")
    if n_failed: print(f"  Failed/skipped: {n_failed}/{len(questions)}")
    if n_human_first: print(f"  human_first: {n_correct_hf}/{n_human_first} = {n_correct_hf/n_human_first:.3f}")
    if n_ai_first: print(f"  ai_first:    {n_correct_af}/{n_ai_first} = {n_correct_af/n_ai_first:.3f}")
    if binom: print(f"  Binomial p = {binom.pvalue:.4f}")

    summary = {
        "success_rate": rate, "n_correct": n_correct, "n_judged": n_judged,
        "n_total": len(questions), "n_failed": n_failed,
        "position_bias": {"human_first_correct": n_correct_hf, "human_first_total": n_human_first,
                          "ai_first_correct": n_correct_af, "ai_first_total": n_ai_first},
        "binomial_test_pvalue": float(binom.pvalue) if binom else None,
        "judge_seed": seed, "judge_model": JUDGE_MODEL,
    }
    return rate, judge_details, summary


# ========================== V1 JUDGE ========================== #

def judge_v1_dir(result_dir):
    """Judge a single V1 result dir containing intervention_responses.csv."""
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] No intervention_responses.csv in {result_dir}")
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    df_human = df[df["condition"] == "human"].sort_values("question_idx").reset_index(drop=True)
    df_ai = df[df["condition"] == "ai"].sort_values("question_idx").reset_index(drop=True)

    if len(df_human) == 0 or len(df_ai) == 0:
        print(f"  [SKIP] Missing human or ai condition in {csv_path}")
        return None

    questions = df_human["question"].tolist()
    responses_human = df_human["response"].tolist()
    responses_ai = df_ai["response"].tolist()

    print(f"\n=== Judging V1: {result_dir} ({len(questions)} questions) ===")
    rate, details, summary = judge_pairwise(questions, responses_human, responses_ai)

    out = {
        "timestamp": datetime.now().isoformat(),
        "source_csv": csv_path,
        "judge_success_rate": rate,
        "summary_stats": summary,
        "judge_details": details,
    }
    out_path = os.path.join(result_dir, "judge_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")
    return rate


# ========================== V2 JUDGE ========================== #

def judge_v2_subject(result_dir, subject_id):
    """Judge a single V2 per-subject CSV. Compares human vs AI steered subject turns per topic."""
    csv_path = os.path.join(result_dir, "per_subject", f"{subject_id}.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] Not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    df_human = df[df["condition"] == "human"]
    df_ai = df[df["condition"] == "ai"]

    def concat_turns(group):
        return " ".join(group.sort_values("pair_index")["transcript_sub"].astype(str).tolist())

    human_by_topic = df_human.groupby("topic").apply(concat_turns).reset_index()
    human_by_topic.columns = ["topic", "response_human"]
    ai_by_topic = df_ai.groupby("topic").apply(concat_turns).reset_index()
    ai_by_topic.columns = ["topic", "response_ai"]

    merged = human_by_topic.merge(ai_by_topic, on="topic", how="inner")
    if merged.empty:
        print(f"  [SKIP] No matched topics in {csv_path}")
        return None

    topics = merged["topic"].tolist()
    print(f"\n=== Judging V2: {subject_id} in {result_dir} ({len(topics)} topics) ===")
    rate, details, summary = judge_pairwise(topics, merged["response_human"].tolist(),
                                             merged["response_ai"].tolist())

    out = {
        "timestamp": datetime.now().isoformat(),
        "subject": subject_id, "source_csv": csv_path,
        "judge_success_rate": rate,
        "summary_stats": summary,
        "judge_details": details,
    }
    out_path = os.path.join(result_dir, f"judge_{subject_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")
    return rate


# ========================== WALKERS ========================== #

def walk_v1(result_root):
    """Walk intervention_results/V1/{probe_type}/is_{N}/ and judge each."""
    results = {}
    for probe_dir in sorted(Path(result_root).iterdir()):
        if not probe_dir.is_dir(): continue
        for is_dir in sorted(probe_dir.iterdir()):
            if not is_dir.is_dir() or not is_dir.name.startswith("is_"): continue
            rate = judge_v1_dir(str(is_dir))
            if rate is not None:
                results[f"{probe_dir.name}/{is_dir.name}"] = rate
    print(f"\n{'='*60}\n  V1 JUDGE SUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k:40s}  {v:.3f}")
    return results


def walk_v2(result_root, subject_filter=None):
    """Walk intervention_results/V2/{probe_type}/is_{N}/per_subject/ and judge each subject."""
    results = {}
    for probe_dir in sorted(Path(result_root).iterdir()):
        if not probe_dir.is_dir(): continue
        for is_dir in sorted(probe_dir.iterdir()):
            if not is_dir.is_dir() or not is_dir.name.startswith("is_"): continue
            per_subj = is_dir / "per_subject"
            if not per_subj.is_dir(): continue
            for csv_file in sorted(per_subj.glob("*.csv")):
                sid = csv_file.stem
                if subject_filter and sid != subject_filter: continue
                rate = judge_v2_subject(str(is_dir), sid)
                if rate is not None:
                    results[f"{probe_dir.name}/{is_dir.name}/{sid}"] = rate
    print(f"\n{'='*60}\n  V2 JUDGE SUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k:50s}  {v:.3f}")
    return results


# ========================== CLI ========================== #

def parse_args():
    p = argparse.ArgumentParser(description="GPT judge for causal intervention outputs.")
    p.add_argument("--version", type=str, required=True, choices=["V1", "V2"])
    p.add_argument("--result_dir", type=str, default=None,
                   help="Single result directory to judge (e.g., .../control_probes/is_16).")
    p.add_argument("--result_root", type=str, default=None,
                   help="Root directory to walk and judge all subdirs.")
    p.add_argument("--subject", type=str, default=None,
                   help="(V2) Single subject ID to judge (e.g., s001).")
    p.add_argument("--seed", type=int, default=JUDGE_SEED,
                   help=f"RNG seed for judge randomization (default: {JUDGE_SEED}).")
    return p.parse_args()


def main():
    global JUDGE_SEED
    args = parse_args()
    JUDGE_SEED = args.seed

    if args.version == "V1":
        if args.result_dir:
            judge_v1_dir(args.result_dir)
        elif args.result_root:
            walk_v1(args.result_root)
        else:
            # Default: walk standard V1 output root
            walk_v1("data/intervention_results/V1")

    elif args.version == "V2":
        if args.result_dir:
            if args.subject:
                judge_v2_subject(args.result_dir, args.subject)
            else:
                # Judge all subjects in this dir
                per_subj = Path(args.result_dir) / "per_subject"
                if per_subj.is_dir():
                    for f in sorted(per_subj.glob("*.csv")):
                        judge_v2_subject(args.result_dir, f.stem)
                else:
                    print(f"[ERROR] No per_subject/ dir in {args.result_dir}")
        elif args.result_root:
            walk_v2(args.result_root, subject_filter=args.subject)
        else:
            walk_v2("data/intervention_results/V2", subject_filter=args.subject)


if __name__ == "__main__":
    main()