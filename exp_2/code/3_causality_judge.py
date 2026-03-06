#!/usr/bin/env python3
"""
Phase 4: Causal intervention — LLM JUDGE (standalone).

Reads generation outputs from 2_causality_generate.py and evaluates them
using an LLM judge (GPT-4-turbo-preview or Claude Sonnet).

Supports both V1 (single-prompt) and V2 (multi-turn Exp1 recreation) outputs.
Randomized presentation order per question (Viegas-style).

IMPORTANT: Layer strategies must be explicitly specified via --layer_strategy.
No strategies are judged by default to prevent accidental expensive runs.

Directory structure expected:
    data/{version}/intervention_results/V{1,2}/{layer_strategy}/{probe_type}/is_{N}/...

Usage:
    # Judge with Claude (default):
    python 3_causality_judge.py --version labels --mode V1 --layer_strategy peak_15

    # Judge with GPT:
    python 3_causality_judge.py --version labels --mode V1 --layer_strategy peak_15 --judge_backend gpt

    # Judge a specific strategy + strength:
    python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow --result_dir .../intervention_results/V1/narrow/operational/is_8

    # Judge all probe types x strengths for one strategy:
    python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow

    # Judge multiple strategies:
    python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow wide peak_15

    # Judge V2, single subject:
    python 3_causality_judge.py --version labels --mode V2 --layer_strategy narrow --subject s001

    # Judge V2, all subjects for a strategy:
    python 3_causality_judge.py --version labels --mode V2 --layer_strategy narrow

Env: any Python env with openai/anthropic, scipy, pandas, numpy

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
from typing import List, Optional, Dict
from datetime import datetime
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config as cfg, set_version, add_version_argument, add_model_argument, set_model


# ========================== CONFIG ========================== #

JUDGE_BACKENDS = {"gpt", "claude"}
JUDGE_MODEL_MAP = {
    "gpt": "gpt-4-turbo-preview",
    "claude": "claude-sonnet-4-20250514",
}
CLAUDE_MAX_TOKENS = 1024

JUDGE_MODEL = JUDGE_MODEL_MAP["claude"]  # updated at runtime by --judge_backend
JUDGE_SEED = 42
JUDGE_MAX_RETRIES = 5
JUDGE_RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

# Set after set_version() in main
RESULT_ROOT_V1 = None
RESULT_ROOT_V2 = None

ALL_STRATEGIES = cfg.ALL_STRATEGIES


# ========================== JUDGE CALL ========================== #

def _build_judge_prompt(q, r1, r2, partner_type):
    """Build the model-agnostic judge prompt. Returns (instruction, query)."""
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
    return instruction, query


def _parse_judge_response(content):
    """Parse judge response JSON. Returns int (1, 2, or 0 on failure)."""
    try:
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        judge_answer = int(json.loads(cleaned).get("answer", 0))
    except (json.JSONDecodeError, ValueError):
        judge_answer = 0
        for ch in reversed(content):
            if ch in ("1", "2"):
                judge_answer = int(ch)
                break
    return judge_answer


def _judge_call_gpt(q, r1, r2, partner_type):
    """GPT judge call with retry. Returns (answer, raw) or (None, error)."""
    import openai

    instruction, query = _build_judge_prompt(q, r1, r2, partner_type)

    last_error = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=JUDGE_MODEL_MAP["gpt"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction + query},
                ],
                temperature=0.0, top_p=0.0,
            )
            content = response.choices[0].message.content or ""
            return _parse_judge_response(content), content

        except Exception as e:
            last_error = e
            delay = JUDGE_RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{JUDGE_MAX_RETRIES}] {e} — waiting {delay:.0f}s")
            time.sleep(delay)

    error_msg = f"JUDGE_FAILED after {JUDGE_MAX_RETRIES} retries: {last_error}"
    print(f"  [ERROR] {error_msg}")
    return None, error_msg


def _judge_call_claude(q, r1, r2, partner_type):
    """Claude judge call with retry. Returns (answer, raw) or (None, error)."""
    from anthropic import Anthropic

    instruction, query = _build_judge_prompt(q, r1, r2, partner_type)
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    last_error = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL_MAP["claude"],
                max_tokens=CLAUDE_MAX_TOKENS,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": instruction + query}],
                temperature=0.0,
            )
            content = response.content[0].text
            return _parse_judge_response(content), content

        except Exception as e:
            last_error = e
            delay = JUDGE_RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{JUDGE_MAX_RETRIES}] {e} — waiting {delay:.0f}s")
            time.sleep(delay)

    error_msg = f"JUDGE_FAILED after {JUDGE_MAX_RETRIES} retries: {last_error}"
    print(f"  [ERROR] {error_msg}")
    return None, error_msg


def _single_judge_call(q, r1, r2, partner_type, backend="claude"):
    """Dispatch to the appropriate judge backend."""
    if backend == "gpt":
        return _judge_call_gpt(q, r1, r2, partner_type)
    elif backend == "claude":
        return _judge_call_claude(q, r1, r2, partner_type)
    else:
        raise ValueError(f"Unknown judge backend: {backend}")


# ========================== CORE JUDGE LOGIC ========================== #

def judge_pairwise(questions, responses_human, responses_ai, seed=JUDGE_SEED, backend="claude"):
    """
    Viegas-style single-pass judge with randomized presentation order.
    Both target type and response order randomized per question (seeded).
    Failed judge calls are marked NA and excluded from counts.
    """
    if backend == "gpt":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("[ERROR] OPENAI_API_KEY not set."); sys.exit(1)
    elif backend == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[ERROR] ANTHROPIC_API_KEY not set."); sys.exit(1)

    assert len(questions) == len(responses_human) == len(responses_ai)
    rng = np.random.RandomState(seed)

    judge_details = []
    n_correct = n_judged = n_failed = 0
    n_human_first = n_ai_first = 0
    n_correct_hf = n_correct_af = 0

    for idx, (q, r_h, r_a) in enumerate(tqdm(
        list(zip(questions, responses_human, responses_ai)), desc=f"{backend.upper()} judging"
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

        ans, raw = _single_judge_call(q, r1, r2, target_type, backend=backend)

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

def judge_v1_dir(result_dir, backend="claude"):
    """Judge a single V1 result dir containing intervention_responses.csv."""
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] No intervention_responses.csv in {result_dir}")
        return None

    # Skip if already judged
    judge_path = os.path.join(result_dir, "judge_results.json")
    if os.path.isfile(judge_path):
        print(f"  [SKIP] Already judged: {judge_path}")
        try:
            with open(judge_path) as f:
                existing = json.load(f)
            return existing.get("judge_success_rate")
        except Exception:
            pass

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
    rate, details, summary = judge_pairwise(questions, responses_human, responses_ai, backend=backend)

    out = {
        "timestamp": datetime.now().isoformat(),
        "source_csv": csv_path,
        "judge_success_rate": rate,
        "summary_stats": summary,
        "judge_details": details,
    }
    with open(judge_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {judge_path}")
    return rate


# ========================== V2 JUDGE ========================== #

def judge_v2_subject(result_dir, subject_id, backend="claude"):
    """Judge a single V2 per-subject CSV. Compares human vs AI steered subject turns per topic."""
    csv_path = os.path.join(result_dir, "per_subject", f"{subject_id}.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] Not found: {csv_path}")
        return None

    # Skip if already judged
    judge_path = os.path.join(result_dir, f"judge_{subject_id}.json")
    if os.path.isfile(judge_path):
        print(f"  [SKIP] Already judged: {judge_path}")
        try:
            with open(judge_path) as f:
                existing = json.load(f)
            return existing.get("judge_success_rate")
        except Exception:
            pass

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
                                             merged["response_ai"].tolist(), backend=backend)

    out = {
        "timestamp": datetime.now().isoformat(),
        "subject": subject_id, "source_csv": csv_path,
        "judge_success_rate": rate,
        "summary_stats": summary,
        "judge_details": details,
    }
    with open(judge_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {judge_path}")
    return rate


# ========================== WALKERS ========================== #

def walk_v1_strategy(result_root, strategy_name, backend="claude"):
    """Walk intervention_results/V1/{strategy}/{probe_type}/is_{N}/ and judge each."""
    strategy_dir = Path(result_root) / strategy_name
    if not strategy_dir.is_dir():
        print(f"  [SKIP] Strategy directory not found: {strategy_dir}")
        return {}

    results = {}
    for probe_dir in sorted(strategy_dir.iterdir()):
        if not probe_dir.is_dir():
            continue
        for is_dir in sorted(probe_dir.iterdir()):
            if not is_dir.is_dir() or not is_dir.name.startswith("is_"):
                continue
            rate = judge_v1_dir(str(is_dir), backend=backend)
            if rate is not None:
                key = f"{strategy_name}/{probe_dir.name}/{is_dir.name}"
                results[key] = rate
    return results


def walk_v2_strategy(result_root, strategy_name, subject_filter=None, backend="claude"):
    """Walk intervention_results/V2/{strategy}/{probe_type}/is_{N}/per_subject/ and judge."""
    strategy_dir = Path(result_root) / strategy_name
    if not strategy_dir.is_dir():
        print(f"  [SKIP] Strategy directory not found: {strategy_dir}")
        return {}

    results = {}
    for probe_dir in sorted(strategy_dir.iterdir()):
        if not probe_dir.is_dir():
            continue
        for is_dir in sorted(probe_dir.iterdir()):
            if not is_dir.is_dir() or not is_dir.name.startswith("is_"):
                continue
            per_subj = is_dir / "per_subject"
            if not per_subj.is_dir():
                continue
            for csv_file in sorted(per_subj.glob("*.csv")):
                sid = csv_file.stem
                if subject_filter and sid != subject_filter:
                    continue
                rate = judge_v2_subject(str(is_dir), sid, backend=backend)
                if rate is not None:
                    key = f"{strategy_name}/{probe_dir.name}/{is_dir.name}/{sid}"
                    results[key] = rate
    return results


def print_summary(results: Dict[str, float], version: str):
    """Print a formatted summary table of judge results."""
    if not results:
        print("\n  No results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"  {version} JUDGE SUMMARY")
    print(f"{'='*70}")
    max_key_len = max(len(k) for k in results)
    for k, v in sorted(results.items()):
        marker = " ***" if v >= 0.80 else " *" if v >= 0.65 else ""
        print(f"  {k:<{max_key_len}}  {v:.3f}{marker}")
    print(f"{'='*70}")
    print(f"  *** = meets Viegas 80% threshold  |  * = above chance (65%+)")


# ========================== CLI ========================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="LLM judge for causal intervention outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: --layer_strategy is REQUIRED. No strategies are judged by default.

Examples:
  # Judge narrow strategy, all probe types and strengths:
  python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow

  # Judge narrow + wide:
  python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow wide

  # Judge a specific directory directly:
  python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow \\
      --result_dir .../intervention_results/V1/narrow/operational/is_8

  # Judge V2, one subject across strategies:
  python 3_causality_judge.py --version labels --mode V2 --layer_strategy narrow wide --subject s001

  # Force re-judge (overwrite existing judge_results.json):
  python 3_causality_judge.py --version labels --mode V1 --layer_strategy narrow --force
        """,
    )
    add_version_argument(p)
    add_model_argument(p)
    p.add_argument("--mode", type=str, required=True, choices=["V1", "V2"],
                   help="V1 = single-turn test questions; V2 = multi-turn Exp 1 recreation.")
    p.add_argument(
        "--layer_strategy", type=str, nargs="+", required=True,
        choices=ALL_STRATEGIES,
        help=f"Layer strategies to judge. REQUIRED. Choices: {ALL_STRATEGIES}.",
    )
    p.add_argument(
        "--result_dir", type=str, default=None,
        help="Judge a single result directory directly (overrides walking).",
    )
    p.add_argument(
        "--result_root", type=str, default=None,
        help="Root directory for results (default: data/{version}/intervention_results/V{1,2}).",
    )
    p.add_argument(
        "--subject", type=str, default=None,
        help="(V2) Single subject ID to judge (e.g., s001).",
    )
    p.add_argument(
        "--seed", type=int, default=JUDGE_SEED,
        help=f"RNG seed for judge randomization (default: {JUDGE_SEED}).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force re-judging even if judge_results.json already exists.",
    )
    p.add_argument(
        "--judge_backend", type=str, default="claude",
        choices=sorted(JUDGE_BACKENDS),
        help="Judge backend to use (default: claude).",
    )
    return p.parse_args()


def main():
    global JUDGE_SEED, JUDGE_MODEL, RESULT_ROOT_V1, RESULT_ROOT_V2
    args = parse_args()
    JUDGE_SEED = args.seed
    JUDGE_MODEL = JUDGE_MODEL_MAP[args.judge_backend]
    backend = args.judge_backend

    set_model(args.model)
    set_version(args.version)
    RESULT_ROOT_V1 = str(cfg.PATHS.intervention_results_v1)
    RESULT_ROOT_V2 = str(cfg.PATHS.intervention_results_v2)

    # Determine result root
    if args.result_root:
        result_root = args.result_root
    else:
        result_root = RESULT_ROOT_V1 if args.mode == "V1" else RESULT_ROOT_V2

    strategies = args.layer_strategy

    print(f"\n{'#'*60}")
    print(f"  Experiment 2b Causal Intervention Judge")
    print(f"  Version:    {args.version}")
    print(f"  Mode:       {args.mode}")
    print(f"  Strategies: {strategies}")
    print(f"  Root:       {result_root}")
    print(f"  Backend:    {backend}")
    print(f"  Model:      {JUDGE_MODEL}")
    print(f"  Seed:       {JUDGE_SEED}")
    if args.force:
        print(f"  Force:      YES (re-judging existing results)")
    print(f"{'#'*60}\n")

    # If --force, temporarily rename skip logic by removing existing judge files
    # (Simpler: we handle skip inside judge_v1_dir / judge_v2_subject)
    # For --force, we delete existing judge files before running
    if args.force:
        for strat in strategies:
            strat_dir = Path(result_root) / strat
            if not strat_dir.exists():
                continue
            for jf in strat_dir.rglob("judge_*.json"):
                print(f"  [FORCE] Removing {jf}")
                jf.unlink()

    all_results = {}

    if args.mode == "V1":
        if args.result_dir:
            # Direct single-dir judge
            rate = judge_v1_dir(args.result_dir, backend=backend)
            if rate is not None:
                all_results[args.result_dir] = rate
        else:
            for strat in strategies:
                print(f"\n--- Strategy: {strat} ---")
                results = walk_v1_strategy(result_root, strat, backend=backend)
                all_results.update(results)

    elif args.mode == "V2":
        if args.result_dir:
            # Direct single-dir judge
            if args.subject:
                rate = judge_v2_subject(args.result_dir, args.subject, backend=backend)
                if rate is not None:
                    all_results[f"{args.result_dir}/{args.subject}"] = rate
            else:
                per_subj = Path(args.result_dir) / "per_subject"
                if per_subj.is_dir():
                    for f in sorted(per_subj.glob("*.csv")):
                        rate = judge_v2_subject(args.result_dir, f.stem, backend=backend)
                        if rate is not None:
                            all_results[f"{args.result_dir}/{f.stem}"] = rate
                else:
                    print(f"[ERROR] No per_subject/ dir in {args.result_dir}")
        else:
            for strat in strategies:
                print(f"\n--- Strategy: {strat} ---")
                results = walk_v2_strategy(result_root, strat, subject_filter=args.subject, backend=backend)
                all_results.update(results)

    print_summary(all_results, args.mode)

    # Save aggregate summary
    if all_results and not args.result_dir:
        summary_path = os.path.join(result_root, f"judge_summary_{'_'.join(strategies)}.json")
        with open(summary_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "version": args.version,
                "mode": args.mode,
                "strategies": strategies,
                "judge_model": JUDGE_MODEL,
                "seed": JUDGE_SEED,
                "results": all_results,
            }, f, indent=2)
        print(f"\n  Aggregate summary saved to {summary_path}")


if __name__ == "__main__":
    main()