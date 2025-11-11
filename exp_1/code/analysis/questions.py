"""
Script name: questions.py
Purpose: Analyze total number of questions participants asked across the experiment.
    - Count questions using two methods:
        (1) question marks ("?")
        (2) regex-based interrogative detection.
    - Exclude participants who asked 0 questions in both conditions.
    - Test for effects of Condition × Sociality (Human vs Bot × Topic type).
    - Save results (CSVs, plots, stats) to results/behavior/questions/[method]/.

Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)

Outputs:
    - Separate subfolders for each analysis method:
        question_marks/
        regex/
        nonaskers/question_marks/
        nonaskers/regex/
    - Each includes:
        - per-trial and per-subject CSVs
        - combined stats file (paired t-tests + ANOVA)
        - violin plots (main effect + Condition × Sociality)
        - run log + config snapshot

Usage:
    python code/behavior/questions.py --config configs/behavior.json
    python code/behavior/questions.py --config configs/behavior.json --sub-id sub-001

Author: Rachel C. Metzgar
Date: 2025-11-10
"""

from __future__ import annotations
import os, re, sys
import pandas as pd

# --- Ensure project utils import works ---
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.generic_analysis import run_generic_main, load_experiment_data, run_generic_analysis
from utils.globals import RESULTS_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.print_helpers import print_warn, print_header, print_info

SCRIPT_NAME = "questions"
HEADER = "Total Questions — Human vs Bot × Sociality"

# ============================================================
# Regex-based question detection
# ============================================================

QUESTION_STARTS = re.compile(
    r'^\s*(who|what|when|where|why|how|which|do|does|did|can|could|would|will|should|is|are|am|was|were)\b',
    re.I,
)

def regex_question_count(text: str) -> int:
    """Count sentences that look like questions using regex patterns."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sum(bool(s.strip() and (s.strip().endswith("?") or QUESTION_STARTS.match(s.strip()))) for s in sentences)


# ============================================================
# Feature computation functions
# ============================================================

def add_question_mark_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add question count via question-mark detection."""
    df = df.copy()
    df["Question_Count"] = df["transcript_sub"].astype(str).apply(lambda x: x.count("?"))
    return df


def add_regex_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add question count via regex-based interrogative detection."""
    df = df.copy()
    df["Question_Count"] = df["transcript_sub"].apply(regex_question_count)
    return df


# ============================================================
# Re-run analysis excluding non-askers
# ============================================================

def run_excluding_nonaskers(feature_func, method: str, base_out_dir: str):
    """Rerun generic analysis after filtering out subjects who asked 0 questions in both conditions."""
    df = feature_func(load_experiment_data())

    # Identify subjects with >0 questions in either condition
    totals = df.groupby(["Subject", "Condition"])["Question_Count"].sum().unstack(fill_value=0)
    keep_subjects = totals[(totals["hum"] > 0) | (totals["bot"] > 0)].index
    df = df[df["Subject"].isin(keep_subjects)]

    if df.empty:
        print_warn("No subjects left after excluding non-askers.")
        return

    out_dir = os.path.join(base_out_dir, "nonaskers", method)
    os.makedirs(out_dir, exist_ok=True)

    print_header(f"Excluding Non-Askers — {method.upper()} Method")
    run_generic_analysis(
        df=df,
        SCRIPT_NAME=SCRIPT_NAME,
        HEADER=f"{HEADER} — Excluding Non-Askers ({method})",
        METRIC_COL="Question_Count",
        out_dir=out_dir,
    )


# ============================================================
# Main
# ============================================================

def main():
    args, cfg = parse_and_load_config("Total Questions Analysis")
    base_out_dir = os.path.join(RESULTS_DIR, SCRIPT_NAME)
    os.makedirs(base_out_dir, exist_ok=True)

    logger, _, _, _ = init_run(
        output_dir=base_out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    # -----------------------------
    # (1) Question-mark method
    # -----------------------------
    run_generic_main(
        SCRIPT_NAME=SCRIPT_NAME,
        HEADER=f"{HEADER} — Question Mark Method",
        feature_func=add_question_mark_metric,
        METRIC_COL="Question_Count",
        extra_dir="question_marks",
    )
    run_excluding_nonaskers(add_question_mark_metric, "question_marks", base_out_dir)

    # -----------------------------
    # (2) Regex method
    # -----------------------------
    run_generic_main(
        SCRIPT_NAME=SCRIPT_NAME,
        HEADER=f"{HEADER} — Regex Method",
        feature_func=add_regex_metric,
        METRIC_COL="Question_Count",
        extra_dir="regex",
    )
    run_excluding_nonaskers(add_regex_metric, "regex", base_out_dir)

    logger.info("✅ Total questions analysis complete.")
    print("\n[DONE] ✅ Total questions analysis complete.\n")


if __name__ == "__main__":
    main()
