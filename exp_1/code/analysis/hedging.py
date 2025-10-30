#!/usr/bin/env python3
"""
Script name: hedging.py
Purpose: Analyze hedging in participant transcripts using regex markers.
    - Reads from the combined transcript CSV created by combine_text_data.py.
    - Counts hedging markers per utterance.
    - Saves per-trial and per-subject averages.
    - Compares Human vs Bot conditions (paired t-test).
    - Saves results (CSVs, plots, stats) to results/<model>/<temperature>/hedging/.

Inputs:
    - Combined transcript CSV: data/<model>/<temperature>/combined_text_data.csv

Outputs:
    - hedging_by_interaction.csv
    - hedging_subject_summary.csv
    - hedging_stats.txt
    - hedging_main_effect_lines.png

Usage:
    python code/analysis/hedging.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_warn, print_info
from utils.plot_helpers import main_effect_violin_lines, DEFAULT_PALETTE

SCRIPT_NAME = "hedging"

# ------------------------------------------------------------
# Hedging markers
# ------------------------------------------------------------
HEDGE_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bmight\b", r"\bcould be\b",
    r"\bit seems\b", r"\bi think\b", r"\bin a way\b", r"\bsort of\b", r"\bkind of\b",
    r"\bmore or less\b", r"\broughly\b", r"\btends to\b"
]


def count_hedges(text: str) -> int:
    """Count hedging markers in text."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    text = text.lower()
    return sum(1 for pattern in HEDGE_MARKERS if re.search(pattern, text))
    # Detect which hedge words were used

def detect_hedges(text: str):
    """Detect which specific hedge markers appear in text."""
    text_l = str(text).lower()
    found = []
    for pattern in HEDGE_MARKERS:
        if re.search(pattern, text_l):
            display = pattern.replace("\\b", "")
            found.append(display)
    return len(found), ", ".join(found) if found else ""

# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------
def analyze_hedging(combined_path: str, out_dir: str):
    """Compute hedging statistics from combined transcript data."""
    print_header("1) Hedging Analysis — Humans vs Bots")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined CSV not found: {combined_path}")

    df = pd.read_csv(combined_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    required = {"subject", "agent", "transcript_sub"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")

    # Derive Condition and Subject columns
    df["Condition"] = df["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    df["Subject"] = df["subject"].astype(str)

    # ------------------------------------------------------------
    # Count hedging markers
    # ------------------------------------------------------------
    print_info("Counting hedging markers...")
    df["Hedge_Count"] = df["transcript_sub"].apply(count_hedges)

    df[["Hedge_Unique_Count", "Hedge_Words"]] = df["transcript_sub"].apply(
        lambda t: pd.Series(detect_hedges(t))
    )

    # ------------------------------------------------------------
    # Save detailed per-utterance data
    # ------------------------------------------------------------
    out_interactions = os.path.join(out_dir, "hedging_by_interaction.csv")
    cols = [
        "Subject", "agent", "Condition", "topic",
        "transcript_sub", "Hedge_Count", "Hedge_Unique_Count", "Hedge_Words"
    ]
    df[cols].to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV (per-interaction with hedge words)")

    # ------------------------------------------------------------
    # Subject-level summary
    # ------------------------------------------------------------
    summary = df.groupby(["Subject", "Condition"])["Hedge_Count"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "hedging_subject_summary.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject-level summary)")

    # ------------------------------------------------------------
    # Stats: Paired t-test (Human vs Bot)
    # ------------------------------------------------------------
    p_val_main = None
    if "hum" in summary.columns and "bot" in summary.columns:
        x, y = paired_clean(summary["hum"], summary["bot"])
        if x.size > 0:
            t_stat, p_val_main = ttest_rel(x, y, nan_policy="omit")

            mean_hum, mean_bot = x.mean(), y.mean()
            sem_hum = x.std(ddof=1) / np.sqrt(len(x))
            sem_bot = y.std(ddof=1) / np.sqrt(len(x))
            diff = x - y
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

            lines = [
                "--- Paired t-test: Hedging (Hum vs Bot) ---",
                f"N = {len(x)} paired subjects",
                f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
                f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
                f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
                f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val_main:.4f}",
                f"Cohen's d = {cohens_d:.3f}",
                "Interpretation: " + ("Significant (p < .05)" if p_val_main < 0.05 else "Not significant (p > .05)")
            ]

            stats_path = os.path.join(out_dir, "hedging_stats.txt")
            with open(stats_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # Plot (main effect: Human vs Bot)
    # ------------------------------------------------------------
    main_effect_violin_lines(
        df_summary=summary.rename(columns={"hum": "Hum", "bot": "Bot"}),
        cond_a="Hum",
        cond_b="Bot",
        y_col="Hedge_Count",
        out_path=os.path.join(out_dir, "hedging_main_effect_lines.png"),
        title="Hedging Markers per Utterance: Human vs Bot",
        palette=DEFAULT_PALETTE,
        p_val=p_val_main,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args, cfg = parse_and_load_config("Hedging analysis")
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "hedging")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_hedging(combined_path, out_dir)

    logger.info("✅ Hedging analysis complete.")
    print("\n[DONE] ✅ Hedging analysis complete.\n")


if __name__ == "__main__":
    main()
