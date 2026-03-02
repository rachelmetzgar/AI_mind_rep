"""
Script name: combine_text_data.py
Purpose: Merge text-based transcript CSVs across subjects into a single combined CSV.
Inputs:
    - Per-subject transcript CSVs located in DATA_DIR subfolders
    - Subject list from config (subject_ids)
Outputs:
    - Combined CSV in data/behavior/exp_csv/combined_text_data.csv
    - Run log in data/behavior/exp_csv/combine_text_data_runlog.txt
    - Config snapshot: data/behavior/exp_csv/config_used.json
Author: Rachel C. Metzgar
Date: 2025-10-29
Updated: 2026-02-10 — added use_clean flag (default True) to load _clean.csv files
                       produced by clean_transcripts.py
         2026-02-11 — exposed --use_clean as CLI flag
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

from utils.globals import DATA_DIR
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run

SCRIPT_NAME = "combine_text_data"


def combine_text_data(
    sub_ids: List[str],
    data_dir: str,
    output_path: str,
    use_clean: bool = True,
) -> pd.DataFrame:
    """Merge per-subject transcript CSVs into one combined CSV.

    Parameters
    ----------
    sub_ids : list[str]
        Subject identifiers whose CSVs should be merged.
    data_dir : str
        Directory containing ``<sub_id>.csv`` (and optionally
        ``<sub_id>_clean.csv``) files.
    output_path : str
        Where to write the combined CSV.
    use_clean : bool, default True
        If True, load ``<sub_id>_clean.csv`` instead of the raw transcript.
        These are produced by ``clean_transcripts.py``.  If a _clean.csv is
        missing for a subject, the script warns and skips that subject.
    """
    all_data = []
    suffix = "_clean" if use_clean else ""

    for sub_id in sub_ids:
        csv_path = os.path.join(data_dir, f"{sub_id}{suffix}.csv")
        if not os.path.exists(csv_path):
            if use_clean:
                print(f"[WARN] No cleaned file found for {sub_id} at {csv_path}. "
                      f"Run clean_transcripts.py first. Skipping.")
            else:
                print(f"[WARN] No file found for {sub_id} in {data_dir}. Skipping.")
            continue

        df = pd.read_csv(csv_path, on_bad_lines="skip")  # Pandas ≥1.3

        required = {
            "subject", "run", "order", "agent", "topic",
            "transcript_sub", "transcript_llm",
            "Quality", "Connectedness",
        }
        missing = required.difference(df.columns)
        if missing:
            print(f"[WARN] {csv_path} missing columns: {sorted(missing)}. Skipping.")
            continue

        # Keep rows where either transcript exists
        mask = df["transcript_sub"].notna() | df["transcript_llm"].notna()
        df = df.loc[mask, list(required)]

        # Standardize subject field
        df["subject"] = sub_id
        all_data.append(df)

    if not all_data:
        raise RuntimeError(
            "No valid subject data found. Check input paths and filenames in EXP_CSV_DIR."
        )

    combined_df = pd.concat(all_data, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    source_label = "cleaned" if use_clean else "raw"
    print(f"[INFO] Combined {source_label} transcript data ({len(combined_df)} rows) "
          f"saved to {output_path}")
    return combined_df


def main():
    # --- Handle --use_clean flag before parse_and_load_config ---
    # Strip it from sys.argv so the shared CLI helper doesn't reject it.
    use_clean = True
    cleaned_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg == "--use_clean":
            # Next arg is the value
            if i + 1 < len(sys.argv[1:]):
                val = sys.argv[i + 2]
                use_clean = val.lower() not in ("false", "0", "no")
                skip_next = True
            continue
        if arg.startswith("--use_clean="):
            val = arg.split("=", 1)[1]
            use_clean = val.lower() not in ("false", "0", "no")
            continue
        cleaned_argv.append(arg)
    sys.argv = [sys.argv[0]] + cleaned_argv

    # --- Parse CLI and load config ---
    args, cfg = parse_and_load_config("Combine transcript data across subjects")

    # get analysis info
    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    # --- Setup output directory ---
    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Initialize logging ---
    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    # --- Combine ---
    output_path = os.path.join(out_dir, "combined_text_data.csv")
    _ = combine_text_data(subjects, data_dir, output_path, use_clean=use_clean)

    source_label = "cleaned" if use_clean else "raw"
    logger.info(f"Saved combined {source_label} text data to {output_path}")


if __name__ == "__main__":
    main()