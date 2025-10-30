"""
 Script name: word_timing.py
 Purpose: Transcribe experimental audio segments using Whisper (or MFA, placeholder),
          and output word-level timing information aligned to scanner time.

 Inputs:
   - Audio files:   {AUDIO_DIR}/{old_sub}/trials/*.wav
   - Behavior csvs: {EXP_CSV_DIR}/{old_sub}.csv

 Outputs:
   - Per-segment CSVs:
       {TRANSCRIPT_DIR}/<transcript_model>/<sub-###>/
         * sub-###_run-##_..._timing.csv
   - Subject summary:
         sub-###_all_word_timings.csv
   - Logs:
         {TRANSCRIPT_DIR}/<transcript_model>/<sub-###>/logs/{sub-###}_runlog.txt

 Usage:
   python code/behavior/word_timing.py \
     --config configs/model1_whisper_base.json \
     --sub-id sub-###
 Author: Rachel C. Metzgar
 Date: 2025-09-24
"""

from __future__ import annotations

# --- Bootstrap so it runs outside PYTHONPATH ---
import os, sys
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import argparse
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict

from utils.globals import AUDIO_DIR, EXP_CSV_DIR, get_sub_id_map
from utils.subject_utils import standardize_sub_id, find_old_id
from utils.file_parsing import parse_trial_filename, normalize_seg_path
from utils.cli_helpers import parse_cfg_and_subject
from utils.path_helpers import find_transcript_model, require_cfg_key
from utils.run_logger import init_run, log_env_specs

SCRIPT_NAME = "word_timing"

# ---------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------
def transcribe_whisper(audio_path: str, model) -> Dict[str, Any]:
    """Transcribe audio file with Whisper, return dict with 'segments' + 'text'."""
    return model.transcribe(audio_path, word_timestamps=True)

def transcribe_mfa(audio_path: str, transcript_txt_path: str, mfa_model_dir: str):
    """Placeholder for MFA forced alignment (not implemented)."""
    raise NotImplementedError("MFA transcription not implemented yet.")

# ---------------------------------------------------------------------
# Subject-level processing
# ---------------------------------------------------------------------
def process_subject(sub_id: str, old_id: str, model_func, model_args: Dict[str, Any],
                    out_root: str, logger) -> None:
    """Transcribe all audio segments for subject and write timing CSVs + summary."""
    sav_dir = os.path.join(out_root, sub_id)
    os.makedirs(sav_dir, exist_ok=True)

    behav_file = os.path.join(EXP_CSV_DIR, f"{old_id}.csv")
    if not os.path.exists(behav_file):
        logger.warning(f"Behavior file not found: {behav_file}")
        return

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df[(df['sub_segment_name'].notnull()) | (df['llm_segment_name'].notnull())]

    # Build run_scanner_times map
    run_scanner_times = {}
    for run in df['run'].unique():
        first_trial = df[df['run'] == run].iloc[0]
        run_scanner_times[run] = first_trial['run_start_time']

    num_files = sum(
        pd.notna(row[f'{seg_type}_segment_name']) and bool(str(row[f'{seg_type}_segment_name']).strip())
        for _, row in df.iterrows() for seg_type in ['sub', 'llm']
    )

    all_rows = []
    with tqdm(total=num_files, desc=f"Subject {sub_id}", ncols=80) as pbar:
        for _, row in df.iterrows():
            for seg_type in ['sub', 'llm']:
                seg_name = row.get(f'{seg_type}_segment_name', '')
                if not pd.notna(seg_name) or not str(seg_name).strip():
                    continue

                audio_path, seg_rel = normalize_seg_path(seg_name, AUDIO_DIR, old_id)
                try:
                    meta = parse_trial_filename(seg_rel)
                    segment_rel_bids = (
                        f"{meta['segment_dir']}/"
                        f"{meta['run_bids']}_{meta['trial']}_{meta['topic']}_"
                        f"{meta['agent']}_{meta['block']}_{meta['interaction']}_{meta['segment']}.wav"
                    )
                except Exception:
                    meta = None
                    segment_rel_bids = seg_rel

                if not os.path.exists(audio_path):
                    logger.warning(f"Audio not found: {audio_path}")
                    pbar.update(1)
                    continue

                try:
                    result = model_func(audio_path, **model_args)
                except Exception as e:
                    logger.warning(f"ERROR processing {seg_name}: {e}")
                    pbar.update(1)
                    continue

                out_rows = []
                base_start = row['sub_speech_start'] if seg_type == 'sub' else row['LLM_response_start']
                run_num = row['run']
                run_start_time = run_scanner_times.get(run_num, 0)

                for seg in result.get('segments', []):
                    for word in seg.get('words', []):
                        start_time_trial = base_start + word['start']
                        end_time_trial   = base_start + word['end']
                        cur_row = {
                            'subject': sub_id,
                            'run': run_num,
                            'order': row['order'],
                            'agent': row['agent'],
                            'topic': row['topic'],
                            'segment': seg_type,
                            'segment_name': segment_rel_bids,
                            'segment_name_raw': seg_rel,
                            'run_bids': meta['run_bids'] if meta else None,
                            'transcript': result.get('text', ''),
                            'word': word['word'],
                            'start_time_wav': word['start'],
                            'end_time_wav': word['end'],
                            'start_time_scanner': start_time_trial - run_start_time,
                            'end_time_scanner': end_time_trial - run_start_time,
                        }
                        out_rows.append(cur_row)
                        all_rows.append(cur_row)

                if out_rows:
                    try:
                        run_bids = meta["run_bids"]
                        out_base = (
                            f"{sub_id}_{run_bids}_{meta['trial']}_"
                            f"{meta['topic']}_{meta['agent']}_{meta['block']}_"
                            f"{meta['interaction']}_{meta['segment']}_timing.csv"
                        )
                    except Exception:
                        base = os.path.splitext(os.path.basename(seg_rel))[0]
                        out_base = f"{sub_id}_{base}_timing.csv"
                    out_file = os.path.join(sav_dir, out_base)
                    pd.DataFrame(out_rows).to_csv(out_file, index=False)
                    logger.info(f"Wrote timing: {out_file} ({len(out_rows)} rows)")
                else:
                    logger.warning(f"No words found in {seg_name}")
                pbar.update(1)

    if all_rows:
        big_file = os.path.join(sav_dir, f"{sub_id}_all_word_timings.csv")
        pd.DataFrame(all_rows).to_csv(big_file, index=False)
        logger.info(f"Wrote subject summary: {big_file} ({len(all_rows)} rows)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """CLI entrypoint for subject-level transcription."""
    args, cfg, sub_id, used_alias = parse_cfg_and_subject(
        "Transcribe subject audio with Whisper/MFA and write word timings.",
        add_overwrite=True,
        add_dry_run=True,
        add_verbose=True,
        accept_legacy_project_root=True,
    )

    # Config-driven transcript model
    transcript_model, transcript_model_dir = find_transcript_model(cfg, with_dir=True)
    out_root = transcript_model_dir
    os.makedirs(out_root, exist_ok=True)

    # Logging setup
    logger, seed, overwrite, dry_run = init_run(
        output_dir=os.path.join(out_root, sub_id),
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=used_alias,
    )
    log_env_specs(logger, extras={
        "stage": "asr_word_timing",
        "subject": sub_id,
        "transcript_model": transcript_model,
    })

    # Map IDs
    sub_id_std = standardize_sub_id(sub_id)
    old_id = find_old_id(sub_id_std)
    if old_id is None:
        logger.warning(f"No mapping found for subject {sub_id}")
        return

    # Load backend model
    model_type = require_cfg_key(cfg, "transcript_model")
    if model_type.startswith("whisper"):
        import whisper
        parts = model_type.split("_", 1) # accept "whisper", "whisper_base", "whisper-large-v3", etc.
        if len(parts) == 2 and parts[1]:
            whisper_variant = parts[1]
        else: # no variant provided and no whisper_model in config → hard error
            raise ValueError(f"Config must specify whisper variant explicitly: got transcript_model='{model_type}'")
        model_instance = whisper.load_model(whisper_variant)
        model_func = transcribe_whisper
        model_args = {"model": model_instance}
    elif model_type == "mfa":
        model_func = transcribe_mfa
        model_args = {"mfa_model_dir": cfg.get("mfa_model_dir", "/path/to/mfa")}
    else:
        raise ValueError(f"Unknown transcript_model: {model_type}")

    logger.info(f"Processing subject: {sub_id_std} (old_id={old_id})")

    if dry_run:
        logger.info("DRY-RUN: would process subject, no files written.")
        return

    process_subject(sub_id_std, old_id, model_func, model_args, out_root, logger)
    logger.info("Finished subject processing.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
