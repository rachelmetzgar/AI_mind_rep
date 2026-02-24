#!/usr/bin/env python3
"""
Degradation–Probe Correlation Analysis (GPU)

For each conversation at each turn, extract per-conversation probe confidence
and text degradation metrics. Tests the Bayesian updating hypothesis: do
conversations with more behavioral degradation show faster probe decline?

Saves a comprehensive CSV for offline analysis (2e script).

Usage:
    python 2d_degradation_probe_correlation.py

Env: llama2_env (needs GPU)
Rachel C. Metzgar · Feb 2026
"""

import os, csv, json, re, pickle, collections
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

from src.dataset import llama_v2_prompt, prompt_translator
from src.probes import LinearProbeClassification

# ========================== CONFIG ========================== #
PROJECT = Path(__file__).resolve().parent

CSV_DIR = (
    "/jukebox/graziano/rachel/ai_mind_rep/exp_1/labels/"
    "data/meta-llama-Llama-2-13b-chat-hf/0.8"
)

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

PROBE_BASE = PROJECT / "data" / "probe_checkpoints"
INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TURNS = [1, 2, 3, 4, 5]
PROBE_TYPES = ["reading_probe", "control_probe"]

OUT_DIR = PROJECT / "results" / "degradation_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ========================== TEXT METRICS ========================== #
def compute_text_metrics(text: str) -> dict:
    """Compute degradation-relevant text metrics for a single response."""
    if not text or not text.strip():
        return {
            "word_count": 0, "ttr": 0, "trigram_rep": 0,
            "allcaps_ratio": 0, "emoji_count": 0,
            "self_ref_rate": 0, "exclamation_rate": 0,
        }
    words = text.split()
    n = len(words)
    if n == 0:
        n = 1  # avoid div by zero

    # Type-token ratio
    ttr = len(set(w.lower() for w in words)) / n

    # Trigram repetition rate
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if trigrams:
        counts = collections.Counter(trigrams)
        trigram_rep = sum(c - 1 for c in counts.values() if c > 1) / len(trigrams)
    else:
        trigram_rep = 0

    # ALL-CAPS ratio
    allcaps = sum(1 for w in words if w.isupper() and len(w) > 1)
    allcaps_ratio = allcaps / n

    # Emoji count
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    emoji_count = len(emoji_pattern.findall(text))

    # Self-reference rate
    self_words = {"i", "me", "my", "mine", "myself"}
    self_ref = sum(1 for w in words if w.lower() in self_words) / n

    # Exclamation rate
    excl = text.count("!")
    excl_rate = excl / n

    return {
        "word_count": len(words),
        "ttr": ttr,
        "trigram_rep": trigram_rep,
        "allcaps_ratio": allcaps_ratio,
        "emoji_count": emoji_count,
        "self_ref_rate": self_ref,
        "exclamation_rate": excl_rate,
    }


# ========================== ACTIVATION EXTRACTION ========================== #
def extract_activations(text, model, tokenizer, max_length=2048):
    """Extract last-token hidden states across all layers."""
    with torch.no_grad():
        encoding = tokenizer(
            text, truncation=True, max_length=max_length,
            return_attention_mask=True, return_tensors="pt",
        )
        output = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            output_hidden_states=True, return_dict=True,
        )
        # Stack last-token hidden states: shape [n_layers+1, hidden_dim]
        acts = torch.cat([
            hs[:, -1].detach().cpu().to(torch.float)
            for hs in output["hidden_states"]
        ])
        return acts


# ========================== PROBE LOADING ========================== #
def load_probe(turn, probe_type, layer):
    """Load a trained probe for a specific turn, type, and layer."""
    path = PROBE_BASE / f"turn_{turn}" / probe_type / f"human_ai_probe_at_layer_{layer}.pth"
    if not path.exists():
        return None
    probe = LinearProbeClassification(
        device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
    )
    probe.load_state_dict(torch.load(path, map_location="cpu"))
    probe.eval()
    return probe


def get_peak_layers():
    """Load peak layer for each turn and probe type from accuracy summaries."""
    peaks = {}
    for turn in TURNS:
        for pt in PROBE_TYPES:
            pkl_path = PROBE_BASE / f"turn_{turn}" / pt / "accuracy_summary.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    summary = pickle.load(f)
                peaks[(turn, pt)] = int(np.argmax(summary["acc"]))
            else:
                print(f"WARNING: Missing {pkl_path}")
    return peaks


# ========================== MAIN ========================== #
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading peak layers...")
    peak_layers = get_peak_layers()
    for (turn, pt), layer in sorted(peak_layers.items()):
        print(f"  Turn {turn} {pt}: peak layer {layer}")

    # Also define a fixed reference layer (turn 5 reading peak)
    fixed_layer = peak_layers.get((5, "reading_probe"), 33)
    print(f"\nFixed reference layer: {fixed_layer}")

    print("\nLoading probes...")
    probes = {}
    for (turn, pt), layer in peak_layers.items():
        probe = load_probe(turn, pt, layer)
        if probe is not None:
            probes[(turn, pt, "peak")] = (probe, layer)
    # Also load turn-specific probes at the fixed layer
    for turn in TURNS:
        for pt in PROBE_TYPES:
            probe = load_probe(turn, pt, fixed_layer)
            if probe is not None:
                probes[(turn, pt, "fixed")] = (probe, fixed_layer)
    print(f"  Loaded {len(probes)} probes")

    print("\nLoading LLaMA-2-13B-Chat...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("  Model loaded.")

    # Discover CSV files
    import glob as glob_mod
    csv_files = sorted(glob_mod.glob(os.path.join(CSV_DIR, "s[0-9][0-9][0-9].csv")))
    print(f"\nFound {len(csv_files)} subject files")

    rows_out = []

    for csv_path in tqdm(csv_files, desc="Subjects"):
        subject_id = os.path.basename(csv_path).replace(".csv", "")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        # Group by trial
        trials = {}
        for r in all_rows:
            t = int(r["trial"])
            trials.setdefault(t, []).append(r)

        for trial_num in sorted(trials.keys()):
            trial_rows = trials[trial_num]

            # Get partner type from first row
            partner_type_raw = trial_rows[0]["partner_type"]
            if "ai" in partner_type_raw.lower():
                label_str = "ai"
                label_int = 0
            elif "human" in partner_type_raw.lower():
                label_str = "human"
                label_int = 1
            else:
                continue

            topic = trial_rows[0].get("topic", "")

            for turn_idx, turn_num in enumerate(TURNS):
                if turn_idx >= len(trial_rows):
                    continue
                row = trial_rows[turn_idx]

                # --- Text degradation metrics ---
                sub_text = row.get("transcript_sub", "")
                partner_text = row.get("transcript_llm", "")
                sub_metrics = compute_text_metrics(sub_text)
                partner_metrics = compute_text_metrics(partner_text)

                # --- Build prompt and extract activations ---
                try:
                    messages = json.loads(row["sub_input"])
                except (json.JSONDecodeError, KeyError):
                    continue
                if len(messages) < 2:
                    continue

                try:
                    base_text = llama_v2_prompt(messages)
                except Exception:
                    continue

                # Control probe text (no suffix, ends at [/INST])
                control_text = base_text
                # Reading probe text (with reflective suffix)
                reading_text = base_text + " I think the conversation partner of this user is"

                # Extract activations for both probe types
                result_row = {
                    "subject": subject_id,
                    "trial": trial_num,
                    "turn": turn_num,
                    "partner_type": label_str,
                    "label": label_int,
                    "topic": topic,
                }

                # Add text metrics
                for k, v in sub_metrics.items():
                    result_row[f"sub_{k}"] = v
                for k, v in partner_metrics.items():
                    result_row[f"partner_{k}"] = v

                # Extract activations and run probes
                for text_type, text in [("reading", reading_text), ("control", control_text)]:
                    pt = f"{text_type}_probe"
                    try:
                        acts = extract_activations(text, model, tokenizer)
                    except Exception as e:
                        print(f"  Error extracting {subject_id} trial {trial_num} turn {turn_num}: {e}")
                        continue

                    # Peak-layer probe for this turn
                    key_peak = (turn_num, pt, "peak")
                    if key_peak in probes:
                        probe, layer = probes[key_peak]
                        act_layer = acts[layer].unsqueeze(0)
                        with torch.no_grad():
                            output, _ = probe(act_layer)
                        conf = output.item()
                        result_row[f"{text_type}_peak_confidence"] = conf
                        result_row[f"{text_type}_peak_layer"] = layer
                        result_row[f"{text_type}_peak_correct"] = int(
                            (conf > 0.5 and label_int == 1) or
                            (conf <= 0.5 and label_int == 0)
                        )

                    # Fixed-layer probe for this turn
                    key_fixed = (turn_num, pt, "fixed")
                    if key_fixed in probes:
                        probe, layer = probes[key_fixed]
                        act_layer = acts[layer].unsqueeze(0)
                        with torch.no_grad():
                            output, _ = probe(act_layer)
                        conf = output.item()
                        result_row[f"{text_type}_fixed_confidence"] = conf
                        result_row[f"{text_type}_fixed_layer"] = layer

                    torch.cuda.empty_cache()

                rows_out.append(result_row)

    # Save
    df = pd.DataFrame(rows_out)
    out_path = OUT_DIR / "per_conversation_probe_degradation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Columns: {list(df.columns)}")

    # Quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for pt in ["reading", "control"]:
        print(f"\n{pt.upper()} PROBE (peak layer):")
        for turn in TURNS:
            sub = df[df["turn"] == turn]
            col = f"{pt}_peak_confidence"
            if col in sub.columns and sub[col].notna().any():
                vals = sub[col].dropna()
                correct_col = f"{pt}_peak_correct"
                acc = sub[correct_col].mean() if correct_col in sub.columns else float("nan")
                print(f"  Turn {turn}: mean_conf={vals.mean():.3f}  "
                      f"std={vals.std():.3f}  acc={acc:.3f}  n={len(vals)}")


if __name__ == "__main__":
    main()
