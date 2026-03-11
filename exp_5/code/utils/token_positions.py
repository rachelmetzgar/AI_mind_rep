"""
Experiment 5 — Mental State Attribution RSA
Token position utilities: identify verb, object, subject, and period token
indices for each sentence in the stimulus set.

Rachel C. Metzgar · Mar 2026
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helper ──────────────────────────────────────────────────────────────────

def _find_subsequence(haystack, needle):
    """Find first occurrence of needle list in haystack list.

    Returns:
        (start, end) indices such that haystack[start:end] == needle.

    Raises:
        ValueError if not found.
    """
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i, i + len(needle)
    raise ValueError(f"Could not find {needle} in {haystack}")


# ── Core functions ──────────────────────────────────────────────────────────

def find_token_positions(sentence, item, condition, tokenizer):
    """Identify verb, object, subject, and period token positions.

    Args:
        sentence: the raw sentence string
        item: dict from STIMULI with keys id, cat, mverb, averb, obj, ...
        condition: one of the 6 condition labels
        tokenizer: a HuggingFace tokenizer (e.g. LlamaTokenizer)

    Returns:
        dict with keys: sentence, n_tokens, verb_start, verb_end, verb_idx,
        object_start, object_end, object_idx, subject_idx, period_idx
    """
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"][0].tolist()
    n_tokens = len(input_ids)

    # Period is always the last token
    period_idx = n_tokens - 1

    # Determine which verb to use
    is_mental = condition in ("mental_state", "dis_mental", "scr_mental")
    verb_word = item["mverb"] if is_mental else item["averb"]

    # Get object word (strip "the " prefix from item["obj"])
    obj_phrase = item["obj"]  # e.g. "the crack"
    obj_word = obj_phrase.replace("the ", "", 1)  # e.g. "crack"

    # Tokenize verb and object words with space prefix to match mid-sentence
    # encoding (sentencepiece adds ▁ prefix for word-initial tokens)
    verb_ids = tokenizer.encode(" " + verb_word, add_special_tokens=False)
    obj_ids = tokenizer.encode(" " + obj_word, add_special_tokens=False)

    # Find verb subsequence in full token sequence
    verb_start, verb_end = _find_subsequence(input_ids, verb_ids)
    verb_idx = verb_end - 1  # last subword token of verb

    # Find object subsequence
    obj_start, obj_end = _find_subsequence(input_ids, obj_ids)
    object_idx = obj_end - 1  # last subword token of object

    # Subject: only present in C1 (mental_state) and C4 (action)
    if condition in ("mental_state", "action"):
        subject_idx = 1  # "He" is token 1 after BOS at position 0
    else:
        subject_idx = -1  # no subject

    return {
        "sentence": sentence,
        "n_tokens": n_tokens,
        "verb_start": verb_start,
        "verb_end": verb_end,
        "verb_idx": verb_idx,
        "object_start": obj_start,
        "object_end": obj_end,
        "object_idx": object_idx,
        "subject_idx": subject_idx,
        "period_idx": period_idx,
    }


def build_position_map(tokenizer):
    """Build position map for all 336 sentences.

    Args:
        tokenizer: a HuggingFace tokenizer

    Returns:
        list of 336 dicts, each with keys: idx, item_id, condition, category,
        sentence, n_tokens, verb_start, verb_end, verb_idx, object_start,
        object_end, object_idx, subject_idx, period_idx
    """
    from stimuli import get_all_sentences, STIMULI

    sentences = get_all_sentences()
    results = []
    for idx, (item_id, condition, category, sentence) in enumerate(sentences):
        item = STIMULI[item_id - 1]  # item_id is 1-based
        pos = find_token_positions(sentence, item, condition, tokenizer)
        results.append({
            "idx": idx,
            "item_id": item_id,
            "condition": condition,
            "category": category,
            **pos,
        })
    return results


def save_position_map(position_map, path):
    """Save position map to CSV.

    Args:
        position_map: list of dicts from build_position_map
        path: output CSV path
    """
    fields = [
        "idx", "item_id", "condition", "category", "sentence", "n_tokens",
        "verb_start", "verb_end", "verb_idx", "object_start", "object_end",
        "object_idx", "subject_idx", "period_idx",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(position_map)


def load_position_map(path):
    """Load position map from CSV.

    Args:
        path: path to CSV file written by save_position_map

    Returns:
        list of dicts with numeric fields converted to int
    """
    int_fields = [
        "idx", "item_id", "n_tokens", "verb_start", "verb_end",
        "verb_idx", "object_start", "object_end", "object_idx",
        "subject_idx", "period_idx",
    ]
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in int_fields:
                row[k] = int(row[k])
            rows.append(row)
    return rows


def validate_positions(position_map, tokenizer):
    """Validate that found verb/object tokens match expected words.

    Detokenizes the verb and object spans and compares to the expected
    verb/object from STIMULI.  Prints warnings for mismatches.

    Args:
        position_map: list of dicts from build_position_map
        tokenizer: a HuggingFace tokenizer

    Returns:
        n_warnings: number of mismatches found
    """
    from stimuli import STIMULI

    n_warnings = 0
    for entry in position_map:
        item = STIMULI[entry["item_id"] - 1]
        condition = entry["condition"]
        sentence = entry["sentence"]

        encoding = tokenizer(sentence, return_tensors="pt")
        input_ids = encoding["input_ids"][0].tolist()

        # Check verb
        is_mental = condition in ("mental_state", "dis_mental", "scr_mental")
        expected_verb = item["mverb"] if is_mental else item["averb"]
        verb_tokens = input_ids[entry["verb_start"] : entry["verb_end"]]
        decoded_verb = tokenizer.decode(verb_tokens).strip()
        if decoded_verb.lower() != expected_verb.lower():
            print(
                f"  WARN verb mismatch idx={entry['idx']}: "
                f"expected '{expected_verb}', got '{decoded_verb}' "
                f"in '{sentence}'"
            )
            n_warnings += 1

        # Check object
        obj_word = item["obj"].replace("the ", "", 1)
        obj_tokens = input_ids[entry["object_start"] : entry["object_end"]]
        decoded_obj = tokenizer.decode(obj_tokens).strip()
        if decoded_obj.lower() != obj_word.lower():
            print(
                f"  WARN obj mismatch idx={entry['idx']}: "
                f"expected '{obj_word}', got '{decoded_obj}' "
                f"in '{sentence}'"
            )
            n_warnings += 1

    print(f"Validation: {len(position_map)} entries, {n_warnings} warnings")
    return n_warnings
