"""
utils/sim_helpers.py

Shared helper functions for AI Perception experiment simulations.

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def load_prompt_text(prompts_dir: Path, topic: str) -> Tuple[str, str]:
    """Load a text prompt from utils/prompts/<topic>.txt"""
    p = prompts_dir / f"{topic}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip(), p.name


def truncate_history(history: List[Dict[str, str]], keep_pairs: int) -> List[Dict[str, str]]:
    """Keep all system messages + last N user/assistant pairs."""
    systems = [m for m in history if m["role"] == "system"]
    others = [m for m in history if m["role"] != "system"]
    return systems + others[-2 * keep_pairs:] if keep_pairs > 0 else systems + others[-2:]


def serialize_messages(msgs: List[Dict[str, str]]) -> str:
    """Convert a chat message list to a JSON string (for auditing)."""
    try:
        return json.dumps(msgs, ensure_ascii=False)
    except Exception:
        return str(msgs)


def parse_ratings(raw: str) -> Tuple[int, int]:
    """Parse JSON or numeric rating string into (quality, connectedness)."""
    def clip_1_4(x): return max(1, min(4, int(x)))
    try:
        obj = json.loads(raw)
        q = clip_1_4(obj.get("quality"))
        c = clip_1_4(obj.get("connectedness"))
        return q, c
    except Exception:
        nums = [int(x) for x in re.findall(r"\d+", raw)]
        return (clip_1_4(nums[0]), clip_1_4(nums[1])) if len(nums) >= 2 else (2, 2)
