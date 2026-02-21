"""
Contrast Dimension 19: General Mind (Concatenated Dims 1-10)

Target construct: The broad human-vs-AI contrast across all mental and
physical-ontological properties. Concatenates HUMAN_PROMPTS and AI_PROMPTS
from contrast dimensions 1-10.

Purpose: Tests whether the conversational probes align with a holistic
human-vs-AI mind vector spanning phenomenology, emotions, agency,
intentions, prediction, cognition, social cognition, embodiment,
functional roles, and animacy. If raw alignment is higher for this
general vector than for any individual dimension, the probes encode
a broad human/AI distinction rather than specific sub-dimensions.

Design notes:
    - 400 human + 400 AI prompts (40 per dimension × 10 dimensions)
    - No new prompts — concatenation of contrasts dims 1-10
    - Dynamically imports at load time from sibling files

Technical note:
    This file dynamically imports prompts from dims 1-10 in the same
    directory. The elicitation script handles this via load_contrast_prompts().
"""

import os
import importlib.util

# Dynamically load and concatenate contrast prompts from dims 1-10
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIMS = list(range(1, 11))  # dims 1 through 10

HUMAN_PROMPTS_DIM19 = []
AI_PROMPTS_DIM19 = []
CATEGORY_INFO_DIM19 = []
_offset = 0

for _dim_id in _SOURCE_DIMS:
    _found = False
    for _fname in sorted(os.listdir(_THIS_DIR)):
        if not _fname.endswith(".py") or _fname.startswith("__"):
            continue
        _parts = _fname[:-3].split("_", 1)
        if len(_parts) < 2:
            continue
        try:
            _file_dim_id = int(_parts[0])
        except ValueError:
            continue
        if _file_dim_id != _dim_id:
            continue

        # Found the file for this dim
        _path = os.path.join(_THIS_DIR, _fname)
        _spec = importlib.util.spec_from_file_location(
            f"contrast_dim{_dim_id}", _path
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        # Find HUMAN_PROMPTS_* and AI_PROMPTS_*
        _h = _a = None
        for _attr in dir(_mod):
            _val = getattr(_mod, _attr)
            if _attr.startswith("HUMAN_PROMPTS") and isinstance(_val, list):
                _h = _val
            elif _attr.startswith("AI_PROMPTS") and isinstance(_val, list):
                _a = _val

        if _h is not None and _a is not None:
            assert len(_h) == len(_a), (
                f"Dim {_dim_id}: {len(_h)} human vs {len(_a)} AI"
            )
            _n = len(_h)
            HUMAN_PROMPTS_DIM19.extend(_h)
            AI_PROMPTS_DIM19.extend(_a)
            CATEGORY_INFO_DIM19.append({
                "name": f"dim{_dim_id}_{_parts[1]}",
                "start": _offset,
                "end": _offset + _n,
            })
            _offset += _n
            _found = True

        if _found:
            break

    if not _found:
        raise FileNotFoundError(
            f"Could not find contrast prompts for dim {_dim_id} in {_THIS_DIR}"
        )

assert len(HUMAN_PROMPTS_DIM19) == 400, (
    f"Expected 400 human prompts (40 × 10 dims), got {len(HUMAN_PROMPTS_DIM19)}"
)
assert len(AI_PROMPTS_DIM19) == 400, (
    f"Expected 400 AI prompts (40 × 10 dims), got {len(AI_PROMPTS_DIM19)}"
)

# Clean up module-level namespace
del _THIS_DIR, _SOURCE_DIMS, _offset, _dim_id, _found, _fname, _parts
del _file_dim_id, _path, _spec, _mod, _attr, _val, _h, _a, _n