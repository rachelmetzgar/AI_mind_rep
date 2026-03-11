"""
Experiment 5 — Mental State Attribution RSA
Configuration: paths, model settings, constants.

Rachel C. Metzgar · Mar 2026
"""

from pathlib import Path
from dataclasses import dataclass, field

# ── Directories ──────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent          # exp_5/
PROJECT_ROOT = ROOT_DIR.parent                              # mind_rep/
CODE_DIR = ROOT_DIR / "code"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

# ── Model ────────────────────────────────────────────────────────────────────

VALID_MODELS = ("llama2_13b_chat", "llama3_8b_base", "llama3_8b_instruct")

_HF_CACHE = "/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"

_MODEL_PATHS = {
    "llama2_13b_chat": (
        "/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
        "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
        "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
    ),
    "llama3_8b_base": None,      # filled by _resolve_snapshot
    "llama3_8b_instruct": None,  # filled by _resolve_snapshot
}

_MODEL_HF_NAMES = {
    "llama3_8b_base": "models--meta-llama--Meta-Llama-3-8B",
    "llama3_8b_instruct": "models--meta-llama--Meta-Llama-3-8B-Instruct",
}

_MODEL_CONFIGS = {
    "llama2_13b_chat": {"hidden_dim": 5120, "n_layers": 41},
    "llama3_8b_base": {"hidden_dim": 4096, "n_layers": 33},
    "llama3_8b_instruct": {"hidden_dim": 4096, "n_layers": 33},
}


def _resolve_snapshot(model_key):
    """Resolve snapshot path for HF-cached models."""
    if model_key not in _MODEL_HF_NAMES:
        return _MODEL_PATHS[model_key]
    snap_dir = Path(_HF_CACHE) / _MODEL_HF_NAMES[model_key] / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(
            f"Model not downloaded: {snap_dir}\n"
            f"Download with: huggingface_hub.snapshot_download('{_MODEL_HF_NAMES[model_key].replace('models--', '').replace('--', '/')}')"
        )
    snaps = sorted(snap_dir.iterdir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {snap_dir}")
    return str(snaps[-1])

# ── Active model state ───────────────────────────────────────────────────────

_active_model = None


def set_model(model: str):
    global _active_model
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model {model!r}. Valid: {VALID_MODELS}")
    _active_model = model


def get_model():
    if _active_model is None:
        raise RuntimeError("Call set_model() before accessing model config.")
    return _active_model


def add_model_argument(parser):
    parser.add_argument(
        "--model", type=str, default="llama2_13b_chat",
        choices=VALID_MODELS, help="Model key"
    )
    return parser


def model_path():
    m = get_model()
    if _MODEL_PATHS.get(m) is None:
        _MODEL_PATHS[m] = _resolve_snapshot(m)
    return _MODEL_PATHS[m]


def hidden_dim():
    return _MODEL_CONFIGS[get_model()]["hidden_dim"]


def n_layers():
    return _MODEL_CONFIGS[get_model()]["n_layers"]


# ── Results helpers ──────────────────────────────────────────────────────────

def results_dir(phase: str = ""):
    """results/{model}/{phase}/"""
    d = RESULTS_DIR / get_model()
    if phase:
        d = d / phase
    return d


def data_dir(phase: str):
    """results/{model}/{phase}/data/"""
    return results_dir(phase) / "data"


def figures_dir(phase: str):
    """results/{model}/{phase}/figures/"""
    return results_dir(phase) / "figures"


def logs_dir(phase: str = ""):
    d = LOGS_DIR
    if phase:
        d = d / phase
    return d


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Experiment constants ─────────────────────────────────────────────────────

N_ITEMS = 56
N_CONDITIONS = 6
N_SENTENCES = N_ITEMS * N_CONDITIONS  # 336

CONDITION_LABELS = [
    "mental_state", "dis_mental", "scr_mental",
    "action", "dis_action", "scr_action",
]

CATEGORY_LABELS = [
    "attention", "memory", "sensation", "belief",
    "desire", "emotion", "intention",
]

ITEMS_PER_CATEGORY = 8
N_CATEGORIES = len(CATEGORY_LABELS)

N_PERMUTATIONS = 10_000

# ── Probe / intervention constants ──────────────────────────────────────────

POSITION_LABELS = ["verb", "object", "period"]
N_POSITIONS = len(POSITION_LABELS)  # 3
N_PERM_PROBES = 200       # permutation iterations for baseline probes
N_PERM_CRITICAL = 10_000  # for critical tests
N_BOOTSTRAP = 10_000      # for direction comparison

# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for m in VALID_MODELS:
        set_model(m)
        try:
            mp = model_path()
        except FileNotFoundError as e:
            mp = f"NOT DOWNLOADED ({e})"
        print(f"[{m}]")
        print(f"  Model path:  {mp}")
        print(f"  Hidden dim:  {hidden_dim()}")
        print(f"  N layers:    {n_layers()}")
        print(f"  Results dir: {results_dir('rsa')}")
    print("Config OK.")
