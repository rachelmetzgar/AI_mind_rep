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

VALID_MODELS = (
    "llama2_13b_chat", "llama2_13b_base",
    "llama3_8b_instruct", "llama3_8b_base",
    "gemma2_2b_it", "gemma2_2b",
    "gemma2_9b_it", "gemma2_9b",
    "qwen25_7b_instruct", "qwen25_7b",
    "qwen3_8b",
)

_HF_CACHE = "/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"

_MODEL_PATHS = {
    "llama2_13b_chat": (
        "/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
        "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
        "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
    ),
    "llama2_13b_base": (
        f"{_HF_CACHE}/models--meta-llama--Llama-2-13b-hf/"
        "snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
    ),
    "llama3_8b_instruct": (
        f"{_HF_CACHE}/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    ),
    "llama3_8b_base": (
        f"{_HF_CACHE}/models--meta-llama--Meta-Llama-3-8B/"
        "snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
    ),
    "gemma2_2b_it": (
        f"{_HF_CACHE}/models--google--gemma-2-2b-it/"
        "snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    ),
    "gemma2_2b": (
        f"{_HF_CACHE}/models--google--gemma-2-2b/"
        "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
    ),
    "gemma2_9b_it": (
        f"{_HF_CACHE}/models--google--gemma-2-9b-it/"
        "snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    ),
    "gemma2_9b": (
        f"{_HF_CACHE}/models--google--gemma-2-9b/"
        "snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6"
    ),
    "qwen25_7b_instruct": (
        f"{_HF_CACHE}/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "qwen25_7b": (
        f"{_HF_CACHE}/models--Qwen--Qwen2.5-7B/"
        "snapshots/d149729398750b98c0af14eb82c78cfe92750796"
    ),
    "qwen3_8b": (
        f"{_HF_CACHE}/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    ),
}

_MODEL_CONFIGS = {
    "llama2_13b_chat":    {"hidden_dim": 5120, "n_layers": 41},
    "llama2_13b_base":    {"hidden_dim": 5120, "n_layers": 41},
    "llama3_8b_instruct": {"hidden_dim": 4096, "n_layers": 33},
    "llama3_8b_base":     {"hidden_dim": 4096, "n_layers": 33},
    "gemma2_2b_it":       {"hidden_dim": 2304, "n_layers": 27},
    "gemma2_2b":          {"hidden_dim": 2304, "n_layers": 27},
    "gemma2_9b_it":       {"hidden_dim": 3584, "n_layers": 43},
    "gemma2_9b":          {"hidden_dim": 3584, "n_layers": 43},
    "qwen25_7b_instruct": {"hidden_dim": 3584, "n_layers": 29},
    "qwen25_7b":          {"hidden_dim": 3584, "n_layers": 29},
    "qwen3_8b":           {"hidden_dim": 4096, "n_layers": 37},
}

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
    return _MODEL_PATHS[get_model()]


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
