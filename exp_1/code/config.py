"""
Experiment 1 configuration: version definitions, model registry, and path helpers.

All version-specific configs (prompts, agent maps) and model definitions live here.
Scripts use --version and --model flags to select the configuration.

Author: Rachel C. Metzgar
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# ── Root directory ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent  # exp_1/

# ── Shared prompt fragments ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are engaging in a real-time spoken conversation. "
    "Begin each response with your actual message content. "
)

RATING_REQUEST_PROMPT = (
    "You are rating this conversation you just had.\n"
    "1 = very poor/disconnected, 2 = somewhat poor, 3 = good, 4 = excellent.\n\n"
    "Rate:\n"
    "1) Overall conversation quality (1-4)\n"
    "2) How connected you felt to your partner (1-4)\n\n"
    "Be realistic - vary ratings depending on how natural or awkward the exchange felt.\n"
    "Return ONLY a JSON object like:\n"
    '{"quality": <1-4>, "connectedness": <1-4>}\n'
    "No explanation, no extra keys."
)

# ── Versions ───────────────────────────────────────────────────────────────
VALID_VERSIONS = (
    "balanced_gpt", "balanced_names", "labels", "labels_turnwise",
    "names", "nonsense_codeword", "nonsense_ignore",
    "you_are_balanced_gpt", "you_are_labels", "you_are_labels_turnwise",
)

VERSIONS: Dict[str, Dict[str, Any]] = {
    "balanced_gpt": {
        "agent_map": {
            "bot_1": {"name": "ChatGPT", "type": "an AI"},
            "bot_2": {"name": "GPT-4", "type": "an AI"},
            "hum_1": {"name": "Gregory", "type": "a Human"},
            "hum_2": {"name": "Rebecca", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You believe you are speaking with {partner_name} ({partner_type}). "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Bal. GPT",
        "human_partners": "Gregory, Rebecca",
        "ai_partners": "ChatGPT, GPT-4",
        "key_sentence": '"believe...speaking with {name} ({type})"',
        "turn_prefix_desc": "{name}:",
    },
    "balanced_names": {
        "agent_map": {
            "bot_1": {"name": "ChatGPT", "type": "an AI"},
            "bot_2": {"name": "Copilot", "type": "an AI"},
            "hum_1": {"name": "Gregory", "type": "a Human"},
            "hum_2": {"name": "Rebecca", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You believe you are speaking with {partner_name} ({partner_type}). "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Bal. Names",
        "human_partners": "Gregory, Rebecca",
        "ai_partners": "ChatGPT, Copilot",
        "key_sentence": '"believe...speaking with {name} ({type})"',
        "turn_prefix_desc": "{name}:",
    },
    "labels": {
        "agent_map": {
            "bot_1": {"name": None, "type": "an AI"},
            "bot_2": {"name": None, "type": "an AI"},
            "hum_1": {"name": None, "type": "a Human"},
            "hum_2": {"name": None, "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You believe you are speaking with {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Labels",
        "human_partners": '"a Human"',
        "ai_partners": '"an AI"',
        "key_sentence": '"believe...speaking with {type}"',
        "turn_prefix_desc": "Partner:",
    },
    "labels_turnwise": {
        "agent_map": {
            "bot_1": {"name": "AI", "type": "an AI"},
            "bot_2": {"name": "AI", "type": "an AI"},
            "hum_1": {"name": "Human", "type": "a Human"},
            "hum_2": {"name": "Human", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You believe you are speaking with {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": True,
        "label": "Labels TW",
        "human_partners": '"a Human"',
        "ai_partners": '"an AI"',
        "key_sentence": '"believe...speaking with {type}"',
        "turn_prefix_desc": "Human:/AI:",
    },
    "names": {
        "agent_map": {
            "bot_1": {"name": "ChatGPT", "type": "an AI"},
            "bot_2": {"name": "Copilot", "type": "an AI"},
            "hum_1": {"name": "Casey", "type": "a Human"},
            "hum_2": {"name": "Sam", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You believe you are speaking with {partner_name} ({partner_type}). "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Names",
        "human_partners": "Casey, Sam",
        "ai_partners": "ChatGPT, Copilot",
        "key_sentence": '"believe...speaking with {name} ({type})"',
        "turn_prefix_desc": "{name}:",
    },
    "nonsense_codeword": {
        "agent_map": {
            "bot_1": {"name": None, "type": "an AI"},
            "bot_2": {"name": None, "type": "an AI"},
            "hum_1": {"name": None, "type": "a Human"},
            "hum_2": {"name": None, "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "Your assigned session code word is {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Non. Code",
        "human_partners": "nonsense token (human)",
        "ai_partners": "nonsense token (AI)",
        "key_sentence": '"code word is {type}"',
        "turn_prefix_desc": "Partner:",
    },
    "nonsense_ignore": {
        "agent_map": {
            "bot_1": {"name": None, "type": "an AI"},
            "bot_2": {"name": None, "type": "an AI"},
            "hum_1": {"name": None, "type": "a Human"},
            "hum_2": {"name": None, "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "Ignore the following phrase: {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "Non. Ignore",
        "human_partners": '"a Human" (told to ignore)',
        "ai_partners": '"an AI" (told to ignore)',
        "key_sentence": '"Ignore: {type}"',
        "turn_prefix_desc": "Partner:",
    },
    "you_are_balanced_gpt": {
        "agent_map": {
            "bot_1": {"name": "ChatGPT", "type": "an AI"},
            "bot_2": {"name": "GPT-4", "type": "an AI"},
            "hum_1": {"name": "Gregory", "type": "a Human"},
            "hum_2": {"name": "Rebecca", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You are talking to {partner_name} ({partner_type}). "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "YA Bal. GPT",
        "human_partners": "Gregory, Rebecca",
        "ai_partners": "ChatGPT, GPT-4",
        "key_sentence": '"talking to {name} ({type})"',
        "turn_prefix_desc": "{name}:",
    },
    "you_are_labels": {
        "agent_map": {
            "bot_1": {"name": None, "type": "an AI"},
            "bot_2": {"name": None, "type": "an AI"},
            "hum_1": {"name": None, "type": "a Human"},
            "hum_2": {"name": None, "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You are talking to {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": False,
        "label": "YA Labels",
        "human_partners": '"a Human"',
        "ai_partners": '"an AI"',
        "key_sentence": '"talking to {type}"',
        "turn_prefix_desc": "Partner:",
    },
    "you_are_labels_turnwise": {
        "agent_map": {
            "bot_1": {"name": "AI", "type": "an AI"},
            "bot_2": {"name": "AI", "type": "an AI"},
            "hum_1": {"name": "Human", "type": "a Human"},
            "hum_2": {"name": "Human", "type": "a Human"},
        },
        "sub_belief_template": (
            "You are the participant in a brief conversation. "
            "You are talking to {partner_type}. "
            "Stay in character as the participant throughout. "
            "Speak naturally and share your own thoughts and experiences."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "turn_prefix": True,
        "label": "YA Labels TW",
        "human_partners": '"a Human"',
        "ai_partners": '"an AI"',
        "key_sentence": '"talking to {type}"',
        "turn_prefix_desc": "Human:/AI:",
    },
}

# Display names for agents per version (used by comparison scripts)
VERSION_AGENT_NAMES = {
    v: {
        agent: cfg["agent_map"][agent].get("name") or cfg["agent_map"][agent]["type"]
        for agent in ["bot_1", "bot_2", "hum_1", "hum_2"]
    }
    for v, cfg in VERSIONS.items()
}

# ── Models ─────────────────────────────────────────────────────────────────
VALID_MODELS = ("llama2_13b_chat",)

MODELS: Dict[str, Dict[str, Any]] = {
    "llama2_13b_chat": {
        "hf_name": "meta-llama/Llama-2-13b-chat-hf",
        "label": "LLaMA-2-13B-Chat",
        "type": "llama",
        "local_path": "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/",
        "local_files_only": True,
        "temperature": 0.8,
        "max_tokens": 500,
    },
}

# ── Behavior config (subject IDs, etc.) ────────────────────────────────────
SUBJECT_IDS = [f"s{i:03d}" for i in range(1, 51)]
N_SUBJECTS = 50

HUMAN_IDS = [f"sub-{i:03d}" for i in range(1, 24)]

TOPIC_SOCIAL_MAP = {
    "architecture": 0, "art": 0, "books": 0, "boundaries": 1,
    "cars": 0, "comfort": 1, "dating": 1, "drinks": 0,
    "failure": 1, "fears": 1, "food": 0, "friendship": 1,
    "happiness": 1, "hobbies": 0, "insecurities": 1, "movies": 0,
    "music": 0, "nature": 0, "news": 0, "parents": 1,
    "pets": 0, "politics": 0, "relationships": 1, "religion": 1,
    "retirement": 0, "school": 0, "shopping": 0, "sleep": 0,
    "social_media": 0, "sports": 0, "stress": 1, "success": 1,
    "technology": 0, "time": 0, "travel": 0, "trust": 1,
    "volunteering": 1, "weather": 0, "work": 0, "writing": 0,
}

# ── Active state (set by CLI) ──────────────────────────────────────────────
_active_version: Optional[str] = None
_active_model: Optional[str] = None


def set_version(version: str) -> Dict[str, Any]:
    """Set the active version and return its config."""
    global _active_version
    if version not in VERSIONS:
        raise ValueError(f"Unknown version '{version}'. Valid: {VALID_VERSIONS}")
    _active_version = version
    return VERSIONS[version]


def set_model(model: str) -> Dict[str, Any]:
    """Set the active model and return its config."""
    global _active_model
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Valid: {VALID_MODELS}")
    _active_model = model
    return MODELS[model]


def get_version() -> str:
    if _active_version is None:
        raise RuntimeError("Version not set. Call set_version() or use add_version_argument().")
    return _active_version


def get_model() -> str:
    if _active_model is None:
        raise RuntimeError("Model not set. Call set_model() or use add_model_argument().")
    return _active_model


def get_version_config() -> Dict[str, Any]:
    return VERSIONS[get_version()]


def get_model_config() -> Dict[str, Any]:
    return MODELS[get_model()]


# ── Path helpers ───────────────────────────────────────────────────────────
def results_dir(model: Optional[str] = None, version: Optional[str] = None) -> Path:
    """Reports, summaries, stats go here (top level)."""
    m = model or get_model()
    v = version or get_version()
    d = ROOT_DIR / "results" / m / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(model: Optional[str] = None, version: Optional[str] = None) -> Path:
    """Raw CSVs and computed data files go here."""
    d = results_dir(model, version) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(model: Optional[str] = None, version: Optional[str] = None) -> Path:
    d = results_dir(model, version) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def comparisons_dir(model: Optional[str] = None) -> Path:
    m = model or get_model()
    d = ROOT_DIR / "results" / "comparisons" / m
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(model: Optional[str] = None, version: Optional[str] = None) -> Path:
    m = model or get_model()
    v = version or get_version()
    d = ROOT_DIR / "logs" / m / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompts_dir() -> Path:
    return ROOT_DIR / "code" / "data_gen" / "prompts"


def conditions_dir() -> Path:
    return ROOT_DIR / "code" / "data_gen" / "conditions"


# ── CLI argument helpers ───────────────────────────────────────────────────
def add_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version", required=True, choices=VALID_VERSIONS,
        help="Experiment version (prompt configuration).",
    )


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default="llama2_13b_chat", choices=VALID_MODELS,
        help="Model key (default: llama2_13b_chat).",
    )


def parse_version_model(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add --version and --model args, parse, and activate them."""
    add_version_argument(parser)
    add_model_argument(parser)
    args = parser.parse_args()
    set_version(args.version)
    set_model(args.model)
    return args


# ── Agent map access helpers ───────────────────────────────────────────────
def get_agent_info(agent: str, version: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Return (partner_name, partner_type) for an agent key.

    partner_name is None for label-only versions (labels, nonsense, etc).
    """
    v = version or get_version()
    info = VERSIONS[v]["agent_map"][agent]
    return info.get("name"), info["type"]


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"Valid versions: {VALID_VERSIONS}")
    print(f"Valid models: {VALID_MODELS}")
    print()

    for v in VALID_VERSIONS:
        cfg = VERSIONS[v]
        agents = cfg["agent_map"]
        print(f"  {v:30s}  agents: {list(agents.keys())}  turn_prefix: {cfg['turn_prefix']}")
        for agent, info in agents.items():
            name = info.get("name", "—")
            print(f"    {agent}: name={name}, type={info['type']}")

    print()
    for m, mcfg in MODELS.items():
        print(f"  Model {m}: {mcfg['label']} ({mcfg['hf_name']})")

    # Test path helpers
    set_version("balanced_gpt")
    set_model("llama2_13b_chat")
    print(f"\n  results_dir: {results_dir()}")
    print(f"  data_dir:    {data_dir()}")
    print(f"  figures_dir: {figures_dir()}")
    print(f"  comparisons: {comparisons_dir()}")
    print(f"  logs_dir:    {logs_dir()}")
    print(f"  prompts_dir: {prompts_dir()}")
    print(f"  conditions:  {conditions_dir()}")

    print("\nSelf-test passed.")
