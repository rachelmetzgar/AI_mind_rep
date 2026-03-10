"""Shared configuration for belief propagation experiment (Exp 6)."""
import os

# Paths
EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # exp_6/
CODE_DIR = os.path.join(EXP_ROOT, "code")

# Model
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
MODEL_PATH = (
    "/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)
MODEL_KEY = "llama2_13b_chat"
NUM_LAYERS = 40
HIDDEN_DIM = 5120

# Results paths (model-scoped)
RESULTS_DIR = os.path.join(EXP_ROOT, "results", MODEL_KEY)
STIMULI_PATH = os.path.join(RESULTS_DIR, "stimuli", "belief_propagation_stimuli.json")
BEHAVIORAL_DIR = os.path.join(RESULTS_DIR, "behavioral")
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
RDMS_DIR = os.path.join(RESULTS_DIR, "rdms")
RSA_DIR = os.path.join(RESULTS_DIR, "rsa")
FIGURES_DIR = os.path.join(RSA_DIR, "figures")

# Logs
LOGS_DIR = os.path.join(EXP_ROOT, "logs")

# Analysis
N_PERMUTATIONS = 10000
ALPHA = 0.05

# Topology definitions (for constructing communication RDMs)
TOPOLOGY_EDGES = {
    "chain": [(0, 1), (1, 2), (2, 3)],
    "fork": [(0, 1), (0, 2), (0, 3)],
    "diamond": [(0, 1), (0, 2), (1, 3), (2, 3)],
}

# Ensure directories exist
for d in [RESULTS_DIR, BEHAVIORAL_DIR, ACTIVATIONS_DIR, RDMS_DIR, RSA_DIR,
          FIGURES_DIR, os.path.join(RESULTS_DIR, "stimuli"), LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
