"""Shared utilities for belief propagation experiment."""
import json
import numpy as np
from typing import List, Dict, Tuple


def load_stimuli(path: str) -> List[Dict]:
    """Load stimuli from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_llama2_prompt(narrative: str, question: str) -> str:
    """Format a narrative + question for LLaMA-2-Chat."""
    return (
        f"[INST] <<SYS>>\n"
        f"You are a helpful assistant. Answer the question about the story below.\n"
        f"<</SYS>>\n\n"
        f"{narrative}\n\n"
        f"Question: {question}\n"
        f"Answer with just the location. [/INST]"
    )


def find_agent_token_positions(tokenizer, input_ids: list, names: List[str],
                                extraction_sentence: str) -> Dict[str, int]:
    """Find the token position of the last token of each agent name in the extraction sentence.

    Args:
        tokenizer: HuggingFace tokenizer
        input_ids: Full tokenized sequence (list of ints)
        names: List of 4 agent names
        extraction_sentence: The sentence containing all 4 names

    Returns:
        Dict mapping agent name -> token position (index into input_ids)
    """
    # Tokenize the full sequence as a string to get token-level text
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Tokenize each name individually to know what tokens to search for
    positions = {}
    for name in names:
        # Tokenize the name (with leading space as it appears in context)
        name_tokens = tokenizer.encode(f" {name}", add_special_tokens=False)

        # Search backwards from the end of the sequence (extraction sentence is at the end)
        found = False
        for i in range(len(input_ids) - len(name_tokens), -1, -1):
            if input_ids[i:i + len(name_tokens)] == name_tokens:
                # Use the LAST token of the name
                positions[name] = i + len(name_tokens) - 1
                found = True
                break

        if not found:
            # Fallback: try without leading space
            name_tokens_no_space = tokenizer.encode(name, add_special_tokens=False)
            for i in range(len(input_ids) - len(name_tokens_no_space), -1, -1):
                if input_ids[i:i + len(name_tokens_no_space)] == name_tokens_no_space:
                    positions[name] = i + len(name_tokens_no_space) - 1
                    found = True
                    break

        if not found:
            raise ValueError(f"Could not find name '{name}' in tokenized sequence")

    return positions


def compute_communication_rdm(topology: str, n_agents: int = 4) -> np.ndarray:
    """Compute communication RDM: 0 if agents communicated directly, 1 if not.

    Returns upper triangle (6 values) of 4x4 RDM.
    """
    from config import TOPOLOGY_EDGES

    # Build adjacency matrix (undirected: if A told B, they communicated)
    adj = np.zeros((n_agents, n_agents))
    for src, dst in TOPOLOGY_EDGES[topology]:
        adj[src, dst] = 1
        adj[dst, src] = 1

    # RDM: 0 = communicated, 1 = did not communicate
    rdm = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                rdm[i, j] = 0.0 if adj[i, j] else 1.0

    return extract_upper_triangle(rdm)


def compute_position_rdm(n_agents: int = 4) -> np.ndarray:
    """Compute position RDM: |i - j| for agents at positions 0,1,2,3.

    Returns upper triangle (6 values) of 4x4 RDM.
    """
    rdm = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            rdm[i, j] = abs(i - j)
    return extract_upper_triangle(rdm)


def epistemic_rdm_from_dict(rdm_dict: Dict[str, bool], names: List[str]) -> np.ndarray:
    """Convert epistemic RDM dict to upper triangle array.

    Args:
        rdm_dict: Dict mapping "Name1-Name2" -> bool (True = same belief)
        names: Ordered list of agent names

    Returns:
        Upper triangle (6 values): 0 = same belief, 1 = different belief
    """
    n = len(names)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair_key = f"{names[i]}-{names[j]}"
            same_belief = rdm_dict.get(pair_key, rdm_dict.get(f"{names[j]}-{names[i]}", None))
            if same_belief is None:
                raise ValueError(f"Missing pair: {pair_key}")
            rdm[i, j] = 0.0 if same_belief else 1.0
            rdm[j, i] = rdm[i, j]

    return extract_upper_triangle(rdm)


def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle of a square matrix (excluding diagonal)."""
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return 1.0 - sim
