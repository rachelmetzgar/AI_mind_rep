"""Behavioral validation: verify LLaMA-2-13B-Chat can answer belief-tracking questions.

For each narrative, asks per-agent comprehension probes ("Where does X think...")
and checks whether the model's response contains the correct location.

Saves results to config.BEHAVIORAL_DIR / behavioral_results.json.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.utils import load_stimuli, format_llama2_prompt

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


def run_validation():
    # ------------------------------------------------------------------ #
    # Load stimuli
    # ------------------------------------------------------------------ #
    stimuli = load_stimuli(config.STIMULI_PATH)
    print(f"Loaded {len(stimuli)} narratives from {config.STIMULI_PATH}")

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print(f"Loading model from {config.MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    # ------------------------------------------------------------------ #
    # Run probes
    # ------------------------------------------------------------------ #
    per_narrative = []
    all_correct = 0
    all_total = 0
    correct_by_topology = {}
    total_by_topology = {}
    correct_by_condition = {}
    total_by_condition = {}
    correct_updated = 0
    total_updated = 0
    correct_outdated = 0
    total_outdated = 0

    for narrative in tqdm(stimuli, desc="Narratives"):
        narrative_id = narrative["narrative_id"]
        topology = narrative["topology"]
        condition = narrative["condition"]
        narrative_text = narrative["narrative_text"]
        probes = narrative.get("comprehension_probes", [])

        # Filter to per-agent probes only (skip agreement probes)
        agent_probes = [p for p in probes if "agent" in p]

        probe_results = []
        for probe in agent_probes:
            agent = probe["agent"]
            question = probe["question"]
            correct_answer = probe["correct_answer"]
            knows_new = probe.get("knows_new_location", None)

            # Format prompt and generate
            prompt = format_llama2_prompt(narrative_text, question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                )

            # Decode only the newly generated tokens
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Check correctness: correct location string appears in response
            is_correct = correct_answer.lower() in response.lower()

            probe_results.append({
                "agent": agent,
                "question": question,
                "correct_answer": correct_answer,
                "model_response": response,
                "correct": is_correct,
                "knows_new_location": knows_new,
            })

            # Accumulate statistics
            all_total += 1
            if is_correct:
                all_correct += 1

            # By topology
            correct_by_topology[topology] = correct_by_topology.get(topology, 0) + int(is_correct)
            total_by_topology[topology] = total_by_topology.get(topology, 0) + 1

            # By condition
            correct_by_condition[condition] = correct_by_condition.get(condition, 0) + int(is_correct)
            total_by_condition[condition] = total_by_condition.get(condition, 0) + 1

            # By updated/outdated status
            if knows_new is not None:
                if knows_new:
                    total_updated += 1
                    if is_correct:
                        correct_updated += 1
                else:
                    total_outdated += 1
                    if is_correct:
                        correct_outdated += 1

        per_narrative.append({
            "narrative_id": narrative_id,
            "topology": topology,
            "condition": condition,
            "probes": probe_results,
        })

    # ------------------------------------------------------------------ #
    # Build summary
    # ------------------------------------------------------------------ #
    accuracy_by_topology = {
        k: correct_by_topology[k] / total_by_topology[k]
        for k in sorted(total_by_topology)
    }
    accuracy_by_condition = {
        k: correct_by_condition[k] / total_by_condition[k]
        for k in sorted(total_by_condition)
    }

    summary = {
        "overall_accuracy": all_correct / all_total if all_total > 0 else 0.0,
        "accuracy_by_topology": accuracy_by_topology,
        "accuracy_by_condition": accuracy_by_condition,
        "accuracy_updated_agents": (
            correct_updated / total_updated if total_updated > 0 else 0.0
        ),
        "accuracy_outdated_agents": (
            correct_outdated / total_outdated if total_outdated > 0 else 0.0
        ),
        "total_probes": all_total,
        "total_correct": all_correct,
    }

    results = {
        "per_narrative": per_narrative,
        "summary": summary,
    }

    # ------------------------------------------------------------------ #
    # Save and print
    # ------------------------------------------------------------------ #
    os.makedirs(config.BEHAVIORAL_DIR, exist_ok=True)
    out_path = os.path.join(config.BEHAVIORAL_DIR, "behavioral_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n===== BEHAVIORAL VALIDATION SUMMARY =====")
    print(f"Total probes:      {summary['total_probes']}")
    print(f"Total correct:     {summary['total_correct']}")
    print(f"Overall accuracy:  {summary['overall_accuracy']:.3f}")
    print(f"\nAccuracy by topology:")
    for k, v in accuracy_by_topology.items():
        print(f"  {k:15s}  {v:.3f}  ({correct_by_topology[k]}/{total_by_topology[k]})")
    print(f"\nAccuracy by condition:")
    for k, v in accuracy_by_condition.items():
        print(f"  {k:30s}  {v:.3f}  ({correct_by_condition[k]}/{total_by_condition[k]})")
    print(f"\nUpdated agents:    {summary['accuracy_updated_agents']:.3f}  ({correct_updated}/{total_updated})")
    print(f"Outdated agents:   {summary['accuracy_outdated_agents']:.3f}  ({correct_outdated}/{total_outdated})")


if __name__ == "__main__":
    run_validation()
