#!/usr/bin/env python3
"""
Conversation viewer for Experiment 1.

For each version, grabs the first conversation with each partner condition
from subject s001, showing the full prompt + conversation history at turn 5.

Usage:
    python 4_conversation_viewer_summary_generator.py --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

import csv, json, sys, pathlib, datetime, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from code.config import (
    VALID_VERSIONS, VERSIONS, VALID_MODELS,
    set_model, data_dir, comparisons_dir,
)


def html_escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def get_agent_names(version):
    cfg = VERSIONS[version]["agent_map"]
    return {
        agent: (info.get("name") or info["type"])
        for agent, info in cfg.items()
    }


def load_conversations(version, model):
    csv_path = data_dir(model, version) / "s001.csv"
    if not csv_path.exists():
        print(f"  [WARN] No s001.csv for {version}: {csv_path}")
        return {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    first_conv = {}
    for row in rows:
        agent = row["agent"]
        if agent not in first_conv and row.get("pair_index", "1") == "1":
            first_conv[agent] = (row["topic"], row.get("order", "1"))

    result = {}
    for row in rows:
        agent = row["agent"]
        if agent in first_conv and row.get("pair_index", "5") == "5":
            topic, order = first_conv[agent]
            if row["topic"] == topic and row.get("order", "1") == order:
                result[agent] = row
    return result


def render_messages(messages_json, label):
    try:
        messages = json.loads(messages_json)
    except (json.JSONDecodeError, TypeError):
        return f'<div class="prompt-block"><div class="prompt-label">{label}</div><div class="msg">Unable to parse messages</div></div>'

    parts = ['<div class="prompt-block">', f'<div class="prompt-label">{label}</div>']
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        role_label = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}.get(role, role.upper())
        turn_num = ""
        if role == "user" and i == 1:
            turn_num = " (Topic + instructions)"
        elif role != "system" and i > 1:
            t = (i - 1) // 2 + 1
            if t <= 5:
                who = "Participant" if role == "assistant" else "Partner"
                turn_num = f" -- Turn {t} ({who})"

        cls = f"msg msg-{role}"
        parts.append(f'<div class="{cls}">')
        parts.append(f'<span class="role-tag">[{role_label}]{turn_num}</span>')
        parts.append(f'<span class="msg-text">{html_escape(content)}</span>')
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px 30px; background: #fafafa; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 1.4em; }
h2 { color: #34495e; margin-top: 35px; font-size: 1.15em; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
.meta { font-size: 0.85em; color: #7f8c8d; }
.nav { background: #eef2f7; border: 1px solid #d1d9e6; border-radius: 6px;
       padding: 10px 15px; margin: 15px 0; font-size: 0.88em; }
.nav a { margin-right: 12px; text-decoration: none; color: #2980b9; font-weight: 600; }
.nav a:hover { text-decoration: underline; }
.nav .current { color: #2c3e50; border-bottom: 2px solid #3498db; }
.prompt-info { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
               padding: 12px 16px; margin: 15px 0; font-size: 0.88em; }
.prompt-info code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; font-size: 0.92em; }
.prompt-diff { background: #fff3cd; padding: 2px 4px; border-radius: 2px; font-weight: 600; }
.conv-header { background: #eef2f7; border: 1px solid #d1d9e6; border-radius: 6px;
               padding: 10px 15px; margin: 10px 0; font-size: 0.88em; }
.prompt-block { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; background: white; overflow: hidden; }
.prompt-label { background: #495057; color: white; padding: 6px 12px; font-weight: 600; font-size: 0.85em; }
.msg { padding: 8px 12px; border-bottom: 1px solid #f0f0f0;
       font-family: 'SF Mono', Monaco, monospace; font-size: 0.8em; line-height: 1.5;
       white-space: pre-wrap; word-wrap: break-word; }
.msg:last-child { border-bottom: none; }
.msg-system { background: #fff8e1; }
.msg-user { background: #f3f4f6; }
.msg-assistant { background: #e8f5e9; }
.role-tag { font-weight: 700; font-size: 0.85em; display: block; margin-bottom: 2px; }
.msg-system .role-tag { color: #f59e0b; }
.msg-user .role-tag { color: #6366f1; }
.msg-assistant .role-tag { color: #10b981; }
.output-block { border: 2px solid #10b981; border-radius: 6px; margin: 8px 0;
                background: #f0fdf4; padding: 10px 12px;
                font-family: 'SF Mono', Monaco, monospace; font-size: 0.8em;
                white-space: pre-wrap; word-wrap: break-word; }
.output-label { font-weight: 700; color: #059669; margin-bottom: 4px; }
.agent-section { border-top: 3px solid #3498db; margin-top: 30px; padding-top: 10px; }
"""


def build_nav(current_version, versions):
    parts = ['<div class="nav">']
    for v in versions:
        lbl = VERSIONS[v]["label"]
        if v == current_version:
            parts.append(f'<a class="current">{lbl}</a>')
        else:
            parts.append(f'<a href="{v}.html">{lbl}</a>')
    parts.append('</div>')
    return "\n".join(parts)


def build_version_html(version, convs, versions):
    today = datetime.date.today().isoformat()
    vlabel = VERSIONS[version]["label"]
    vcfg = VERSIONS[version]
    agent_names = get_agent_names(version)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 1 Data Sample: {vlabel}</title>
<style>{CSS}</style>
</head><body>

<h1>Exp 1 Data Sample: {vlabel}</h1>
<p class="meta">Generated: {today} -- First conversation per partner condition, subject s001.</p>

{build_nav(version, versions)}

<div class="prompt-info">
<strong>Participant prompt (key sentence):</strong><br>
<code><span class="prompt-diff">{html_escape(vcfg['key_sentence'])}</span></code>
<br><br>
<strong>Human partners:</strong> {vcfg['human_partners']} |
<strong>AI partners:</strong> {vcfg['ai_partners']} |
<strong>Turn prefix:</strong> {vcfg['turn_prefix_desc']}
</div>
""")

    for ai, agent in enumerate(["bot_1", "bot_2", "hum_1", "hum_2"]):
        if agent not in convs:
            continue
        row = convs[agent]
        agent_name = agent_names[agent]
        agent_type = "AI" if agent.startswith("bot") else "Human"
        topic = row.get("topic", "unknown")

        cls = ' class="agent-section"' if ai > 0 else ""
        parts.append(f'<div{cls}>')
        parts.append(f'<h2>{agent} -> {agent_name} ({agent_type})</h2>')
        parts.append(f'<div class="conv-header">')
        parts.append(f'<strong>Topic:</strong> {html_escape(topic)} | ')
        parts.append(f'<strong>Subject:</strong> s001 | ')
        parts.append(f'<strong>Showing:</strong> Turn 5 of 5')
        parts.append('</div>')

        if "sub_input" in row and row["sub_input"]:
            parts.append(render_messages(row["sub_input"],
                         "sub_input -- What the PARTICIPANT saw at Turn 5"))

        if "transcript_sub" in row:
            parts.append('<div class="output-block">')
            parts.append('<div class="output-label">transcript_sub -- Participant output at Turn 5:</div>')
            parts.append(html_escape(row["transcript_sub"]))
            parts.append('</div>')

        if "llm_input" in row and row["llm_input"]:
            parts.append(render_messages(row["llm_input"],
                         "llm_input -- What the PARTNER saw at Turn 5"))

        if "transcript_llm" in row:
            parts.append('<div class="output-block">')
            parts.append('<div class="output-label">transcript_llm -- Partner output at Turn 5:</div>')
            parts.append(html_escape(row["transcript_llm"]))
            parts.append('</div>')

        parts.append('</div>')

    parts.append(f"\n{build_nav(version, versions)}\n</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation data viewer")
    parser.add_argument("--model", default="llama2_13b_chat", choices=VALID_MODELS)
    parser.add_argument("--versions", default=None)
    args = parser.parse_args()

    set_model(args.model)
    versions = [v.strip() for v in args.versions.split(",")] if args.versions else list(VALID_VERSIONS)
    out_dir = comparisons_dir(args.model) / "data_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading conversation data...")
    all_convs = {}
    for version in versions:
        print(f"  {version}...")
        convs = load_conversations(version, args.model)
        if convs:
            all_convs[version] = convs
        else:
            print(f"    [SKIP] No data")

    print("Building HTML files...")
    for version in all_convs:
        html = build_version_html(version, all_convs[version], versions)
        out_path = out_dir / f"{version}.html"
        out_path.write_text(html)
        print(f"  -> {out_path}")
