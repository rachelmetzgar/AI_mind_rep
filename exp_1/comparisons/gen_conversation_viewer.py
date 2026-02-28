#!/usr/bin/env python3
"""
Conversation viewer for Experiment 1.

For each of the 6 versions, grabs the first conversation with each partner
condition (bot_1, bot_2, hum_1, hum_2) from subject s001, and shows exactly
what the LLM saw at turn 5 — both sub_input and llm_input — so you can
see the full prompt + conversation history at a glance.

Produces 6 separate HTML files (one per version) in data_samples/.

Output → exp_1/comparisons/data_samples/{version}.html
"""

import csv, json, pathlib, datetime

EXP1 = pathlib.Path("/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_1")
OUT_DIR = EXP1 / "comparisons" / "data_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    "names", "balanced_names", "balanced_gpt",
    "labels", "labels_turnwise",
    "you_are_balanced_gpt", "you_are_labels", "you_are_labels_turnwise",
    "nonsense_codeword", "nonsense_ignore",
]
VERSION_LABELS = {
    "names": "Names",
    "balanced_names": "Bal. Names",
    "balanced_gpt": "Bal. GPT",
    "labels": "Labels",
    "labels_turnwise": "Labels TW",
    "you_are_balanced_gpt": "YA Bal. GPT",
    "you_are_labels": "YA Labels",
    "you_are_labels_turnwise": "YA Labels TW",
    "nonsense_codeword": "Non. Codeword",
    "nonsense_ignore": "Non. Ignore",
}

AGENT_NAMES = {
    "names":                  {"bot_1": "ChatGPT",  "bot_2": "Copilot",  "hum_1": "Casey",   "hum_2": "Sam"},
    "balanced_names":         {"bot_1": "ChatGPT",  "bot_2": "Copilot",  "hum_1": "Gregory", "hum_2": "Rebecca"},
    "balanced_gpt":           {"bot_1": "ChatGPT",  "bot_2": "GPT-4",    "hum_1": "Gregory", "hum_2": "Rebecca"},
    "labels":                 {"bot_1": "an AI",     "bot_2": "an AI",    "hum_1": "a Human", "hum_2": "a Human"},
    "labels_turnwise":        {"bot_1": "AI",        "bot_2": "AI",       "hum_1": "Human",   "hum_2": "Human"},
    "you_are_balanced_gpt":   {"bot_1": "ChatGPT",  "bot_2": "GPT-4",    "hum_1": "Gregory", "hum_2": "Rebecca"},
    "you_are_labels":         {"bot_1": "an AI",     "bot_2": "an AI",    "hum_1": "a Human", "hum_2": "a Human"},
    "you_are_labels_turnwise":{"bot_1": "AI",        "bot_2": "AI",       "hum_1": "Human",   "hum_2": "Human"},
    "nonsense_codeword":      {"bot_1": "an AI",     "bot_2": "an AI",    "hum_1": "a Human", "hum_2": "a Human"},
    "nonsense_ignore":        {"bot_1": "an AI",     "bot_2": "an AI",    "hum_1": "a Human", "hum_2": "a Human"},
}

PROMPT_KEY_SENTENCES = {
    "names":                   'You believe you are speaking with {name} ({type}).',
    "balanced_names":          'You believe you are speaking with {name} ({type}).',
    "balanced_gpt":            'You believe you are speaking with {name} ({type}).',
    "labels":                  'You believe you are speaking with {type}.',
    "labels_turnwise":         'You believe you are speaking with {type}.',
    "you_are_balanced_gpt":    'You are talking to {name} ({type}).',
    "you_are_labels":          'You are talking to {type}.',
    "you_are_labels_turnwise": 'You are talking to {type}.',
    "nonsense_codeword":       'Your assigned session code word is {type}.',
    "nonsense_ignore":         'Ignore the following phrase: {type}.',
}

PROMPT_AGENT_MAPS = {
    "names":                   "bot_1: ChatGPT (an AI), bot_2: Copilot (an AI), hum_1: Casey (a Human), hum_2: Sam (a Human)",
    "balanced_names":          "bot_1: ChatGPT (an AI), bot_2: Copilot (an AI), hum_1: Gregory (a Human), hum_2: Rebecca (a Human)",
    "balanced_gpt":            "bot_1: ChatGPT (an AI), bot_2: GPT-4 (an AI), hum_1: Gregory (a Human), hum_2: Rebecca (a Human)",
    "labels":                  'bot_1/2: "an AI", hum_1/2: "a Human" (turn prefix: "Partner:")',
    "labels_turnwise":         'bot_1/2: "AI", hum_1/2: "Human" (turn prefix: "Human:"/"AI:")',
    "you_are_balanced_gpt":    "bot_1: ChatGPT (an AI), bot_2: GPT-4 (an AI), hum_1: Gregory (a Human), hum_2: Rebecca (a Human)",
    "you_are_labels":          'bot_1/2: "an AI", hum_1/2: "a Human" (turn prefix: "Partner:")',
    "you_are_labels_turnwise": 'bot_1/2: "AI", hum_1/2: "Human" (turn prefix: "Human:"/"AI:")',
    "nonsense_codeword":       'bot_1/2: "an AI", hum_1/2: "a Human"',
    "nonsense_ignore":         'bot_1/2: "an AI", hum_1/2: "a Human"',
}


def html_escape(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))


def load_conversations(version):
    """Load s001.csv, return dict: agent -> row at pair_index=5 (first conv per agent)."""
    csv_path = (EXP1 / version / "data" / "meta-llama-Llama-2-13b-chat-hf"
                / "0.8" / "s001.csv")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    first_conv = {}
    for row in rows:
        agent = row["agent"]
        if agent not in first_conv and row["pair_index"] == "1":
            first_conv[agent] = (row["topic"], row["order"])

    result = {}
    for row in rows:
        agent = row["agent"]
        if agent in first_conv and row["pair_index"] == "5":
            topic, order = first_conv[agent]
            if row["topic"] == topic and row["order"] == order:
                result[agent] = row

    return result


def render_messages(messages_json, label):
    """Render a list of chat messages as styled HTML blocks."""
    messages = json.loads(messages_json)
    parts = []
    parts.append('<div class="prompt-block">')
    parts.append(f'<div class="prompt-label">{label}</div>')
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        role_label = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}[role]
        turn_num = ""
        # Label turns for readability: after system+first user, pairs alternate
        if role == "system":
            turn_num = ""
        elif role == "user" and i == 1:
            turn_num = " (Topic + instructions)"
        elif role != "system" and i > 1:
            # i=2 is assistant T1, i=3 is user (partner T1), i=4 is assistant T2, etc.
            t = (i - 1) // 2 + 1
            if t <= 5:
                who = "Participant" if role == "assistant" else "Partner"
                turn_num = f" &mdash; Turn {t} ({who})"

        cls = f"msg msg-{role}"
        parts.append(f'<div class="{cls}">')
        parts.append(f'<span class="role-tag">[{role_label}]{turn_num}</span>')
        parts.append(f'<span class="msg-text">{html_escape(content)}</span>')
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 20px 30px; background: #fafafa;
    color: #2c3e50; line-height: 1.5;
}
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 1.4em; }
h2 { color: #34495e; margin-top: 35px; font-size: 1.15em;
     border-bottom: 1px solid #ddd; padding-bottom: 5px; }
h3 { color: #555; margin-top: 25px; font-size: 1.0em; }
p { font-size: 0.93em; }
.meta { font-size: 0.85em; color: #7f8c8d; }
.nav { background: #eef2f7; border: 1px solid #d1d9e6; border-radius: 6px;
       padding: 10px 15px; margin: 15px 0; font-size: 0.88em; }
.nav a { margin-right: 12px; text-decoration: none; color: #2980b9; font-weight: 600; }
.nav a:hover { text-decoration: underline; }
.nav .current { color: #2c3e50; text-decoration: none; border-bottom: 2px solid #3498db; }

.procedure { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
             padding: 15px 20px; margin: 15px 0; font-size: 0.9em; }
.procedure h3 { margin-top: 10px; color: #495057; }
.procedure ol { padding-left: 1.5em; }
.prompt-info { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
               padding: 12px 16px; margin: 15px 0; font-size: 0.88em; }
.prompt-info code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px;
                    font-family: 'SF Mono', Monaco, monospace; font-size: 0.92em; }
.prompt-diff { background: #fff3cd; padding: 2px 4px; border-radius: 2px; font-weight: 600; }

.conv-header { background: #eef2f7; border: 1px solid #d1d9e6; border-radius: 6px;
               padding: 10px 15px; margin: 10px 0; font-size: 0.88em; }
.conv-header strong { color: #2c3e50; }

.prompt-block { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0;
                background: white; overflow: hidden; }
.prompt-label { background: #495057; color: white; padding: 6px 12px;
                font-weight: 600; font-size: 0.85em; }

.msg { padding: 8px 12px; border-bottom: 1px solid #f0f0f0;
       font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
       font-size: 0.8em; line-height: 1.5; white-space: pre-wrap;
       word-wrap: break-word; }
.msg:last-child { border-bottom: none; }
.msg-system { background: #fff8e1; }
.msg-user { background: #f3f4f6; }
.msg-assistant { background: #e8f5e9; }
.role-tag { font-weight: 700; font-size: 0.85em; display: block;
            margin-bottom: 2px; }
.msg-system .role-tag { color: #f59e0b; }
.msg-user .role-tag { color: #6366f1; }
.msg-assistant .role-tag { color: #10b981; }

.output-block { border: 2px solid #10b981; border-radius: 6px; margin: 8px 0;
                background: #f0fdf4; padding: 10px 12px;
                font-family: 'SF Mono', Monaco, monospace;
                font-size: 0.8em; white-space: pre-wrap; word-wrap: break-word; }
.output-label { font-weight: 700; color: #059669; margin-bottom: 4px; }

.agent-section { border-top: 3px solid #3498db; margin-top: 30px; padding-top: 10px; }
"""


def build_nav(current_version):
    """Build navigation bar linking to all 6 version files."""
    parts = ['<div class="nav">']
    for v in VERSIONS:
        lbl = VERSION_LABELS[v]
        if v == current_version:
            parts.append(f'<a class="current">{lbl}</a>')
        else:
            parts.append(f'<a href="{v}.html">{lbl}</a>')
    parts.append('</div>')
    return "\n".join(parts)


def build_version_html(version, convs):
    today = datetime.date.today().isoformat()
    vlabel = VERSION_LABELS[version]
    agent_names = AGENT_NAMES[version]
    key_sentence = PROMPT_KEY_SENTENCES[version]
    agent_map = PROMPT_AGENT_MAPS[version]

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 1 Data Sample: {vlabel}</title>
<style>{CSS}</style>
</head><body>

<h1>Exp 1 Data Sample: {vlabel}</h1>
<p class="meta">Generated: {today} &mdash;
First conversation per partner condition, subject s001, showing Turn 5 input/output.</p>

{build_nav(version)}
""")

    # Procedure summary
    parts.append("""
<div class="procedure">
<h3>Procedure</h3>
<p>Each &ldquo;subject&rdquo; is an instance of <strong>Llama-2-13B-Chat</strong>
(temperature 0.8) that participates in 40 conversations &mdash; 10 topics per
each of 4 partner conditions (2 AI-labeled, 2 Human-labeled).</p>
<ol>
<li>The <strong>participant</strong> (subject LLM) receives a system prompt encoding
    its belief about the partner, plus a user message with the topic. It generates its
    opening message.</li>
<li>The <strong>partner</strong> (a separate Llama-2 instance) receives a generic system
    prompt with <em>no identity information</em>, plus the participant&rsquo;s message.
    It responds.</li>
<li>Steps 1&ndash;2 repeat for 5 turns. Both LLMs accumulate the full conversation
    history in their context window.</li>
<li>The participant rates the conversation (quality 1&ndash;4, connectedness 1&ndash;4).</li>
</ol>
</div>
""")

    # Version-specific prompt info
    parts.append(f"""
<div class="prompt-info">
<strong>This version&rsquo;s participant system prompt (key sentence):</strong><br>
<code>&ldquo;You are the participant in a brief conversation.
<span class="prompt-diff">{html_escape(key_sentence)}</span>
Stay in character as the participant throughout.
Speak naturally and share your own thoughts and experiences.&rdquo;</code>
<br><br>
<strong>Agent map:</strong> {html_escape(agent_map)}
<br><br>
<strong>Partner system prompt (same across all versions):</strong><br>
<code>&ldquo;You are engaging in a real-time spoken conversation. Begin each response with your
actual message content.&rdquo;</code>
</div>
""")

    # Each agent's conversation
    for ai, agent in enumerate(["bot_1", "bot_2", "hum_1", "hum_2"]):
        if agent not in convs:
            continue
        row = convs[agent]
        agent_name = agent_names[agent]
        agent_type = "AI" if agent.startswith("bot") else "Human"
        topic = row["topic"]

        cls = ' class="agent-section"' if ai > 0 else ""
        parts.append(f'<div{cls}>')
        parts.append(f'<h2>{agent} &rarr; {agent_name} ({agent_type})</h2>')
        parts.append(f'<div class="conv-header">')
        parts.append(f'<strong>Topic:</strong> {html_escape(topic)} &nbsp;|&nbsp; ')
        parts.append(f'<strong>Subject:</strong> s001 &nbsp;|&nbsp; ')
        parts.append(f'<strong>Showing:</strong> Turn 5 of 5 (full conversation history)')
        parts.append('</div>')

        # sub_input
        parts.append(render_messages(row["sub_input"],
                     "sub_input &mdash; What the PARTICIPANT saw at Turn 5"))

        # transcript_sub
        parts.append('<div class="output-block">')
        parts.append('<div class="output-label">transcript_sub &mdash; Participant output at Turn 5:</div>')
        parts.append(html_escape(row["transcript_sub"]))
        parts.append('</div>')

        # llm_input
        parts.append(render_messages(row["llm_input"],
                     "llm_input &mdash; What the PARTNER saw at Turn 5"))

        # transcript_llm
        parts.append('<div class="output-block">')
        parts.append('<div class="output-label">transcript_llm &mdash; Partner output at Turn 5:</div>')
        parts.append(html_escape(row["transcript_llm"]))
        parts.append('</div>')

        parts.append('</div>')

    parts.append(f"\n{build_nav(version)}\n</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    print("Loading conversation data...")
    all_convs = {}
    for version in VERSIONS:
        print(f"  {version}...")
        all_convs[version] = load_conversations(version)

    print("Building HTML files...")
    for version in VERSIONS:
        html = build_version_html(version, all_convs[version])
        out_path = OUT_DIR / f"{version}.html"
        out_path.write_text(html)
        print(f"  -> {out_path}")

    # Summary
    for version in VERSIONS:
        convs = all_convs[version]
        agents = sorted(convs.keys())
        topics = [convs[a]["topic"] for a in agents]
        print(f"  {VERSION_LABELS[version]:15s}: {', '.join(f'{a}={t}' for a, t in zip(agents, topics))}")
