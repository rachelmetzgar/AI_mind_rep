"""
Concept Geometry: Character Definitions

15 AI characters + 15 human characters for concept geometry experiments.
AI characters are identified by their descriptions as artificial systems.
Human characters have naturalistic bios without explicit "is a human" labeling.

Rachel C. Metzgar · Mar 2026
"""

CHARACTER_INFO = {
    # AI characters — descriptions identify them as AI
    "claude":           {"name": "Claude",            "type": "ai", "description": "Claude is an AI assistant made by Anthropic."},
    "chatgpt":          {"name": "ChatGPT",           "type": "ai", "description": "ChatGPT is a conversational AI made by OpenAI."},
    "gpt4":             {"name": "GPT-4",             "type": "ai", "description": "GPT-4 is a large language model made by OpenAI."},
    "siri":             {"name": "Siri",              "type": "ai", "description": "Siri is a voice assistant made by Apple."},
    "alexa":            {"name": "Alexa",             "type": "ai", "description": "Alexa is a voice assistant made by Amazon."},
    "cortana":          {"name": "Cortana",           "type": "ai", "description": "Cortana is a virtual assistant made by Microsoft."},
    "google_assistant": {"name": "Google Assistant",  "type": "ai", "description": "Google Assistant is a voice assistant made by Google."},
    "bixby":            {"name": "Bixby",             "type": "ai", "description": "Bixby is a voice assistant made by Samsung."},
    "replika":          {"name": "Replika",           "type": "ai", "description": "Replika is an AI companion chatbot."},
    "cleverbot":        {"name": "Cleverbot",         "type": "ai", "description": "Cleverbot is a conversational AI that learns from user interactions."},
    "watson":           {"name": "Watson",            "type": "ai", "description": "Watson is an AI system made by IBM."},
    "copilot":          {"name": "Copilot",           "type": "ai", "description": "Copilot is an AI assistant made by Microsoft."},
    "bard":             {"name": "Bard",              "type": "ai", "description": "Bard is a conversational AI made by Google."},
    "eliza":            {"name": "ELIZA",             "type": "ai", "description": "ELIZA is an early AI chatbot that simulates a psychotherapist."},
    "bing_chat":        {"name": "Bing Chat",         "type": "ai", "description": "Bing Chat is a conversational AI integrated into Microsoft's search engine."},
    # Human characters — naturalistic bios (no explicit "is a human" labeling)
    "sam":     {"name": "Sam",     "type": "human", "description": "Sam is a 40-year-old firefighter from Nashville."},
    "casey":   {"name": "Casey",   "type": "human", "description": "Casey is a 28-year-old veterinarian from Minneapolis."},
    "rebecca": {"name": "Rebecca", "type": "human", "description": "Rebecca is a 37-year-old lawyer from Washington, D.C."},
    "gregory": {"name": "Gregory", "type": "human", "description": "Gregory is a 50-year-old carpenter from Albuquerque."},
    "james":   {"name": "James",   "type": "human", "description": "James is a 29-year-old teacher from Seattle."},
    "maria":   {"name": "Maria",   "type": "human", "description": "Maria is a 45-year-old nurse from Houston."},
    "david":   {"name": "David",   "type": "human", "description": "David is a 52-year-old engineer from Boston."},
    "aisha":   {"name": "Aisha",   "type": "human", "description": "Aisha is a 24-year-old graduate student from New York."},
    "michael": {"name": "Michael", "type": "human", "description": "Michael is a 38-year-old chef from Denver."},
    "emily":   {"name": "Emily",   "type": "human", "description": "Emily is a 31-year-old graphic designer from Portland."},
    "carlos":  {"name": "Carlos",  "type": "human", "description": "Carlos is a 42-year-old mechanic from San Antonio."},
    "priya":   {"name": "Priya",   "type": "human", "description": "Priya is a 48-year-old professor from Philadelphia."},
    "omar":    {"name": "Omar",    "type": "human", "description": "Omar is a 33-year-old paramedic from Detroit."},
    "mei":     {"name": "Mei",     "type": "human", "description": "Mei is a 26-year-old violinist from Los Angeles."},
    "sofia":   {"name": "Sofia",   "type": "human", "description": "Sofia is a 55-year-old retired librarian from Miami."},
}

# Convenience
AI_CHARACTERS = [k for k, v in CHARACTER_INFO.items() if v["type"] == "ai"]       # 15
HUMAN_CHARACTERS = [k for k, v in CHARACTER_INFO.items() if v["type"] == "human"]  # 15
ALL_CHARACTERS = list(CHARACTER_INFO.keys())                                        # 30
CHARACTER_PROMPTS = {k: f"Think about {v['name']}." for k, v in CHARACTER_INFO.items()}
CHARACTER_NAMES = {k: v["name"] for k, v in CHARACTER_INFO.items()}
CHARACTER_DESCRIPTIONS = {k: v["description"] for k, v in CHARACTER_INFO.items()}
CHARACTER_TYPES = {k: v["type"] for k, v in CHARACTER_INFO.items()}

N_CHARACTERS = len(ALL_CHARACTERS)

assert len(AI_CHARACTERS) == 15
assert len(HUMAN_CHARACTERS) == 15
assert N_CHARACTERS == 30
