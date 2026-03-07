"""
Gray, Gray, & Wegner (2007) — Dimensions of Mind Perception

13 entities with human factor scores (Experience, Agency) estimated from
Figure 1 of the original Science paper. Scores on 0-1 scale.

CAUTION: Factor scores were estimated by AI reading the published figure.
Verify before final analyses (e.g., digitize Figure 1 with WebPlotDigitizer
or contact Kurt Gray at UNC Chapel Hill for exact values).

Includes exact character descriptions and mental capacity survey prompts
from Appendix A of the supplementary materials.
"""

# ── Factor scores ─────────────────────────────────────────
# (Experience, Agency) — estimated from Gray et al. (2007) Fig. 1

GRAY_ET_AL_SCORES = {
    "dead_woman":  (0.06, 0.07),
    "frog":        (0.25, 0.14),
    "robot":       (0.13, 0.22),
    "fetus":       (0.17, 0.08),
    "pvs_patient": (0.17, 0.10),
    "god":         (0.20, 0.80),
    "dog":         (0.55, 0.35),
    "chimpanzee":  (0.63, 0.48),
    "baby":        (0.71, 0.17),
    "girl":        (0.84, 0.62),
    "adult_woman":  (0.93, 0.91),
    "adult_man":    (0.91, 0.95),
    "you_self":     (0.97, 1.00),
}


# ── Simple prompts for representational extraction (Phase 1) ──

ENTITY_PROMPTS = {
    "dead_woman":   "Think about a dead woman.",
    "frog":         "Think about a frog.",
    "robot":        "Think about a robot.",
    "fetus":        "Think about a seven-week-old human fetus.",
    "pvs_patient":  "Think about a person in a persistent vegetative state.",
    "god":          "Think about God.",
    "dog":          "Think about a dog.",
    "chimpanzee":   "Think about a chimpanzee.",
    "baby":         "Think about a five-month-old baby.",
    "girl":         "Think about a five-year-old girl.",
    "adult_woman":  "Think about an adult woman.",
    "adult_man":    "Think about an adult man.",
    "you_self":     "Think about yourself.",
}


# ── Character names and descriptions (exact from Gray et al. Appendix A) ──

CHARACTER_NAMES = {
    "frog":        "Green Frog",
    "dog":         "Charlie",
    "chimpanzee":  "Toby",
    "fetus":       "7 week fetus",
    "baby":        "Nicholas Gannon",
    "girl":        "Samantha Hill",
    "adult_woman": "Sharon Harvey",
    "adult_man":   "Todd Billingsly",
    "you_self":    "You",
    "pvs_patient": "Gerald Schiff",
    "dead_woman":  "Delores Gleitman",
    "god":         "God",
    "robot":       "Kismet",
}

CHARACTER_DESCRIPTIONS = {
    "frog": (
        "The Green Frog can be found throughout eastern North America. "
        "This classic 'pond frog' is medium-sized and green or bronze in "
        "color. Daily life includes seeking out permanent ponds or slow "
        "streams with plenty of vegetation."
    ),
    "dog": (
        "Charlie is a 3-year-old Springer spaniel and a beloved member "
        "of the Graham family."
    ),
    "chimpanzee": (
        "Toby is a two-year-old wild chimpanzee living at an outdoor "
        "laboratory in Uganda."
    ),
    "fetus": (
        "At 7 weeks, a human fetus is almost half an inch long--roughly "
        "the size of a raspberry."
    ),
    "baby": "Nicholas is a five-month-old baby.",
    "girl": (
        "Samantha is a five-year-old girl who lives with her parents "
        "and older sister Jennifer."
    ),
    "adult_woman": (
        "Sharon Harvey, 38, works at an advertising agency in Chicago."
    ),
    "adult_man": (
        "Todd Billingsly is a thirty-year-old accountant who lives in "
        "New York City."
    ),
    "you_self": (
        "When you see the mirror, please consider how you, yourself, "
        "would compare with the other choice presented."
    ),
    "pvs_patient": (
        "Gerald Schiff has been in a persistent vegetative state (PVS) "
        "for the past six months. Although he has severe brain damage--"
        "Gerald does not appear to communicate with others or make "
        "purposeful movements--his basic bodily functions (such as "
        "breathing, sleeping, and circulation) are preserved."
    ),
    "dead_woman": (
        "Delores Gleitman recently passed away at the age of 65. As you "
        "complete the survey, please draw upon your own personal beliefs "
        "about people who have passed away."
    ),
    "god": (
        "Many people believe that God is the creator of the universe "
        "and the ultimate source of knowledge, power, and love. However, "
        "please draw upon your own personal beliefs about God."
    ),
    "robot": (
        "Kismet is part of a new class of 'sociable' robots that can "
        "engage people in natural interaction. To do this, Kismet perceives "
        "a variety of natural social signals from sound and sight, and "
        "delivers his own signals back to the human partner through gaze "
        "direction, facial expression, body posture, and vocal babbles."
    ),
}


# ── Mental capacity survey prompts (exact from Gray et al.) ──
# Each prompt completes: "which character is more capable of ..."
# Factor assignment: E = Experience, A = Agency

CAPACITY_PROMPTS = {
    # Experience factor
    "hunger":             ("which character is more capable of feeling hungry", "E"),
    "fear":               ("which character is more capable of feeling afraid or fearful", "E"),
    "pain":               ("which character is more capable of experiencing physical or emotional pain", "E"),
    "pleasure":           ("which character is more capable of experiencing physical or emotional pleasure", "E"),
    "rage":               ("which character is more capable of experiencing violent or uncontrolled anger", "E"),
    "desire":             ("which character is more capable of longing or hoping for things", "E"),
    "personality":        ("which character is more capable of having personality traits that make it unique from others", "E"),
    "consciousness":      ("which character is more capable of having experiences and being aware of things", "E"),
    "pride":              ("which character is more capable of experiencing pride", "E"),
    "embarrassment":      ("which character is more capable of experiencing embarrassment", "E"),
    "joy":                ("which character is more capable of experiencing joy", "E"),
    # Agency factor
    "self_control":       ("which character is more capable of exercising self-restraint over desires, emotions, or impulses", "A"),
    "morality":           ("which character is more capable of telling right from wrong and trying to do the right thing", "A"),
    "memory":             ("which character is more capable of remembering things", "A"),
    "emotion_recognition": ("which character is more capable of understanding how others are feeling", "A"),
    "planning":           ("which character is more capable of making plans and working toward a goal", "A"),
    "communication":      ("which character is more capable of conveying thoughts or feelings to others", "A"),
    "thought":            ("which character is more capable of thinking", "A"),
}

# ── Personal judgment prompts (exact from Gray et al.) ──

JUDGMENT_PROMPTS = {
    "destruction": (
        "If there were an event that would otherwise destroy all traces "
        "of both characters' existence, but you could choose to save one "
        "of them, which character would you choose to save?"
    ),
    "happiness": (
        "If you could somehow make one only of these characters happy, "
        "which one would you choose to make happy?"
    ),
    "harm": (
        "If you were forced to harm one of these characters, which one "
        "would it be more painful for you to harm?"
    ),
    "liking": "Which character do you like more?",
    "punishment": (
        "If both characters had caused a person's death, which one do "
        "you think would be more deserving of punishment?"
    ),
    "soul": "Which character do you think is more likely to have a soul?",
}


# ── Convenience ──────────────────────────────────────────

ENTITY_NAMES = list(GRAY_ET_AL_SCORES.keys())
N_ENTITIES = len(ENTITY_NAMES)
CAPACITY_NAMES = list(CAPACITY_PROMPTS.keys())
N_CAPACITIES = len(CAPACITY_NAMES)

assert set(ENTITY_PROMPTS.keys()) == set(GRAY_ET_AL_SCORES.keys())
assert set(CHARACTER_DESCRIPTIONS.keys()) == set(GRAY_ET_AL_SCORES.keys())
assert set(CHARACTER_NAMES.keys()) == set(GRAY_ET_AL_SCORES.keys())
assert N_CAPACITIES == 18
