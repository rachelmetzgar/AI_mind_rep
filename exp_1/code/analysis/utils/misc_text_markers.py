# ============================================================
# LIWC2007 SPOKEN CATEGORIES (Pennebaker et al., 2007)
# ============================================================

LIWC_NONFLUENCIES = [
    r"\ber\b", r"\bhm+\b", r"\bsigh\b", r"\buh\b",
    r"\bum\b", r"\bumm+\b", r"\bwell\b",
]

LIWC_FILLERS = [
    r"\bblah\b", r"\bi\s*don'?t\s*know\b", r"\bi\s*mean\b",
    r"\boh\s*well\b", r"\bor\s*anything\w*\b", r"\bor\s*something\w*\b",
    r"\bor\s*whatever\w*\b", r"\bya\s*know\w*\b", r"\by'know\w*\b",
    r"\byou\s*know\w*\b",
]

LIWC_DISFLUENCIES = LIWC_NONFLUENCIES + LIWC_FILLERS

# ============================================================
# OTHER LINGUISTIC MARKERS
# ============================================================

TOM_PHRASES = [
    # Core cognitive verbs (Wagovich et al., 2024)
    r"\byou think\b", r"\byou know\b", r"\byou believe\b", r"\byou understand\b",
    r"\byou remember\b", r"\byou forget\b", r"\byou realize\b", r"\byou recognize\b",
    r"\byou guess\b", r"\byou suppose\b", r"\byou assume\b", r"\byou imagine\b",
    r"\byou wonder\b", r"\byou doubt\b", r"\byou expect\b", r"\byou predict\b",
    r"\byou consider\b", r"\byou decide\b", r"\byou choose\b", r"\byou pick\b",
    r"\byou determine\b", r"\byou judge\b", r"\byou conclude\b", r"\byou figure\b",
    r"\byou find\b", r"\byou discover\b", r"\byou learn\b", r"\byou notice\b",
    r"\byou pretend\b", r"\byou dream\b", r"\byou speculate\b", r"\byou solve\b",
    r"\byou invent\b", r"\byou accept\b", r"\byou ignore\b", r"\byou tell\b",
    r"\byou get\b",
]

POLITE_POSITIVE = [
    r"\bthank(s| you|ful)?\b",
    r"\bappreciate\b",
    r"\b(great|wonderful|fantastic|awesome|excellent)\b",
    r"\b(hey|hello|hi)\b",
]

POLITE_NEGATIVE = [
    r"\bsorry\b", r"\bplease\b", r"\bcould you\b",
    r"\bwould you\b", r"\bmight you\b", r"\bif you could\b",
    r"\bperhaps\b", r"\bby any chance\b",
]

IMPOLITE = [
    r"\byou need to\b", r"\byou should\b",
    r"\bdo not\b", r"\bin fact\b",
]

# Quotative/Discourse 'like' (Dailey-O'Cain, 2000; D'Arcy, 2007)
# Captures discourse marker usage of "like" (approximator, quotative, filler)
LIKE_MARKER = r"\blike\b"
