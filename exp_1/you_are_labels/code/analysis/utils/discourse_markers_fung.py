"""
DISCOURSE MARKERS (Fung & Carter 2007):
# Source: Applied Linguistics 28(3): 410-439
# Based on Table 1 (functional paradigm) and Table 2 (frequency analysis)
- fung_interpersonal_rate, fung_referential_rate, fung_structural_rate,
  fung_cognitive_rate, fung_total_rate
"""

# ============================================================
# FUNG & CARTER (2007) DISCOURSE MARKERS
# ============================================================

# INTERPERSONAL CATEGORY
FUNG_INTERPERSONAL = [
    r"\byou know\b", r"\byou see\b", r"\bsee\b", r"\blisten\b",
    r"\bwell\b", r"\breally\b", r"\bi think\b", r"\bobviously\b",
    r"\babsolutely\b", r"\bbasically\b", r"\bactually\b", r"\bexactly\b",
    r"\bsort of\b", r"\bkind of\b", r"\blike\b", r"\bjust\b", r"\boh\b",
    r"\bokay\b", r"\bok\b", r"\bright\b", r"\balright\b",
    r"\byeah\b", r"\byes\b", r"\bi see\b", r"\bgreat\b", r"\bsure\b",
]

# REFERENTIAL CATEGORY
FUNG_REFERENTIAL = [
    r"\bbecause\b", r"\bcos\b", r"\bcause\b",
    r"\bbut\b", r"\byet\b", r"\bhowever\b", r"\bnevertheless\b",
    r"\band\b", r"\bor\b", r"\bso\b",
    r"\banyway\b", r"\banyways\b",
    r"\blikewise\b", r"\bsimilarly\b",
]

# STRUCTURAL CATEGORY
FUNG_STRUCTURAL = [
    r"\bnow\b", r"\bokay\b", r"\bok\b", r"\bright\b", r"\balright\b", r"\bwell\b",
    r"\blet's start\b", r"\blet's discuss\b", r"\blet me conclude\b",
    r"\bfirst\b", r"\bfirstly\b", r"\bsecond\b", r"\bsecondly\b",
    r"\bthird\b", r"\bthirdly\b", r"\bnext\b", r"\bthen\b", r"\bfinally\b",
    r"\bso\b", r"\band what about\b", r"\bhow about\b", r"\bwhat about\b",
    r"\byeah\b", r"\band\b", r"\bcos\b",
]

# COGNITIVE CATEGORY
FUNG_COGNITIVE = [
    r"\bwell\b", r"\bi think\b", r"\bi see\b",
    r"\bi mean\b", r"\bthat is\b", r"\bin other words\b",
    r"\bwhat i mean is\b", r"\bto put it another way\b",
    r"\blike\b", r"\bsort of\b", r"\bkind of\b", r"\byou know\b",
]

# All 23 markers from Table 2
FUNG_ALL_23_MARKERS = [
    r"\band\b", r"\bso\b", r"\byeah\b", r"\bright\b", r"\bbut\b",
    r"\bor\b", r"\bjust\b", r"\bokay\b", r"\bok\b", r"\blike\b",
    r"\byou know\b", r"\bwell\b", r"\bbecause\b", r"\bnow\b", r"\byes\b",
    r"\bsort of\b", r"\bsee\b", r"\bi think\b", r"\bi mean\b",
    r"\bsay\b", r"\bactually\b", r"\boh\b", r"\breally\b", r"\bcos\b",
]

