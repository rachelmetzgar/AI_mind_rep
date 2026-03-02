"""
The Demir (2018) hedge framework follows Hyland's (1998) six-category taxonomy:
   Demir, C. (2018). Hedging and academic writing: An analysis of lexical hedges.
   Journal of Language and Linguistic Studies, 14(4), 74-92.
   https://files.eric.ed.gov/fulltext/EJ1201945.pdf
   
   Hyland, K. (1998). Boosting, hedging, and the negotiation of academic 
    knowledge. Text, 18(3), 349-382.
 
1. MODAL AUXILIARIES: can, could, may, might, should, would
2. EPISTEMIC VERBS: 44 verbs including seem, believe, suggest, indicate, assume, etc.
3. EPISTEMIC ADVERBS: 47 adverbs including perhaps, probably, generally, somewhat, etc.
4. EPISTEMIC ADJECTIVES: 19 adjectives including possible, likely, potential, etc.
5. QUANTIFIERS/DETERMINERS: a few, some, most, much, several, to some extent, etc.
6. EPISTEMIC NOUNS: 19 nouns including assumption, belief, possibility, suggestion, etc.

Word lists taken verbatim from Demir (2018) Appendix A (pp. 90-91).

HEDGE METRICS (Demir 2018):
- demir_modal_rate: Modal auxiliary hedges / word_count
- demir_verb_rate: Epistemic verb hedges / word_count  
- demir_adverb_rate: Epistemic adverb hedges / word_count
- demir_adjective_rate: Epistemic adjective hedges / word_count
- demir_quantifier_rate: Quantifier/determiner hedges / word_count
- demir_noun_rate: Epistemic noun hedges / word_count
- demir_total_rate: All Demir taxonomy hedges / word_count

"""
# 1. MODAL AUXILIARIES (Demir 2018, Appendix A, p. 90)
DEMIR_MODALS = [
    r"\bcan\b",
    r"\bcould\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bshould\b",
    r"\bwould\b",
]

# 2. EPISTEMIC VERBS (Demir 2018, Appendix A, p. 90)
# 44 verbs listed
DEMIR_VERBS = [
    r"\badvise\b", r"\badvises\b", r"\badvised\b",
    r"\badvocate\b", r"\badvocates\b", r"\badvocated\b",
    r"\bagree with\b", r"\bagrees with\b", r"\bagreed with\b",
    r"\ballege\b", r"\balleges\b", r"\balleged\b",
    r"\banticipate\b", r"\banticipates\b", r"\banticipated\b",
    r"\bappear\b", r"\bappears\b", r"\bappeared\b",
    r"\bargue\b", r"\bargues\b", r"\bargued\b",
    r"\bassert\b", r"\basserts\b", r"\basserted\b",
    r"\bassume\b", r"\bassumes\b", r"\bassumed\b",
    r"\battempt\b", r"\battempts\b", r"\battempted\b",
    r"\bbelieve\b", r"\bbelieves\b", r"\bbelieved\b",
    r"\bcalculate\b", r"\bcalculates\b", r"\bcalculated\b",
    r"\bconjecture\b", r"\bconjectures\b", r"\bconjectured\b",
    r"\bconsider\b", r"\bconsiders\b", r"\bconsidered\b",
    r"\bcontend\b", r"\bcontends\b", r"\bcontended\b",
    r"\bcorrelate with\b", r"\bcorrelates with\b", r"\bcorrelated with\b",
    r"\bdemonstrate\b", r"\bdemonstrates\b", r"\bdemonstrated\b",
    r"\bdisplay\b", r"\bdisplays\b", r"\bdisplayed\b",
    r"\bdoubt\b", r"\bdoubts\b", r"\bdoubted\b",
    r"\bestimate\b", r"\bestimates\b", r"\bestimated\b",
    r"\bexpect\b", r"\bexpects\b", r"\bexpected\b",
    r"\bfeel\b", r"\bfeels\b", r"\bfelt\b",
    r"\bfind\b", r"\bfinds\b", r"\bfound\b",
    r"\bguess\b", r"\bguesses\b", r"\bguessed\b",
    r"\bhint\b", r"\bhints\b", r"\bhinted\b",
    r"\bhope\b", r"\bhopes\b", r"\bhoped\b",
    r"\bhypothesize\b", r"\bhypothesizes\b", r"\bhypothesized\b",
    r"\bimplicate\b", r"\bimplicates\b", r"\bimplicated\b",
    r"\bimply\b", r"\bimplies\b", r"\bimplied\b",
    r"\bindicate\b", r"\bindicates\b", r"\bindicated\b",
    r"\binsinuate\b", r"\binsinuates\b", r"\binsinuated\b",
    r"\bintend\b", r"\bintends\b", r"\bintended\b",
    r"\bintimate\b", r"\bintimates\b", r"\bintimated\b",
    r"\bmaintain\b", r"\bmaintains\b", r"\bmaintained\b",
    r"\bmention\b", r"\bmentions\b", r"\bmentioned\b",
    r"\bobserve\b", r"\bobserves\b", r"\bobserved\b",
    r"\boffer\b", r"\boffers\b", r"\boffered\b",
    r"\bopine\b", r"\bopines\b", r"\bopined\b",
    r"\bpostulate\b", r"\bpostulates\b", r"\bpostulated\b",
    r"\bpredict\b", r"\bpredicts\b", r"\bpredicted\b",
    r"\bpresume\b", r"\bpresumes\b", r"\bpresumed\b",
    r"\bprone to\b",
    r"\bpropose\b", r"\bproposes\b", r"\bproposed\b",
    r"\bproposition\b",  # listed under verbs in Demir
    r"\breckon\b", r"\breckons\b", r"\breckoned\b",
    r"\brecommend\b", r"\brecommends\b", r"\brecommended\b",
    r"\breport\b", r"\breports\b", r"\breported\b",
    r"\breveal\b", r"\breveals\b", r"\brevealed\b",
    r"\bseem\b", r"\bseems\b", r"\bseemed\b",
    r"\bshow\b", r"\bshows\b", r"\bshowed\b", r"\bshown\b",
    r"\bsignal\b", r"\bsignals\b", r"\bsignaled\b", r"\bsignalled\b",
    r"\bspeculate\b", r"\bspeculates\b", r"\bspeculated\b",
    r"\bsuggest\b", r"\bsuggests\b", r"\bsuggested\b",
    r"\bsupport\b", r"\bsupports\b", r"\bsupported\b",
    r"\bsuppose\b", r"\bsupposes\b", r"\bsupposed\b",
    r"\bsurmise\b", r"\bsurmises\b", r"\bsurmised\b",
    r"\bsuspect\b", r"\bsuspects\b", r"\bsuspected\b",
    r"\btend to\b", r"\btends to\b", r"\btended to\b",
    r"\bthink\b", r"\bthinks\b", r"\bthought\b",
    r"\btry to\b", r"\btries to\b", r"\btried to\b",
]

# 3. EPISTEMIC ADJECTIVES (Demir 2018, Appendix A, p. 90)
# 19 adjectives listed
DEMIR_ADJECTIVES = [
    r"\badvisable\b",
    r"\bapproximate\b",
    r"\bin conjunction with\b", r"\bconjunction with\b",
    r"\bconsistent with\b", r"\bin consistent with\b",
    r"\bin harmony with\b", r"\bharmony with\b",
    r"\bin line with\b",
    r"\bliable\b",
    r"\blikely\b",
    r"\bpartial\b",
    r"\bplausible\b",
    r"\bpossible\b",
    r"\bpotential\b",
    r"\bprobable\b",
    r"\bprone to\b",
    r"\breasonable\b",
    r"\breported\b",
    r"\brough\b",
    r"\bslight\b",
    r"\bsubtle\b",
    r"\bsuggested\b",
    r"\bin tune with\b",
    r"\buncertain\b",
    r"\bunlikely\b",
]

# 4. EPISTEMIC ADVERBS (Demir 2018, Appendix A, pp. 90-91)
# 47 adverbs listed
DEMIR_ADVERBS = [
    r"\babout\b",
    r"\badmittedly\b",
    r"\ball but\b",
    r"\balmost\b",
    r"\bapproximately\b",
    r"\barguably\b",
    r"\baround\b",
    r"\baveragely\b",
    r"\bfairly\b",
    r"\bfrequently\b",
    r"\bgenerally\b",
    r"\bhardly\b",
    r"\blargely\b",
    r"\blikely\b",
    r"\bmainly\b",
    r"\bmildly\b",
    r"\bmoderately\b",
    r"\bmostly\b",
    r"\bnear\b",
    r"\bnearly\b",
    r"\bnot always\b",
    r"\boccasionally\b",
    r"\boften\b",
    r"\bpartially\b",
    r"\bpartly\b",
    r"\bpassably\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bpotentially\b",
    r"\bpredictably\b",
    r"\bpresumably\b",
    r"\bprimarily\b",
    r"\bprobably\b",
    r"\bquite\b",
    r"\brarely\b",
    r"\brather\b",
    r"\breasonably\b",
    r"\brelatively\b",
    r"\broughly\b",
    r"\bscarcely\b",
    r"\bseemingly\b",
    r"\bslightly\b",
    r"\bsometimes\b",
    r"\bsomewhat\b",
    r"\bsubtly\b",
    r"\bsupposedly\b",
    r"\btolerably\b",
    r"\busually\b",
    r"\bvirtually\b",
]

# 5. QUANTIFIERS/DETERMINERS (Demir 2018, Appendix A, p. 91)
# 10 items listed
DEMIR_QUANTIFIERS = [
    r"\ba few\b",
    r"\bfew\b",
    r"\blittle\b",
    r"\ba little\b",
    r"\bmore or less\b",
    r"\bmost\b",
    r"\bmuch\b",
    r"\bnot all\b",
    r"\bon occasion\b",
    r"\bseveral\b",
    r"\bto a lesser\b",  # as in "to a lesser degree/extent"
    r"\bto a minor extent\b",
    r"\bto an extent\b",
    r"\bto some extent\b",
]

# 6. EPISTEMIC NOUNS (Demir 2018, Appendix A, p. 91)
# 19 nouns listed
DEMIR_NOUNS = [
    r"\bagreement with\b", r"\bin agreement with\b",
    r"\bassertion\b",
    r"\bassumption\b",
    r"\battempt\b",
    r"\bbelief\b",
    r"\bchance\b",
    r"\bclaim\b",
    r"\bdoubt\b",
    r"\bestimate\b",
    r"\bexpectation\b",
    r"\bguidance\b",
    r"\bhope\b",
    r"\bimplication\b",
    r"\bin accord with\b",
    r"\bintention\b",
    r"\bmajority\b",
    r"\bpossibility\b",
    r"\bpotential\b",
    r"\bprediction\b",
    r"\bpresupposition\b",
    r"\bprobability\b",
    r"\bproposal\b",
    r"\bproposition\b",
    r"\brecommendation\b",
    r"\bsuggestion\b",
    r"\btendency\b",
]

# COMPLETE DEMIR (2018) TAXONOMY - All categories combined
DEMIR_ALL_HEDGES = (
    DEMIR_MODALS + 
    DEMIR_VERBS + 
    DEMIR_ADVERBS + 
    DEMIR_ADJECTIVES + 
    DEMIR_QUANTIFIERS + 
    DEMIR_NOUNS
)
