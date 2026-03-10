"""
Experiment 5 — Mental State Attribution RSA
Stimulus definitions: 56 items x 6 conditions = 336 sentences.

Rachel C. Metzgar · Mar 2026
"""

from config import CONDITION_LABELS, CATEGORY_LABELS, N_ITEMS, N_CONDITIONS

STIMULI = [
    # === ATTENTION (8) ===
    {"id": 1,  "cat": "attention",  "mverb": "notices",       "averb": "fills",         "obj": "the crack",
     "mental_state": "He notices the crack.",        "action": "He fills the crack.",
     "dis_mental": "Notice the crack.",              "dis_action": "Fill the crack.",
     "scr_mental": "The crack to notice.",           "scr_action": "The crack to fill."},

    {"id": 2,  "cat": "attention",  "mverb": "observes",      "averb": "sketches",      "obj": "the bird",
     "mental_state": "He observes the bird.",        "action": "He sketches the bird.",
     "dis_mental": "Observe the bird.",              "dis_action": "Sketch the bird.",
     "scr_mental": "The bird to observe.",           "scr_action": "The bird to sketch."},

    {"id": 3,  "cat": "attention",  "mverb": "watches",       "averb": "snuffs",        "obj": "the flame",
     "mental_state": "He watches the flame.",        "action": "He snuffs the flame.",
     "dis_mental": "Watch the flame.",               "dis_action": "Snuff the flame.",
     "scr_mental": "The flame to watch.",            "scr_action": "The flame to snuff."},

    {"id": 4,  "cat": "attention",  "mverb": "sees",          "averb": "scrubs",        "obj": "the stain",
     "mental_state": "He sees the stain.",           "action": "He scrubs the stain.",
     "dis_mental": "See the stain.",                 "dis_action": "Scrub the stain.",
     "scr_mental": "The stain to see.",              "scr_action": "The stain to scrub."},

    {"id": 5,  "cat": "attention",  "mverb": "detects",       "averb": "plugs",         "obj": "the leak",
     "mental_state": "He detects the leak.",         "action": "He plugs the leak.",
     "dis_mental": "Detect the leak.",               "dis_action": "Plug the leak.",
     "scr_mental": "The leak to detect.",            "scr_action": "The leak to plug."},

    {"id": 6,  "cat": "attention",  "mverb": "examines",      "averb": "flips",         "obj": "the coin",
     "mental_state": "He examines the coin.",        "action": "He flips the coin.",
     "dis_mental": "Examine the coin.",              "dis_action": "Flip the coin.",
     "scr_mental": "The coin to examine.",           "scr_action": "The coin to flip."},

    {"id": 7,  "cat": "attention",  "mverb": "inspects",      "averb": "coils",         "obj": "the rope",
     "mental_state": "He inspects the rope.",        "action": "He coils the rope.",
     "dis_mental": "Inspect the rope.",              "dis_action": "Coil the rope.",
     "scr_mental": "The rope to inspect.",           "scr_action": "The rope to coil."},

    {"id": 8,  "cat": "attention",  "mverb": "distinguishes", "averb": "amplifies",     "obj": "the signal",
     "mental_state": "He distinguishes the signal.", "action": "He amplifies the signal.",
     "dis_mental": "Distinguish the signal.",        "dis_action": "Amplify the signal.",
     "scr_mental": "The signal to distinguish.",     "scr_action": "The signal to amplify."},

    # === MEMORY (8) ===
    {"id": 9,  "cat": "memory",     "mverb": "remembers",     "averb": "records",       "obj": "the song",
     "mental_state": "He remembers the song.",       "action": "He records the song.",
     "dis_mental": "Remember the song.",             "dis_action": "Record the song.",
     "scr_mental": "The song to remember.",          "scr_action": "The song to record."},

    {"id": 10, "cat": "memory",     "mverb": "recalls",       "averb": "prints",        "obj": "the name",
     "mental_state": "He recalls the name.",         "action": "He prints the name.",
     "dis_mental": "Recall the name.",               "dis_action": "Print the name.",
     "scr_mental": "The name to recall.",            "scr_action": "The name to print."},

    {"id": 11, "cat": "memory",     "mverb": "forgets",       "averb": "drops",         "obj": "the keys",
     "mental_state": "He forgets the keys.",         "action": "He drops the keys.",
     "dis_mental": "Forget the keys.",               "dis_action": "Drop the keys.",
     "scr_mental": "The keys to forget.",            "scr_action": "The keys to drop."},

    {"id": 12, "cat": "memory",     "mverb": "recognizes",    "averb": "photographs",   "obj": "the face",
     "mental_state": "He recognizes the face.",      "action": "He photographs the face.",
     "dis_mental": "Recognize the face.",            "dis_action": "Photograph the face.",
     "scr_mental": "The face to recognize.",         "scr_action": "The face to photograph."},

    {"id": 13, "cat": "memory",     "mverb": "misremembers",  "averb": "circles",       "obj": "the date",
     "mental_state": "He misremembers the date.",    "action": "He circles the date.",
     "dis_mental": "Misremember the date.",          "dis_action": "Circle the date.",
     "scr_mental": "The date to misremember.",       "scr_action": "The date to circle."},

    {"id": 14, "cat": "memory",     "mverb": "reminisces",    "averb": "packs",         "obj": "the trip",
     "mental_state": "He reminisces about the trip.","action": "He packs the trip.",
     "dis_mental": "Reminisce about the trip.",      "dis_action": "Pack the trip.",
     "scr_mental": "The trip to reminisce about.",   "scr_action": "The trip to pack."},

    {"id": 15, "cat": "memory",     "mverb": "retains",       "averb": "stamps",        "obj": "the fact",
     "mental_state": "He retains the fact.",         "action": "He stamps the fact.",
     "dis_mental": "Retain the fact.",               "dis_action": "Stamp the fact.",
     "scr_mental": "The fact to retain.",            "scr_action": "The fact to stamp."},

    {"id": 16, "cat": "memory",     "mverb": "recollects",    "averb": "paints",        "obj": "the scene",
     "mental_state": "He recollects the scene.",     "action": "He paints the scene.",
     "dis_mental": "Recollect the scene.",           "dis_action": "Paint the scene.",
     "scr_mental": "The scene to recollect.",        "scr_action": "The scene to paint."},

    # === SENSATION (8) ===
    {"id": 17, "cat": "sensation",  "mverb": "feels",         "averb": "blocks",        "obj": "the cold",
     "mental_state": "He feels the cold.",           "action": "He blocks the cold.",
     "dis_mental": "Feel the cold.",                 "dis_action": "Block the cold.",
     "scr_mental": "The cold to feel.",              "scr_action": "The cold to block."},

    {"id": 18, "cat": "sensation",  "mverb": "senses",        "averb": "dampens",       "obj": "the vibration",
     "mental_state": "He senses the vibration.",     "action": "He dampens the vibration.",
     "dis_mental": "Sense the vibration.",           "dis_action": "Dampen the vibration.",
     "scr_mental": "The vibration to sense.",        "scr_action": "The vibration to dampen."},

    {"id": 19, "cat": "sensation",  "mverb": "perceives",     "averb": "sweeps",        "obj": "the shadow",
     "mental_state": "He perceives the shadow.",     "action": "He sweeps the shadow.",
     "dis_mental": "Perceive the shadow.",           "dis_action": "Sweep the shadow.",
     "scr_mental": "The shadow to perceive.",        "scr_action": "The shadow to sweep."},

    {"id": 20, "cat": "sensation",  "mverb": "tastes",        "averb": "spills",        "obj": "the salt",
     "mental_state": "He tastes the salt.",          "action": "He spills the salt.",
     "dis_mental": "Taste the salt.",                "dis_action": "Spill the salt.",
     "scr_mental": "The salt to taste.",             "scr_action": "The salt to spill."},

    {"id": 21, "cat": "sensation",  "mverb": "smells",        "averb": "fans",          "obj": "the smoke",
     "mental_state": "He smells the smoke.",         "action": "He fans the smoke.",
     "dis_mental": "Smell the smoke.",               "dis_action": "Fan the smoke.",
     "scr_mental": "The smoke to smell.",            "scr_action": "The smoke to fan."},

    {"id": 22, "cat": "sensation",  "mverb": "hears",         "averb": "rings",         "obj": "the bell",
     "mental_state": "He hears the bell.",           "action": "He rings the bell.",
     "dis_mental": "Hear the bell.",                 "dis_action": "Ring the bell.",
     "scr_mental": "The bell to hear.",              "scr_action": "The bell to ring."},

    {"id": 23, "cat": "sensation",  "mverb": "touches",       "averb": "cuts",          "obj": "the fabric",
     "mental_state": "He touches the fabric.",       "action": "He cuts the fabric.",
     "dis_mental": "Touch the fabric.",              "dis_action": "Cut the fabric.",
     "scr_mental": "The fabric to touch.",           "scr_action": "The fabric to cut."},

    {"id": 24, "cat": "sensation",  "mverb": "experiences",   "averb": "vents",         "obj": "the heat",
     "mental_state": "He experiences the heat.",     "action": "He vents the heat.",
     "dis_mental": "Experience the heat.",           "dis_action": "Vent the heat.",
     "scr_mental": "The heat to experience.",        "scr_action": "The heat to vent."},

    # === BELIEF (8) ===
    {"id": 25, "cat": "belief",     "mverb": "believes",      "averb": "shreds",        "obj": "the story",
     "mental_state": "He believes the story.",       "action": "He shreds the story.",
     "dis_mental": "Believe the story.",             "dis_action": "Shred the story.",
     "scr_mental": "The story to believe.",          "scr_action": "The story to shred."},

    {"id": 26, "cat": "belief",     "mverb": "knows",         "averb": "types",         "obj": "the password",
     "mental_state": "He knows the password.",       "action": "He types the password.",
     "dis_mental": "Know the password.",             "dis_action": "Type the password.",
     "scr_mental": "The password to know.",          "scr_action": "The password to type."},

    {"id": 27, "cat": "belief",     "mverb": "assumes",       "averb": "measures",      "obj": "the risk",
     "mental_state": "He assumes the risk.",         "action": "He measures the risk.",
     "dis_mental": "Assume the risk.",               "dis_action": "Measure the risk.",
     "scr_mental": "The risk to assume.",            "scr_action": "The risk to measure."},

    {"id": 28, "cat": "belief",     "mverb": "trusts",        "averb": "bumps",         "obj": "the doctor",
     "mental_state": "He trusts the doctor.",        "action": "He bumps the doctor.",
     "dis_mental": "Trust the doctor.",              "dis_action": "Bump the doctor.",
     "scr_mental": "The doctor to trust.",           "scr_action": "The doctor to bump."},

    {"id": 29, "cat": "belief",     "mverb": "doubts",        "averb": "boxes",         "obj": "the evidence",
     "mental_state": "He doubts the evidence.",      "action": "He boxes the evidence.",
     "dis_mental": "Doubt the evidence.",            "dis_action": "Box the evidence.",
     "scr_mental": "The evidence to doubt.",         "scr_action": "The evidence to box."},

    {"id": 30, "cat": "belief",     "mverb": "thinks",        "averb": "burns",         "obj": "the plan",
     "mental_state": "He thinks about the plan.",    "action": "He burns the plan.",
     "dis_mental": "Think about the plan.",          "dis_action": "Burn the plan.",
     "scr_mental": "The plan to think about.",       "scr_action": "The plan to burn."},

    {"id": 31, "cat": "belief",     "mverb": "suspects",      "averb": "waves at",      "obj": "the neighbor",
     "mental_state": "He suspects the neighbor.",    "action": "He waves at the neighbor.",
     "dis_mental": "Suspect the neighbor.",          "dis_action": "Wave at the neighbor.",
     "scr_mental": "The neighbor to suspect.",       "scr_action": "The neighbor to wave at."},

    {"id": 32, "cat": "belief",     "mverb": "supposes",      "averb": "erases",        "obj": "the answer",
     "mental_state": "He supposes the answer.",      "action": "He erases the answer.",
     "dis_mental": "Suppose the answer.",            "dis_action": "Erase the answer.",
     "scr_mental": "The answer to suppose.",         "scr_action": "The answer to erase."},

    # === DESIRE (8) ===
    {"id": 33, "cat": "desire",     "mverb": "wants",         "averb": "shelves",       "obj": "the book",
     "mental_state": "He wants the book.",           "action": "He shelves the book.",
     "dis_mental": "Want the book.",                 "dis_action": "Shelve the book.",
     "scr_mental": "The book to want.",              "scr_action": "The book to shelve."},

    {"id": 34, "cat": "desire",     "mverb": "craves",        "averb": "scoops",        "obj": "the sugar",
     "mental_state": "He craves the sugar.",         "action": "He scoops the sugar.",
     "dis_mental": "Crave the sugar.",               "dis_action": "Scoop the sugar.",
     "scr_mental": "The sugar to crave.",            "scr_action": "The sugar to scoop."},

    {"id": 35, "cat": "desire",     "mverb": "desires",       "averb": "carries",       "obj": "the prize",
     "mental_state": "He desires the prize.",        "action": "He carries the prize.",
     "dis_mental": "Desire the prize.",              "dis_action": "Carry the prize.",
     "scr_mental": "The prize to desire.",           "scr_action": "The prize to carry."},

    {"id": 36, "cat": "desire",     "mverb": "needs",         "averb": "pours",         "obj": "the water",
     "mental_state": "He needs the water.",          "action": "He pours the water.",
     "dis_mental": "Need the water.",                "dis_action": "Pour the water.",
     "scr_mental": "The water to need.",             "scr_action": "The water to pour."},

    {"id": 37, "cat": "desire",     "mverb": "yearns for",    "averb": "dims",          "obj": "the light",
     "mental_state": "He yearns for the light.",     "action": "He dims the light.",
     "dis_mental": "Yearn for the light.",           "dis_action": "Dim the light.",
     "scr_mental": "The light to yearn for.",        "scr_action": "The light to dim."},

    {"id": 38, "cat": "desire",     "mverb": "pursues",       "averb": "chalks",        "obj": "the goal",
     "mental_state": "He pursues the goal.",         "action": "He chalks the goal.",
     "dis_mental": "Pursue the goal.",               "dis_action": "Chalk the goal.",
     "scr_mental": "The goal to pursue.",            "scr_action": "The goal to chalk."},

    {"id": 39, "cat": "desire",     "mverb": "seeks",         "averb": "buries",        "obj": "the truth",
     "mental_state": "He seeks the truth.",          "action": "He buries the truth.",
     "dis_mental": "Seek the truth.",                "dis_action": "Bury the truth.",
     "scr_mental": "The truth to seek.",             "scr_action": "The truth to bury."},

    {"id": 40, "cat": "desire",     "mverb": "prefers",       "averb": "opens",         "obj": "the window",
     "mental_state": "He prefers the window.",       "action": "He opens the window.",
     "dis_mental": "Prefer the window.",             "dis_action": "Open the window.",
     "scr_mental": "The window to prefer.",          "scr_action": "The window to open."},

    # === EMOTION (8) ===
    {"id": 41, "cat": "emotion",    "mverb": "fears",         "averb": "lights",        "obj": "the dark",
     "mental_state": "He fears the dark.",           "action": "He lights the dark.",
     "dis_mental": "Fear the dark.",                 "dis_action": "Light the dark.",
     "scr_mental": "The dark to fear.",              "scr_action": "The dark to light."},

    {"id": 42, "cat": "emotion",    "mverb": "loves",         "averb": "waters",        "obj": "the garden",
     "mental_state": "He loves the garden.",         "action": "He waters the garden.",
     "dis_mental": "Love the garden.",               "dis_action": "Water the garden.",
     "scr_mental": "The garden to love.",            "scr_action": "The garden to water."},

    {"id": 43, "cat": "emotion",    "mverb": "dreads",        "averb": "smashes",       "obj": "the alarm",
     "mental_state": "He dreads the alarm.",         "action": "He smashes the alarm.",
     "dis_mental": "Dread the alarm.",               "dis_action": "Smash the alarm.",
     "scr_mental": "The alarm to dread.",            "scr_action": "The alarm to smash."},

    {"id": 44, "cat": "emotion",    "mverb": "envies",        "averb": "shoves",        "obj": "the winner",
     "mental_state": "He envies the winner.",        "action": "He shoves the winner.",
     "dis_mental": "Envy the winner.",               "dis_action": "Shove the winner.",
     "scr_mental": "The winner to envy.",            "scr_action": "The winner to shove."},

    {"id": 45, "cat": "emotion",    "mverb": "admires",       "averb": "hangs",         "obj": "the painting",
     "mental_state": "He admires the painting.",     "action": "He hangs the painting.",
     "dis_mental": "Admire the painting.",           "dis_action": "Hang the painting.",
     "scr_mental": "The painting to admire.",        "scr_action": "The painting to hang."},

    {"id": 46, "cat": "emotion",    "mverb": "hates",         "averb": "muffles",       "obj": "the noise",
     "mental_state": "He hates the noise.",          "action": "He muffles the noise.",
     "dis_mental": "Hate the noise.",                "dis_action": "Muffle the noise.",
     "scr_mental": "The noise to hate.",             "scr_action": "The noise to muffle."},

    {"id": 47, "cat": "emotion",    "mverb": "resents",       "averb": "posts",         "obj": "the rule",
     "mental_state": "He resents the rule.",         "action": "He posts the rule.",
     "dis_mental": "Resent the rule.",               "dis_action": "Post the rule.",
     "scr_mental": "The rule to resent.",            "scr_action": "The rule to post."},

    {"id": 48, "cat": "emotion",    "mverb": "cherishes",     "averb": "folds",         "obj": "the letter",
     "mental_state": "He cherishes the letter.",     "action": "He folds the letter.",
     "dis_mental": "Cherish the letter.",            "dis_action": "Fold the letter.",
     "scr_mental": "The letter to cherish.",         "scr_action": "The letter to fold."},

    # === INTENTION (8) ===
    {"id": 49, "cat": "intention",  "mverb": "contemplates",  "averb": "staples",       "obj": "the message",
     "mental_state": "He contemplates the message.", "action": "He staples the message.",
     "dis_mental": "Contemplate the message.",       "dis_action": "Staple the message.",
     "scr_mental": "The message to contemplate.",    "scr_action": "The message to staple."},

    {"id": 50, "cat": "intention",  "mverb": "plans",         "averb": "traces",        "obj": "the route",
     "mental_state": "He plans the route.",          "action": "He traces the route.",
     "dis_mental": "Plan the route.",                "dis_action": "Trace the route.",
     "scr_mental": "The route to plan.",             "scr_action": "The route to trace."},

    {"id": 51, "cat": "intention",  "mverb": "expects",       "averb": "wraps",         "obj": "the package",
     "mental_state": "He expects the package.",      "action": "He wraps the package.",
     "dis_mental": "Expect the package.",            "dis_action": "Wrap the package.",
     "scr_mental": "The package to expect.",         "scr_action": "The package to wrap."},

    {"id": 52, "cat": "intention",  "mverb": "anticipates",   "averb": "ends",          "obj": "the call",
     "mental_state": "He anticipates the call.",     "action": "He ends the call.",
     "dis_mental": "Anticipate the call.",           "dis_action": "End the call.",
     "scr_mental": "The call to anticipate.",        "scr_action": "The call to end."},

    {"id": 53, "cat": "intention",  "mverb": "ponders",       "averb": "rolls",         "obj": "the map",
     "mental_state": "He ponders the map.",          "action": "He rolls the map.",
     "dis_mental": "Ponder the map.",                "dis_action": "Roll the map.",
     "scr_mental": "The map to ponder.",             "scr_action": "The map to roll."},

    {"id": 54, "cat": "intention",  "mverb": "decides",       "averb": "tallies",       "obj": "the outcome",
     "mental_state": "He decides the outcome.",      "action": "He tallies the outcome.",
     "dis_mental": "Decide the outcome.",            "dis_action": "Tally the outcome.",
     "scr_mental": "The outcome to decide.",         "scr_action": "The outcome to tally."},

    {"id": 55, "cat": "intention",  "mverb": "chooses",       "averb": "paves",         "obj": "the path",
     "mental_state": "He chooses the path.",         "action": "He paves the path.",
     "dis_mental": "Choose the path.",               "dis_action": "Pave the path.",
     "scr_mental": "The path to choose.",            "scr_action": "The path to pave."},

    {"id": 56, "cat": "intention",  "mverb": "considers",     "averb": "crumples",      "obj": "the offer",
     "mental_state": "He considers the offer.",      "action": "He crumples the offer.",
     "dis_mental": "Consider the offer.",            "dis_action": "Crumple the offer.",
     "scr_mental": "The offer to consider.",         "scr_action": "The offer to crumple."},
]

assert len(STIMULI) == N_ITEMS, f"Expected {N_ITEMS} items, got {len(STIMULI)}"


def get_all_sentences():
    """Return ordered list of (item_id, condition, category, sentence) tuples.

    Order: for each item, iterate through CONDITION_LABELS in order.
    Total: 336 tuples.  Index into this list matches row indices in the
    activation matrix.
    """
    rows = []
    for item in STIMULI:
        for cond in CONDITION_LABELS:
            rows.append((item["id"], cond, item["cat"], item[cond]))
    assert len(rows) == N_ITEMS * N_CONDITIONS
    return rows


def get_condition_indices(condition: str):
    """Return indices (into the 336-row list) for a given condition."""
    assert condition in CONDITION_LABELS
    offset = CONDITION_LABELS.index(condition)
    return list(range(offset, N_ITEMS * N_CONDITIONS, N_CONDITIONS))


def get_category_indices(category: str):
    """Return indices within condition 1 (0..55) for a given category."""
    assert category in CATEGORY_LABELS
    return [i for i, item in enumerate(STIMULI) if item["cat"] == category]


def get_item_groups():
    """Return list of 56 lists, each containing the 6 indices for one item."""
    groups = []
    for i in range(N_ITEMS):
        groups.append(list(range(i * N_CONDITIONS, (i + 1) * N_CONDITIONS)))
    return groups


if __name__ == "__main__":
    sentences = get_all_sentences()
    print(f"Total sentences: {len(sentences)}")
    for cond in CONDITION_LABELS:
        idx = get_condition_indices(cond)
        print(f"  {cond:15s}: {len(idx)} sentences, first idx={idx[0]}")
    for cat in CATEGORY_LABELS:
        idx = get_category_indices(cat)
        print(f"  {cat:12s}: {len(idx)} items, ids={[STIMULI[i]['id'] for i in idx]}")
    print("Stimuli OK.")
