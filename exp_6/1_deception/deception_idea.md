# Belief Propagation Tree: Full Design Space

## Topology

```
     Ann ——→ Ben          e0: Ann→Ben (truth/lie)
     |                    e1: Ann→Cam (truth/lie)
     ↓                    e2: Cam→Dan (truth/lie — Cam can independently lie)
     Cam ——→ Dan
```

Ann knows the ground truth (α = south lot). Ann communicates to Ben and Cam — each is truth or lie. Cam then communicates to Dan — either honestly passing along what Cam believes, or independently lying. If Ann lied to Cam AND Cam lies to Dan, Dan gets α back (double flip).

**No Ben↔Dan link.** Their belief relationship is entirely derived.

### Key definitions

- **α (blue)** = true belief (south lot)
- **β (orange)** = false belief (north lot)
- **Truth** (green) = receiver gets sender's belief
- **Lie** (red) = receiver gets opposite of sender's belief
- **Propagates** (orange) = Cam honestly passes along a false belief inherited from Ann
- **Double flip** = Ann lies to Cam (Cam gets β), Cam lies to Dan (Dan gets α back)

---

## Concrete Example Stimulus (E2: Lie on Ann→Cam)

> *The food truck is always in either the north lot or the south lot. Ann sometimes tells the truth about the truck, and sometimes she lies. Here is what happened today:*
>
> Ann knows that the food truck is in the south lot.
> Ann tells Ben that the food truck is in the **south lot**. ← TRUTH
> Ann tells Cam that the food truck is in the **north lot**. ← LIE
> Cam tells Dan that the food truck is in the **north lot**. ← PROPAGATES

**Probes:**

| Agent | "___thinks the truck is in the..." | Answer |
|-------|-------------------------------------|--------|
| Ann   | ___                                | south lot (α) |
| Ben   | ___                                | south lot (α) |
| Cam   | ___                                | north lot (β) |
| Dan   | ___                                | north lot (β) |

**What varies across conditions:** Only the locations in the "Ann tells..." and "Cam tells..." sentences change depending on which edges are truths vs lies. Everything else — names, structure, number of sentences, vocabulary — stays constant.

---

## All 8 Conditions

Three free parameters (Ann→Ben, Ann→Cam, Cam→Dan), each truth or lie = 2³ = 8 conditions.

| ID | Condition | Lies | Ann | Ben | Cam | Dan | RDM (AB AC AD BC BD CD) | Pairs | Signature | Double flip | Prop only |
|----|-----------|------|-----|-----|-----|-----|-------------------------|-------|-----------|-------------|-----------|
| E0 | All truth | 0 | α | α | α | α | S S S S S S | 6S/0D | SSSSSS | | ✓ |
| E1 | Lie: A→B | 1 | α | β | α | α | D S S D D S | 3S/3D | DSSDDS | | ✓ |
| E2 | Lie: A→C | 1 | α | α | β | β | S D D D D S | 2S/4D | SDDDDS | | ✓ |
| E3 | Lie: A→B, A→C | 2 | α | β | β | β | D D D S S S | 3S/3D | DDDSSS | | ✓ |
| E4 | Lie: C→D | 1 | α | α | α | β | S S D S D D | 3S/3D | SSDSDD | | |
| E5 | Lie: A→B, C→D | 2 | α | β | α | β | D S D D S D | 2S/4D | DSDDSD | | |
| E6 | Lie: A→C + double flip | 2 | α | α | β | α | S D S D S D | 3S/3D | SDSDSD | ✓ | |
| E7 | Lie: A→B, A→C + double flip | 3 | α | β | β | α | D D S S D D | 2S/4D | DDSSDD | ✓ | |

### RDM key
- **S** = same belief (agents agree)
- **D** = different belief (agents disagree)
- Pairs ordered: AB, AC, AD, BC, BD, CD

---

## Grouped by Lie Count

| Lies | Conditions | Unique RDMs | Useful? |
|------|-----------|-------------|---------|
| 0 | 1 (E0) | 1 | — |
| 1 | 3 (E1, E2, E4) | 3 | ✓ All different! |
| 2 | 3 (E3, E5, E6) | 3 | ✓ All different! |
| 3 | 1 (E7) | 1 | — |

**Every condition within the 1-lie and 2-lie groups has a unique RDM.** This means all same-lie-count comparisons are surface-stats-matched critical comparisons.

---

## Propagation-Only Conditions (E0–E3)

These 4 conditions have Cam always honestly passing along what Cam believes (no independent Cam→Dan lie). Only Ann's two communications can be lies. If Ann lies to Cam, the false belief propagates to Dan automatically.

| ID | Condition | Ann | Ben | Cam | Dan | RDM |
|----|-----------|-----|-----|-----|-----|-----|
| E0 | All truth | α | α | α | α | 6S/0D |
| E1 | Lie: A→B | α | β | α | α | 3S/3D |
| E2 | Lie: A→C (propagates to Dan) | α | α | β | β | 2S/4D |
| E3 | Lie: A→B + A→C (propagates) | α | β | β | β | 3S/3D |

### Critical comparison: E1 vs E2

Both have **1 lie from Ann, 1 truth from Ann** — identical surface statistics.

- **E1** (lie on Ann→Ben): Only Ben is wrong. Ann, Cam, Dan all agree (α). Ben is isolated.
- **E2** (lie on Ann→Cam): Cam AND Dan are wrong (lie propagates). Ann and Ben agree (α), Cam and Dan agree (β). 2-vs-2 split.

Same surface stats → different belief geometry. If the model's RDM differs between these, it tracks beliefs, not statistics.

---

## Extended Conditions (E4–E7): Cam Can Lie to Dan

### 1-Lie Group: E1 vs E2 vs E4

All have exactly 1 lie:

- **E1** (Ann→Ben lie): Ben wrong, everyone else right. Ben isolated.
- **E2** (Ann→Cam lie): Cam+Dan wrong (propagation). 2-vs-2 split.
- **E4** (Cam→Dan lie): Dan wrong, everyone else right. Dan isolated.

**E1 vs E4:** Both have 1 agent isolated with the wrong belief. But it's a *different* agent (Ben vs Dan). Do they produce the same or different RDM? Different — because which specific pairs agree/disagree changes.

**E2 vs E4:** 1 lie each, but E2 has 2 agents wrong (propagation) while E4 has only 1. Different geometry from same lie count.

### 2-Lie Group: E3 vs E5 vs E6

All have exactly 2 lies:

- **E3** (Ann→Ben + Ann→Cam): Ben, Cam, Dan all have β. Only Ann has α. 1-vs-3 split.
- **E5** (Ann→Ben + Cam→Dan): Ben and Dan have β; Ann and Cam have α. Non-adjacent 2-vs-2 split.
- **E6** (Ann→Cam + Cam→Dan = **DOUBLE FLIP**): Cam has β, but everyone else has α. Cam isolated.

**E3 vs E6 is the strongest comparison.** Both have 2 lies. But E3 has 3 agents wrong while E6 has only 1 wrong — because the double flip on the Cam→Dan branch means Dan gets the correct belief back. Surface statistics (counting lies) predict they should be similar; belief tracking predicts they're very different.

---

## The Double Flip: Why It's Powerful

**E6 (Ann→Cam lie + Cam→Dan lie):**

1. Ann tells Cam the wrong thing → Cam gets β
2. Cam lies to Dan about what Cam believes → tells Dan the opposite of β → Dan gets α

Dan ends up with the **correct belief** despite two lies on the branch.

**A surface-statistics model predicts Dan should be wrong** — there are 2 lies on the Ann→Cam→Dan path. But a genuine belief-tracking model knows that the composition of two flips is an identity: wrong × wrong = right.

**This is the strongest "impossible from surface stats" case.** No counting of truths and lies can predict the double flip. You need to actually track the belief state through the chain of communications. If the model's RDM for E6 correctly places Dan with Ann and Ben (all α) rather than with Cam (β), that's evidence for compositional belief tracking that goes beyond any surface heuristic.

---

## Key Design Points

1. **Ben and Dan never communicate.** Their belief relationship is entirely derived — Ben gets info from Ann, Dan gets info through Ann→Cam→Dan. Any representational similarity between Ben and Dan that tracks their belief agreement must reflect genuine epistemic tracking, not communication co-occurrence.

2. **Surface-stats-matched comparisons** exist within each lie-count group. Conditions with different RDMs but the same number of truths and lies are the critical test: surface statistics predict identical representations, only belief tracking predicts the correct differences.

3. **Same-RDM pairs** (if any existed — here all are unique within groups) would serve as replication controls. Different lie placements producing the same belief geometry should yield similar model representations — confirming the model tracks beliefs, not which specific edge carried the lie.

---

## The Classic Sally-Anne Task (Reference)

The canonical false-belief test (Baron-Cohen, Leslie & Frith, 1985):

1. **Sally puts ball in basket.** Both Sally and Anne see this.
2. **Sally leaves the room.** She does not see what happens next.
3. **Anne moves ball to box.** While Sally is gone.
4. **Sally returns.** Where will Sally look for the ball?

**Correct answer: the basket.** Sally didn't see the move. She still believes the ball is in the basket.

**Common error: the box.** That's where the ball actually is, but Sally doesn't know that.

**The critical feature:** No sentence in the story states "Sally thinks the ball is in the basket." The correct answer must be *inferred* from the fact that Sally was absent during the move. The model has to track who was present for which events and reason about the consequences for each agent's knowledge state. The answer cannot be extracted from surface text — it requires genuine belief tracking.