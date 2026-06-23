# Four-Gate Paper Screen: `tropical_max_plus`

**Date:** 2026-06-23
**Author:** Wes / Konjo AI (with Claude Code)
**Target constraint (charter §1):** Constraint 1 (sequence mixing) and Constraint 3
(exact in-context retrieval), attacked from the **recall / selection** axis. A
(max, +) semiring mixer makes hard selection the primitive operation instead of a
softmax-weighted average (charter §3.2, "Tropical / max-plus algebra").

**Axis targeted:** recall. This is a *recall-axis* play: it tests whether hard
selection can match soft attention on associative recall, i.e. whether the
softmax average is load-bearing or whether a hard (max, +) gather suffices. It is
**not** expected to move the state-tracking axis (parity / Dyck); a pure selector
carries no recurrent state.

> The screen is an analytic argument written *before any code*. Kill on paper if
> Gate 1 (Expressivity) or Gate 3 (Hardware) fails. A candidate that passes all
> four on paper earns a kill test (MQAR against the baselines). See
> `docs/CHARTER.md` §4.

---

## Graveyard pass (charter §4 step 2)

**The primitive.** Replace attention's convex combination with a tropical
(max, +) "matmul". Per head, with compatibility `a_ij = <q_i, k_j>` (causal,
`j <= i`) and values `V`:

```
attention:  out_ic = sum_j softmax_j(a_ij) * v_jc        (expectation over values)
tropical:   out_ic = max_{j<=i} ( a_ij + v_jc )          ((max,+) matrix product)
```

The tropical output, per channel, selects the position maximizing
`compatibility + value`. Selection is intrinsic, not a softmax that happens to
peak.

**Prior art, by neighbor:**

- **Tropical geometry of ReLU networks** (Zhang, Naitzat, Lim 2018; Maragos et al.
  surveys). Establishes that ReLU networks *are* tropical rational maps. This is an
  analysis of existing nets, not a sequence mixer. Confirms the algebra is native
  to deep nets but leaves the mixer slot empty.
- **Morphological / max-plus layers, maxout** (Goodfellow et al. 2013;
  Charisopoulos & Maragos; Mondal et al. "Morphological Networks"). Max-plus used
  for *channel mixing / activations*, not for *position mixing*. Different slot.
- **Hard attention** (Xu et al. 2015 "Show, Attend and Tell"; monotonic hard
  attention, Raffel et al. 2017). This is the closest empirical lineage: attention
  that selects a single position via argmax. The documented verdict is that hard
  attention is **hard to train** (non-differentiable argmax, high-variance
  REINFORCE / straight-through gradients) and generally **underperformed soft
  attention** on language. It lost on **Gate 2 (trainability)**, not Gate 1.
- **Sparse-but-differentiable attention** (sparsemax, Martins & Astudillo 2016;
  entmax, Peters et al. 2019). This is the field's *resolution* of the hard-vs-soft
  tension: keep exact zeros (real selection) while staying differentiable. It is
  the reason pure (max, +) was never pursued — the differentiable middle ground
  worked, so nobody paid the trainability cost of true max-plus.
- **Tropical algebra elsewhere in ML**: tropical SVMs, tropical polynomial
  regression, shortest-path / Viterbi (which *is* a (max, +) scan). Not used as a
  trainable sequence mixer.

**Read.** The exact formulation — a native (max, +) sequence mixer over `a + v` —
is genuinely under-built, as the charter claims. Its nearest neighbor (hard
attention) was tried and lost on **trainability**, and the field routed around it
with sparsemax/entmax rather than fixing max-plus directly. So the known-fatal
result lands on **Gate 2 (a risk we can mitigate with a temperature relaxation)**,
not on Gate 1 or Gate 3. That is precisely the profile of a candidate that
*earns a kill test* rather than dying on paper. No known result kills exact
(max, +) recall or its hardware mapping outright.

---

## Gate 1 — Expressivity

**Question:** Can it do exact retrieval / associative recall *in principle*?

**Verdict:** `pass`

**Argument.** Associative recall *is* a hard lookup: given a query, return the
value bound to the best-matching key. The (max, +) mixer implements exactly this.
If the learned projections make `a_ij` peaked at the matching key position `j*`
(large contrast versus non-matching `j`), then for every channel `c`,

```
out_ic = max_{j<=i} (a_ij + v_jc) ≈ a_{ij*} + v_{j*,c} = (const_i) + v_{j*,c},
```

a verbatim copy of value vector `v_{j*}` up to a per-position additive constant
that the downstream RMSNorm and projection absorb. This is the induction-head /
MQAR capability in its most direct form — arguably *more* natural than softmax,
which blends competing values and must rely on a sharp peak to approximate a
gather. One subtlety: the per-channel `max` can select different `j` across
channels when `a_ij` is flat; sharp `a` (which the model is incentivized to learn)
makes the selection channel-consistent. Capacity is not state-bounded the way the
`ssm` baseline is: like attention, every past position remains addressable, so
there is no K-ceiling on recall in principle. Gate 1 passes.

---

## Gate 2 — Trainability

**Question:** Do gradients survive at depth and length?

**Verdict:** `risk`

**Argument.** This is the honest kill-gate, and the one hard attention died on.
`max` is subgradient-only: gradient flows to the argmax position alone; every
non-selected position gets zero signal in that step, producing sparse,
high-variance, winner-take-all updates and instability when the argmax flips.
**Mitigation (the reason this is `risk`, not `fail`):** the tropical operator has
a clean differentiable relaxation,

```
out_ic = (1/beta) * logsumexp_{j<=i} ( beta * (a_ij + v_jc) )  -->  max_j (a_ij + v_jc)  as beta -> inf,
```

which is smooth everywhere, gives dense gradients to all positions, and recovers
exact (max, +) in the low-temperature limit. A learnable per-head `beta`
(softplus-parameterized) lets training start soft and sharpen toward hard
selection, exactly the anneal that straight-through hard attention lacked. Note
this is genuinely a tropical operator, **not** attention: it is a smoothed max of
`a + v`, not an expectation `sum softmax(a) v`. Gradients survive; whether the
hardened operator trains *as easily* as softmax is the empirical question the kill
test answers. Gate 2 is a real risk, mitigable.

---

## Gate 3 — Hardware

**Question:** Does it reduce to dense matmul, a parallel scan, or an FFT at high
arithmetic intensity? (Scattered gather / dynamic control flow / per-element
branching = dead at scale.)

**Verdict:** `risk`

**Argument.** The core op is a (max, +) matrix product:
`out_ic = max_j (a_ij + v_jc)`. It has the **same regular access pattern as a
GEMM** — a broadcast-add over the contraction index followed by a max-reduction —
with **no scattered gather, no dynamic control flow, and no per-element
branching**. It tiles like a matmul and parallelizes cleanly over `(i, c)`; the
causal case is a lower-triangular (max, +) product. A streaming form exists too:
(max, +) is associative, so a segmented max-scan computes it (this is the Viterbi
structure). So it clears the Gate-3 disqualifiers outright. The **caveat that
makes it `risk` not `pass`**: `max`-`add` is not a fused multiply-add, so there is
**no tensor-core path** and the op is **bandwidth-bound** with lower arithmetic
intensity than a true GEMM. It is matmul-*adjacent* and feasible at scale (the
charter's own read: "bandwidth-bound but matmul-adjacent; feasible"), but it would
not enjoy the FMA throughput attention gets. Gate 3 is a feasibility-yes,
efficiency-caveat risk. It does **not** fail.

---

## Gate 4 — Scaling

**Question:** Does quality improve predictably as compute is added?

**Verdict:** `risk`

**Argument.** Clean knobs exist (width, heads, depth, and the temperature `beta`),
and there is no structural ceiling like the recurrence's fixed-state recall wall —
every position stays addressable. But hard-selection models have no demonstrated
clean scaling law the way softmax attention does, and winner-take-all dynamics
could plateau. No obvious wall at the gauntlet's scale; predictable frontier
scaling is unproven. `risk`.

---

## Decision

**Paper verdict:** `earned-kill-test`
(Gate 1 = pass, Gate 3 = risk; `paper_verdict()` returns paper-dead only if
expressivity or hardware is `fail`. The honest kill-gates are Gate 2 and Gate 3,
both risks, both mitigable. Consistent with charter §3.2.)

**The kill test:** run `tropical` on MQAR across the K range against the
`attention` and `ssm` baselines (`docs/gauntlet_baselines.md`), and report both
axes (also parity / Dyck).

**Pre-registered falsification criterion (written before results):**

> **Recall axis (the test):** `tropical` must match `attention` MQAR within 5%
> accuracy through K=32. A "matches attention on recall" claim must clear the
> 30-run paired Wilcoxon gate (p < 0.05), or for a non-inferiority framing show
> the paired accuracy difference stays within the 5% band across seeds.
> **State-tracking axis (the control):** `tropical` is **not** expected to beat the
> `attention` baseline on parity / Dyck; if it does, that is a surprise worth its
> own note.
> **Hardware:** the operator must reduce to a Gate-3-plausible parallel form
> (broadcast-add + max-reduction / segmented max-scan), argued above.
>
> If `tropical` cannot match attention's recall **and** offer a path to
> sub-quadratic or otherwise advantaged cost, it does not justify replacing
> softmax selection — and *that is the result*: hard selection suffices for recall
> but tropical buys no efficiency, so it stays a curiosity. A clean "matches on
> recall, no efficiency win" is a legitimate negative (charter §4 step 8).

**One-line reason for the ledger:**

`earned-kill-test`: native (max,+) recall passes Gate 1 and is Gate-3-feasible
(matmul-adjacent, bandwidth-bound); the only real risks are trainability
(subgradient max, mitigated by a logsumexp temperature) and unproven scaling.

---

## Kill test result (empirical, `src/gen/primitives/tropical.py`)

Implemented as the smoothed (max, +) mixer of the screen: per-head QK-normalized
compatibility, learnable per-head temperature `beta` (capped, softplus), values
read by the logsumexp relaxation in the matmul form. Run on the full gauntlet at
seed 0 (`docs/gauntlet_tropical.md`); env header recorded there.

### Both axes vs the baselines

| task | `attention` | `ssm` | **`tropical`** |
| --- | --- | --- | --- |
| MQAR K=1 | 1.000 | 1.000 | 1.000 |
| MQAR K=8 | 1.000 | 0.265 | 1.000 |
| MQAR K=16 | 1.000 | 0.192 | 0.990 |
| MQAR K=32 | 1.000 | 0.150 | **0.151** |
| induction | 0.998 | 0.993 | 0.997 |
| selective_copy | 0.992 | 0.996 | 0.990 |
| parity | 0.573 | 0.969 | 0.642 |
| dyck | 0.541 | 1.000 | 0.547 |

### Verdict vs the pre-registered bar: `killed-empirically`

- **Recall axis — FAILS the bar.** Tropical matches `attention` within 5% through
  K=16 (1.000 at K<=8, 0.990 at K=16), confirming hard (max, +) selection equals
  the softmax average on recall at low-to-moderate key counts: the softmax is
  *not* load-bearing there. But at **K=32 recall collapses to 0.151 (chance)**,
  ~85 points below attention. The pre-registered bar ("match attention within 5%
  through K=32") is not met.
- **Not a temperature artifact.** Re-running K=32 with the cap raised to BETA_MAX=40
  still gives 0.151 (sharper selection does not recover it), and the run uses the
  full 3000-step budget without improving off chance, so it is a hard failure, not
  undertraining.
- **Mechanism (the Gate-1 subtlety, realized).** The per-channel `max` selects a
  position independently per channel; once ~32 keys compete and no single position
  dominates every channel, the gathered "value" mixes channels from different
  positions and recall breaks. The fix would be sharper, channel-consistent
  selection — which needs higher `beta`, which collides with the Gate-2/Gate-3
  numerical fragility (winner-take-all blew the scores to inf until QK-norm bounded
  them; the separable-shift matmul underflows under sharp selection). So the recall
  ceiling is gated by exactly the risks the screen flagged.
- **State-tracking axis — as predicted, unmoved.** parity 0.642 / Dyck 0.547 sit at
  `attention`'s ~chance level, not `ssm`'s ~1.0. Tropical is a pure selector with
  no recurrent state (`step` returns `NO_RECURRENT_FORM`), so it does not touch the
  state-tracking corner.
- **No efficiency win.** The operator is bandwidth-bound (no tensor-core FMA path),
  and the numerically safe reference is *slower* than attention here, not faster.

### Statistical gate (30-run paired Wilcoxon, charter §4 step 6)

At MQAR K=8 (30 paired seeds, identical fixed budget per mixer):

- **`tropical` vs `attention`:** median delta +0.000, **p = 0.861, not
  significant** (n_eff=25). The house gate detects no difference: tropical
  *matches* attention on recall where attention is not yet saturated.
- **`tropical` vs `ssm`:** median delta +0.736, **p = 3.0e-6, significant**.
  Tropical is firmly in attention's recall class, not the recall-limited `ssm`'s.

K=32 collapse is robust across seeds: `tropical` = [0.151, 0.153, 0.149] vs
`attention` = [1.000, 1.000, 1.000].

**Conclusion.** Hard (max, +) selection reproduces soft attention's recall up to
moderate K (statistically indistinguishable at K=8, matched within 5% through
K=16) and buys no efficiency, while inheriting attention's state-tracking blind
spot and adding a hard recall ceiling at K=32 that attention does not have. Per
the pre-registered criterion that is the result: it does not justify replacing
softmax selection. A clean, mechanistic negative (charter §4 step 8).
