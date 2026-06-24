# Next Session — Promoted-candidate follow-through: delta-rule scaling + efficient kernel

**Type:** Promotion follow-through (the first `promoted` candidate earns deeper work).
**Depends on:** `docs/screens/state_expansion.md` (the corner result),
`docs/gauntlet_state_expansion.md`, and `docs/DECISION_LOG.md`.

## Where we are (read first)

The map after three screens — the corner is now occupied:

| mixer | recall (MQAR) | state-tracking (parity/Dyck) | verdict |
| --- | --- | --- | --- |
| `attention` | aces to K=32 | ~chance | promoted (baseline) |
| `ssm` | collapses by K=8 | aces | promoted (baseline) |
| `tropical` | aces to K=16, fails K=32 | ~chance | killed-empirically |
| **`state_expansion`** | **far above ssm; K=32 via wider state** | **aces (1.0)** | **promoted** |

The delta rule holds recall **and** state-tracking together — the first
non-specialist. Two things are now earned and two are open.

**Earned (validated this sprint):**
- Error-correcting writes lift recall far above `ssm` (K=8 Wilcoxon p=4.6e-5).
- `beta in (0,2)` (negative eigenvalues) delivers parity/Dyck = 1.0 (p=1.8e-6).
- The K=32 recall ceiling is a *capacity knob*: `d_key=32 -> 0.151`, `d_key=64 -> 0.991`.

**Open (this follow-through):**
1. **Gate-3 kernel (the load-bearing one).** The reference is a slow sequential
   scan. Implement the **chunked-parallel delta rule** (Yang et al. 2024: intra-chunk
   matmuls + one lower-triangular "UT" solve, inter-chunk state carry) and verify it
   computes the *same* function as the sequential `step` (bit-for-bit within tol) on
   the existing tests. This is what makes the promotion real at scale; until it
   exists, the Gate-3 verdict stays `risk`. Keep the slow scan as the reference
   oracle the kernel is checked against.
2. **Scaling study (Gate 4, properly).** A real width/length sweep: MQAR
   accuracy-vs-K for `d_key in {16,32,64,128}`, and length generalization. Turn the
   two-point `d_key` probe into a curve; find where recall saturates vs state size.
3. **Gated DeltaNet (optional A/B).** Add an input-dependent decay on top of the
   delta rule; screen whether it lifts the recall ceiling further or helps length.
   New `gate_card`, new ledger row — do not silently mutate `state_expansion`.

## The one-line kickoff

> Read `docs/screens/state_expansion.md` and `docs/CHARTER.md` (§4 method, Gate 3).
> The delta rule is `promoted` and occupies the corner. Build the chunked-parallel
> kernel and prove it matches the sequential reference, then run the width/length
> scaling study. Treat the slow scan as the oracle; any kernel that disagrees with
> it is wrong.

## Guardrails carried over

- The chunked kernel must be **verified against the sequential reference**, not just
  "looks plausible" — add a test asserting equality on random inputs.
- Slow-but-correct stays the oracle; do not delete it.
- Determinism + version header in every report; fail loud; shape assertions intact.
- Blocking CI stays blocking. One workstream at a time.
- Recall the tropical sprint's lesson on numerics: bound the state, watch for NaNs.
