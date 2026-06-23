# Konjo Architecture Charter

Building a sequence model from the constraints up, not from the novelty down.

---

## 0. The premise, corrected

The goal is not "components nobody has tried." Anything nameable has been tried, and what remains untried is usually untried because it fails a hard constraint. The goal is a new point in design space that is both novel and has a reason to work, found by attacking an under-served constraint with mathematics that has not been imported into sequence modeling.

Three truths to hold the whole way:

1. **You cannot forget the problems, only the solutions.** The transformer is one solution to a fixed set of constraints (Section 1). Re-derive from the constraints. Novelty is then a consequence, not a target.
2. **Architecture buys efficiency, not raw capability.** No frontier-tier model is sub-quadratic or hybrid; alternatives win on context length, inference cost, sample efficiency, and continual learning. The realistic win condition is "frontier quality on a better scaling curve along one axis," not "smarter."
3. **Nameable means tried.** Neural cellular automata, predictive coding, tropical networks, geometric algebra, reservoir computing: all have papers. The novelty is never the noun. It is the specific *slot* the noun fills and the *formulation* that passes the four gates.

Konjo failure mode to avoid: DREX. HDC was novel and lost to a plain Mamba baseline for a reason knowable before the GPU-hours were spent. The charter exists to make that reason knowable on paper, first, every time.

---

## 1. The invariant constraints (the real foundation)

Any autoregressive sequence model, transformer or not, must solve all of these. This list is the map. A new architecture is a new set of mechanisms assigned to these slots.

1. **Sequence mixing.** Tokens must exchange information across positions. (Attention's job.)
2. **Channel mixing.** Per-position nonlinear transformation of features. (The FFN's job.)
3. **Exact in-context retrieval.** The model must be able to copy/recall a specific earlier token verbatim (the induction-head / associative-recall capability). This is the capability SSMs and linear attention historically fail, and the one your own SSM-state-for-KV proof closed via the availability-or-informativeness dilemma.
4. **Memory/compute tradeoff.** What state to carry forward vs what to recompute. The KV cache is one answer; a recurrent state is another. The tradeoff is fundamental and your dilemma proof is the sharp statement of it.
5. **Causality with parallel training.** Generation is sequential, but training must parallelize across the sequence or it does not scale. (The reason RNNs lost and attention/SSM-scan won.)
6. **Gradient flow at depth.** The thing must be trainable through many layers. (Residuals, normalization, careful init.)
7. **Hardware alignment.** It must map onto matmul throughput and memory bandwidth on real accelerators. This is the silent killer. Most exotic primitives are mathematically fine and die here.
8. **Positional / order information.** Order must enter somewhere.
9. **Predictable scaling.** A clean knob that converts compute into capability along a smooth law. Without this, it never reaches the frontier no matter how elegant.
10. **Adaptive compute.** Spend more on hard tokens, less on easy ones. Largely unsolved at the frontier; a real opening.

### The four-gate filter

Every candidate primitive gets pre-screened on paper against four gates before any code. Most die at Gate 3.

- **Gate 1, Expressivity.** Can it do exact retrieval / associative recall in principle? If a primitive cannot copy a token from 10k positions back, it cannot be the sole mixer. (Kill test: the MQAR / induction synthetic, Section 4.)
- **Gate 2, Trainability.** Do gradients survive at depth and length? Many recurrent and local-learning schemes do not without heavy machinery.
- **Gate 3, Hardware.** Does it reduce to dense matmul, a parallel scan, or an FFT, on a GPU/TPU at high arithmetic intensity? If it needs scattered gather, dynamic control flow, or per-element branching, it is dead at scale regardless of FLOP count. This gate killed your Metal-kernel work and it kills most exotica.
- **Gate 4, Scaling.** Does quality improve predictably as you add compute? A primitive that is great at 10M params and flat at 1B is a toy.

A primitive that passes all four on paper earns a kill test. One that fails Gate 3 on paper is filed in the graveyard with a note, not coded.

---

## 2. The graveyard (what has been tried, by function)

This is the inventory you asked for: the components that have been published, integrated, or dismissed, grouped by the constraint they serve, each with a terse verdict. Use it as the "do not reinvent" list and as the map of which slots are crowded vs thin.

### 2.1 Input / tokenization
- BPE, WordPiece, Unigram, SentencePiece. *Won, ubiquitous.*
- Byte-level, character-level. *Niche, simple, costly.*
- MegaByte, Byte Latent Transformer (entropy patches), H-Net (dynamic chunking). *Active, promising, crowded.*
- Tokenizer-free learned segmentation. *Active.*
- Multi-token / multi-byte prediction objective. *Won as an auxiliary (Medusa, Meta MTP).*

### 2.2 Positional information
- Sinusoidal absolute, learned absolute. *Superseded.*
- Relative (Shaw), T5 bias. *Superseded.*
- RoPE and extensions (NTK, YaRN, LongRoPE, position interpolation). *Won.*
- ALiBi (linear bias). *Niche, strong for length extrapolation.*
- NoPE (no positional encoding, causal decoders learn order). *Live, underexplored, interesting.*
- CoPE (contextual position), stochastic PE. *Niche.*

### 2.3 Sequence mixing (the crowded core)
- Full softmax attention; MHA, MQA, GQA, MLA (latent KV). *Won.*
- Sparse attention: local/sliding, strided, BigBird, Longformer, Reformer (LSH), Routing Transformer, MoBA, NSA, MInference. *Active, efficiency lane.*
- Linear attention (kernel feature maps): Linear Transformer, Performer, cosFormer. *Niche, recall-weak.*
- Gated/decay linear attention: GLA, RetNet, DeltaNet, Gated DeltaNet, RWKV v4-v7, xLSTM, HGRN. *Active, strong, crowded.*
- Structured state-space: S4, S5, H3, Mamba, Mamba2. *Active, the leading alternative; "illusion of state" recall limits known.*
- Long convolution / implicit conv: Hyena, CKConv, SGConv, lightweight/dynamic conv. *Niche.*
- Token-mixing without attention: MLP-Mixer, gMLP, FNet (Fourier), GFNet (global filters). *Mostly dismissed for language, recall-weak.*
- Memory-augmented: Transformer-XL (recurrence), Compressive Transformer, Memorizing Transformer (kNN), Infini-attention, RETRO (retrieval). *Niche.*
- Test-time training / fast weights: TTT layers, Titans, B'MOJO, fast-weight programmers. *Active, biologically-flavored, the live frontier of "learned state."*
- Energy/associative: modern Hopfield networks (equivalent to attention). *Mostly a reinterpretation.*
- Hybrids: Jamba, Zamba, Samba, Bamba, hymba (SSM + attention stacks). *Active, current best efficiency play, crowded.*

### 2.4 Channel mixing / per-token compute
- FFN (2-layer MLP, ReLU/GELU). *Won.*
- GLU family: SwiGLU, GeGLU, ReGLU. *Won.*
- Mixture-of-Experts: Switch, GShard, top-k, expert-choice, soft MoE, DeepSeekMoE (fine-grained + shared). *Won at scale.*
- Memory layers: product-key memory, Meta memory layers at scale. *Niche, reviving.*
- KAN (learnable edge activations). *Tried for LM, underwhelming so far, Gate 3 risk.*
- Higher-order / multiplicative (sigma-pi, pi-sigma, product units). *Largely abandoned, thin slot.*
- Hypernetworks (weights generated per input). *Niche.*

### 2.5 Normalization
- LayerNorm, RMSNorm, pre-norm. *Won.*
- DeepNorm, ReZero, LayerScale, sandwich/normformer, QK-norm. *Niche stabilizers.*
- nGPT (everything on the unit hypersphere). *Live, elegant, underexplored.*
- BatchNorm, GroupNorm for sequences. *Dismissed.*

### 2.6 Connectivity / depth
- Residual connections. *Won, foundational.*
- Reversible layers (RevNet, Reformer), stochastic depth. *Niche, memory tricks.*
- Universal Transformer (weight tying + ACT adaptive halting). *Dismissed then, reviving as latent reasoning.*
- Looped / recurrent-depth transformers (latent-space "thinking"). *Active, interesting.*
- Mixture-of-Depths (dynamic layer skipping), early exit. *Active, adaptive-compute lane.*

### 2.7 Objective / training paradigm
- Next-token autoregressive (causal LM). *Won.*
- Masked LM (BERT), span corruption (T5), permutation LM (XLNet). *Superseded for generation.*
- Multi-token prediction. *Won as auxiliary.*
- Discrete diffusion LMs: D3PM, SEDD, LLaDA, MDLM. *Active, the main non-autoregressive challenger; parallel decode, recall/efficiency open.*
- Discrete flow matching, energy-based, score matching. *Niche, live.*
- Predictive coding as the learning rule (local, gradient-free-ish). *Niche, scaling work underway (μPC, 100+ layers).*
- Synthetic pre-pre-training (formal languages, NCA rollouts) to seed in-context inference. *Newly active 2025-2026; a data move, not a mixer.*

### 2.8 Numerics / representation
- Low-bit / ternary: BitNet, 1.58-bit. *Active, efficiency.*
- Complex-valued networks. *Niche.*
- Hyperbolic / non-Euclidean embeddings. *Niche, good for hierarchy, thin for full LM.*
- Hyperdimensional computing / VSA / holographic reduced representations. *Dismissed for LM (your DREX data point).*
- Capsule networks (routing-by-agreement). *Abandoned.*
- Spiking / event-driven / neuromorphic. *Niche, efficiency-motivated, Gate 3 hard on current hardware.*

---

## 3. The negative space (where novelty actually lives)

Two veins. Each entry names the constraint it might serve better, why it is under-served, and the gate most likely to kill it. Honesty about the kill-gate is the point.

### 3.1 Dismissed for possibly-bad reasons (revisitable)
- **Reservoir computing / echo state networks.** Fixed random recurrent substrate, train only a readout. Dismissed as not end-to-end trainable. But it is nearly free to run and the "computation at the edge of chaos" result is real. *Constraint:* cheap sequence mixing. *Kill-gate:* Gate 1 (can a fixed reservoir do exact retrieval? probably not alone, so it would be a cheap context-summarizer feeding a small exact-recall head). *Under-served because* the dismissal predates modern readout training and hybrid framing.
- **Hebbian / local plasticity as the memory.** Update state by local correlation rules, no backprop-through-time for the memory. Test-time training revived a sliver of this. *Constraint:* memory + continual learning. *Kill-gate:* Gate 2 (does it learn anything non-trivial without global gradients).
- **External addressable memory (NTM/DNC).** Differentiable read/write to a memory matrix. Powerful, abandoned for being hard to train and Gate-3-hostile (scattered addressing). *Constraint:* exact retrieval + long memory. *Kill-gate:* Gate 3. Revisit only with a content-addressing scheme that reduces to matmul.
- **Higher-order multiplicative units.** Additive neurons won; multiplicative/tensor interactions are a thin, mostly-abandoned slot with more raw expressivity per parameter. *Constraint:* channel mixing. *Kill-gate:* Gate 2/Gate 4 (stability and scaling of products).

### 3.2 Unimported mathematics (the richest vein)
- **Tropical / max-plus algebra.** ReLU networks are already piecewise-linear and tropical, but no sequence *mixer* is built natively in the (max, +) semiring. A tropical mixer would make selection and routing the primitive operation rather than a bolt-on. *Constraint:* sequence mixing + native sparsity. *Kill-gate:* Gate 2 (max is subgradient-only) and Gate 3 (max-reductions are bandwidth-bound but matmul-adjacent; feasible). *Why interesting:* sparse selection is the dominant theme of the whole sub-quadratic field, and max-plus makes it intrinsic.
- **Optimal transport as the mixing operator.** Sinkhorn attention exists, but OT as the core routing primitive (move information mass between positions under a learned cost, with mass conservation) is under-built. *Constraint:* sequence mixing with conservation guarantees. *Kill-gate:* Gate 3 (Sinkhorn iterations are matmul-friendly; tractable).
- **Neural-cellular-automata as the causal mixer.** NCA was just taken as a *data/pretraining* trick. The slot still open is NCA dynamics as the actual token-mixing operator: a local synchronous update rule applied over many micro-steps per layer, giving depth-in-time instead of depth-in-layers. *Constraint:* sequence mixing + adaptive compute (run more steps on harder inputs). *Kill-gate:* Gate 1 (locality vs long-range exact recall; needs a propagation argument) and Gate 5 (causal masking of a 1D CA).
- **Reaction-diffusion / neural PDE dynamics.** Continuous-time field evolution over the sequence, queried by perturbation. Neural ODE/CDE on token streams is thin. *Constraint:* continuous memory, adaptive compute via integration time. *Kill-gate:* Gate 3 (stiff solvers are slow) and Gate 1.
- **Geometric / Clifford algebra representations.** Tokens as multivectors, mixing via geometric product. Exists for physics, barely for language. *Constraint:* representation + channel mixing with built-in structure. *Kill-gate:* Gate 4 (does the inductive bias help language, whose symmetries are unclear).
- **Ultrametric / p-adic representation.** Language hierarchy (characters in words in clauses) is naturally tree-structured, and trees are ultrametric. Almost nothing exists. *Constraint:* hierarchical positional + memory addressing. *Kill-gate:* Gate 3.
- **Sheaf / category-theoretic locality-to-global.** Make consistency between local views the primitive, with a learned restriction map. Sheaf nets exist on graphs, not sequences. *Constraint:* sequence mixing with principled local-global glue. *Kill-gate:* Gate 3 and "is the abstraction earning its keep."
- **Renormalization-group depth.** Make each layer an explicit coarse-graining step (RG flow), so depth = scale, with a principled fixed-point story for very long context. *Constraint:* depth semantics + multiscale memory. *Kill-gate:* Gate 4.
- **Predictive coding as the actual learning rule** (not just an architecture). Local error-driven updates, biologically grounded, now scaling past 100 layers. *Constraint:* trainability without global backprop, continual learning. *Kill-gate:* Gate 4 (matching backprop quality at scale is still open).

The honest read: tropical mixing, OT-as-core, and NCA-as-causal-mixer are the three that best combine a real under-served constraint (intrinsic sparse selection; conservation; adaptive depth-in-time) with a plausible path through Gate 3. They are the first candidates worth a paper-screen.

---

## 4. The Konjo method (so this does not become DREX-2)

1. **Constraint-first.** Pick one slot from Section 1 that is genuinely under-served. Do not start from a primitive you find elegant.
2. **Graveyard pass.** Before any build, write the prior-art graveyard for the chosen primitive. If it has been tried and lost, document why and either fix that exact reason or move on.
3. **Four-gate paper screen.** Score the candidate on the four gates analytically. Kill on paper if it fails Gate 1 or Gate 3. No code for paper-dead ideas.
4. **Synthetic gauntlet before language.** Never debug a new primitive on language. Run the standard capability ladder, cheapest first:
   - Induction heads (copy after a bigram trigger).
   - MQAR (multi-query associative recall) at increasing key counts and distances. This is the single best predictor of whether a primitive can replace attention.
   - Selective copying, sorting, formal-language recognition (Dyck, parity, modular arithmetic) for the Chomsky-hierarchy expressivity profile.
   - Long-range arena style retrieval at increasing length.
   A primitive that cannot pass MQAR will not model language. This gauntlet costs hours, not GPU-weeks.
5. **Kill-test-first.** For each component, define the experiment that would falsify the *need* for it before building it. (Your house rule. It already closed Metal kernels and SSM-state-for-KV.)
6. **Pareto evaluation.** Measure capability against compute (FLOPs, wall-clock, memory) on the frontier, not capability alone. A primitive that wins quality at 3x the cost is dominated. 30-run paired Wilcoxon at p<0.05 before any claim merges.
7. **Compose complementary, not redundant.** From the selector conversation: combine across scope or stage (coarse/fine/local, prefilter/rerank), never ensemble redundant estimators of the same quantity.
8. **Negative results are the deliverable.** A clean "tropical mixing fails MQAR above 64 keys for reason X" is a publishable arXiv note and a real contribution. Most of the program will produce these. That is success, not failure.

---

## 5. Where to start

Recommended first target constraint: **exact in-context retrieval at sub-quadratic cost without a growing KV cache**, OR **adaptive per-token compute**. Both are genuinely under-served at the frontier, both are where you already have signal (the dilemma proof bounds the first; nobody has solved the second), and both have a clean synthetic kill test.

Concrete first moves:
1. Stand up the synthetic gauntlet (induction + MQAR + selective copy) as a tiny harness. This is reusable across every future candidate and is the cheapest thing in the whole program. Build it once.
2. Paper-screen the three lead candidates (tropical mixer, OT-as-core, NCA-as-causal-mixer) against the four gates. Expect at least one to die on paper. Document why.
3. For the survivor, the first kill test is MQAR: implement the primitive in the slow-but-correct form (no kernel), run MQAR at growing key counts against a tiny attention baseline and a tiny Mamba baseline. If it cannot match them on associative recall, it cannot be the mixer, and you have spent a day to learn it. If it holds, it has earned real investment.

The decision the gauntlet returns is binary and cheap, which is the whole point. You find out whether a "completely new" mixer has the one capability that matters before committing to it, exactly as the SubQ kill test finds out whether a sparse selector is load-bearing before committing to it.

---

### One-line framing to keep

Do not build the architecture of forgotten components. Build the architecture that solves one invariant constraint better than attention does, using mathematics no one has put in that slot, and let the novelty be the residue of the rigor rather than the goal.
