# Yacine's solution — technical comparison & adoption analysis

Comparison of teammate Yacine's improved solution against the base "amine" 0.703 solution.
Every number below is pulled from Yacine's own tables in `YACINE_NOTES.md` or from the
experiment scripts (`yacine_exp_*.py`, `yacine_sweep_d2.py`, `yacine_confirm_d1.py`,
`yacine_ssl.py`). Claims I could not verify against code are flagged explicitly.

> Source files read (from the box download `…/scratchpad/yacine_box/`):
> `YACINE_NOTES.md`, `make_submission_best.py`, `rankers.py`, `yacine_ssl.py`,
> `yacine_exp_d1.py`, `yacine_exp_d2.py`, `yacine_exp_d3content.py`, `yacine_sweep_d2.py`,
> `yacine_confirm_d1.py`, `yacine_d3_analyze.py`, `yacine_exp_grid.py`.
> Base for diffing (repo root): `rankers.py`, `make_submission_best.py`, `SOLUTION.md`.

---

## Summary

Team progression on the macro-MRR leaderboard:

| Stage | Macro MRR | What changed |
|---|---|---|
| baseline | 0.455 | provided baseline |
| **amine** (base, classical) | **0.703** | MIND descriptor; d2 solved by affine-register-to-canonical + dense-MIND; d3 = shape prior + global-MIND tiebreak |
| **yacine** | **0.789** (78.86%) | 4 changes on top of amine (below) |

Per-dataset deltas, as far as they can be inferred. Note the strong caveat: **most of
Yacine's measured gains are on the OFFLINE PROXY, which is known to be misleading** (the
standard d2 proxy saturates ~0.87 while real-Kaggle d2 ≈ 0.54). Only some numbers are
Kaggle-confirmed.

| Dataset | amine (Kaggle) | yacine change | Evidence basis |
|---|---|---|---|
| d1 | 0.72 | dilation 2→3 (+0.023 proxy); optionally SSL encoder routing | proxy only |
| d2 | ~0.54–0.60 (Kaggle), proxy 0.866 | multi-start registration K=4 | proxy/hard-proxy; real-Kaggle reg path already validated by amine |
| d3 | 0.85 | tiebreak global-MIND → registered dense-MIND | proxy only (shape prior not proxy-testable) |

The +0.086 macro jump (0.703 → 0.789) is real on Kaggle, but the *attribution* of that jump
across the four changes is mostly proxy-inferred — see "real vs proxy" section.

---

## Change-by-change analysis

### 1. Multi-start registration for dataset 2 (K=4 restarts)

**Code.** `rankers.py::register_affine(vol, tmpl_mind, iters=70, lr=0.05, dilation=2,
restarts=1, init_spread=0.5, seed=0)`. The base version had no `restarts`/`init_spread`
arguments — it ran a single Adam optimisation of `(ang, tr, ls)` from **identity init only**.
Yacine refactored the single run into `_register_once(...)` and wraps it in a loop:

```
for j in range(restarts):
    ang0 = zeros            if j == 0   # first restart = original identity init
           else random ±init_spread rad # subsequent restarts = random euler-angle inits
    w, c = _register_once(vol, Tm, iters, lr, dilation, ang0)
    keep per-volume the warp with the best final MIND cosine c  (torch.where)
```

`rank_dense_mind_registered(..., restarts=)` threads it through. `make_submission_best.py`
sets `REG_RESTARTS` (env, default **4**) and `REG_ITERS` (default **100**, up from 70).
Critically, `restarts=1` reproduces amine's original behaviour exactly — it is a pure superset.

**Why it works.** The registration loss (negate MIND cosine to a fuzzy canonical template) is
non-convex in rotation. Identity init is fine for small deformation but sticks in a local
minimum once the rotation is large — which is the regime real d2 actually lives in (notes:
real d2 ≈ 0.54 vs proxy 0.87). Running K independent restarts from random rotations and keeping
the best-cosine warp per volume escapes those minima. No new model, no training.

**Measured improvement** (`yacine_exp_d2.py`, N=40 gallery):

| variant | standard proxy | HARD proxy |
|---|---|---|
| affine (amine, identity init) | 0.866 | 0.654 |
| **multi-start K=4** | 0.871 | **0.806** |
| affine + deformable-to-template | 0.794 | 0.717 |

The win scales with deformation severity: standard proxy barely moves (0.866→0.871, already
saturated), but the HARD proxy jumps **0.654 → 0.806**. The d2 knob sweep
(`yacine_sweep_d2.py`, N=60 hard proxy) gives the chosen defaults:

| restarts | iters | spread | MRR |
|---|---|---|---|
| 1 | 70 | 0.5 | 0.594 |
| 4 | 70 | 0.5 | 0.670 |
| 6 / 8 | 70 | 0.5 | 0.671 / 0.669 |
| **4** | **100** | **0.5** | **0.678** |

Reads: the **knee is at restarts=4** (1→4 is the entire win; 6/8 add nothing), iters 70→100
adds +0.008, spread 0.5 is best.

**Caveats Yacine flags.** (a) Deformable refinement toward the template *hurts*
(0.806 → 0.717 on hard proxy) — it warps subjects toward the fuzzy mean and erases the
identity cues retrieval needs, so it was dropped. (b) All d2 numbers are proxy; the real win
is only inferred to scale similarly. (c) Compute cost is ~Kx the registration (mitigated by
the knee at 4 and the "next idea" of coarse-to-fine).

---

### 2. D1 MIND dilation 2 → 3

**Code.** `make_submission_best.py` adds `D1_DIL = int(os.environ.get("D1_DIL", "3"))` and
`rank_d1` calls `rk.rank_dense_mind(..., dilation=D1_DIL)`. d2/d3 are left at the default
dilation 2. The `_mind(x, dilation)` offsets become 3-voxel face-neighbour shifts on d1's
shared grid.

**Why it works.** On d1 (aligned, shared grid, modality gap only) the larger dilation gives the
MIND descriptor a bigger receptive field that better spans real anatomical structure; dil=1 is
noisy. It only helps where query and gallery share a grid, which is exactly d1.

**Measured improvement** (`yacine_confirm_d1.py`, dense_mind, N=120, 3 seeds):

| dilation | mean MRR |
|---|---|
| 2 (old) | 0.715 |
| **3** | **0.738** |
| 4 | 0.719 |

dil=3 beats 2 on **every seed** (+0.023 mean). Multi-dilation {1,2,3} concat and any NMI blend
*hurt* (`yacine_exp_d1.py`: dil=1 noisy, NMI weak on d1).

**Related rejected sweep — keep GRID=96.** `yacine_exp_grid.py` (dil=3, N=120, 3 seeds): grid
96=0.738, 112=0.726, 128=0.704. Finer grid hurts (MIND is noise-sensitive and the receptive
field shrinks relative to anatomy). Lowest-risk, lowest-effort change of the four.

**Caveat.** Proxy-only; +0.023 is a small absolute gain that may or may not survive on Kaggle.

---

### 3. D3 content tiebreak: global-MIND → registered dense-MIND

**Code.** `make_submission_best.py::rank_d3` now branches on
`D3_CONTENT = os.environ.get("D3_CONTENT", "reg")`. The base did
`rank_mind` (global MIND) as the content term blended at weight `D3_MIND_W=0.3` over the shape
prior `ms.rank_shape`. Yacine's `"reg"` path instead computes
`rk.rank_dense_mind_registered(..., iters=REG_ITERS, restarts=REG_RESTARTS)` — i.e. it reuses
the d2 registration trick as the d3 tiebreak. The shape prior still dominates; content only
reorders ties.

**Why it works.** `yacine_d3_analyze.py` measures the tie structure: the original-array shape
prior uniquely pins only ~36% of d3/test queries (~55% on val), so the other ~64% tie on shape
and are decided entirely by the content term. The old global-MIND tiebreak is essentially noise
on deformed d3; registered dense-MIND recovers real same-subject signal.

**Measured improvement** (`yacine_exp_d3content.py`, simulate_d3 proxy, N=60):

| tiebreak | d3 proxy MRR |
|---|---|
| global MIND (amine) | 0.127 (~noise) |
| dense MIND (no reg) | weak (needs alignment) |
| **registered dense MIND** | **0.584** |

The tiebreak proxy is ~4.6× stronger (0.127 → 0.584).

**Caveat Yacine flags explicitly.** The shape prior itself is real-data and **not
proxy-testable**; the proxy only measures the *content tiebreak* quality, not the end-to-end d3
MRR. He notes this must be validated on Kaggle via `split_submission` (per-dataset score). So
the proxy says the tiebreak is much better, but the actual d3 leaderboard delta is unconfirmed.

---

### 4. Self-supervised 3D encoder (NT-Xent on all 1454 volumes), routing d1 → learned

**Code.** `yacine_ssl.py`. A MIND-input 3D CNN (`Net.encode = F.normalize(enc(rk._mind(x,
dil)))`, encoder from `mind_embedder.Encoder`) trained with NT-Xent contrastive loss. Positives
= two independently augmented views of the **same** volume (`augment()` applies exactly the
d2/d3 distortions: rigid+elastic, bias/gamma/contrast-invert, resection), over all ~1454
volumes, **plus** the 350 labelled cross-modal pairs as true positives (a bonus NT-Xent term).
Inference: `yacine_ssl.rank(qvols, gvols)` → cosine of learned embeddings.
`make_submission_best.py` wires `D1_METHOD = os.environ.get("D1_METHOD", "learned")` so the
**default d1 path routes to the SSL encoder**, with `dense` as fallback.

**Why it works.** amine's labels-only learned model got 0.17 macro (too few pairs, 350, for a
global 3D CNN — confirmed in base `SOLUTION.md`'s rejected list). Self-supervision on all
volumes gives the encoder far more data and explicitly trains invariance to the exact d2/d3
distortions.

**Measured (honest proxy, 60 pairs held OUT of the labelled term — `SSL_HOLDOUT=1` in
`yacine_ssl.py::train`, so no label leak):**

| dataset | learned | dense_mind | dense + 0.5·learned |
|---|---|---|---|
| d1 (un-augmented, clean) | **0.871** | 0.703 | 0.811 |
| d2 | 0.712 | 0.186 (no reg) | 0.304 |
| d3 | 0.466 | 0.145 | 0.217 |

Routing decision in the notes:
- **d1 → learned** (0.87 vs 0.70; the d1 proxy is clean/un-augmented so it should transfer).
- **d2 → registered dense-MIND** (not learned) — the registered path is Kaggle-proven and the
  learned d2 number carries aug-optimism risk.
- **d3 → shape prior + registered tiebreak** (learned d3 0.47 < registered 0.58).
- Blending learned with dense always hurt → use learned **alone** on d1.

**Caveats Yacine flags.** (a) **Aug-optimism**: the encoder is trained on exactly the proxy
augmentations, so proxy d2/d3 learned numbers are optimistic — this is why he does NOT route
d2/d3 to learned. (b) The d1 proxy is "clean/un-augmented" so he argues it transfers, but this
is the **only** assumption protecting the biggest single proxy gain (0.70 → 0.87 on d1) and it
is **unvalidated on Kaggle**. (c) **Portability**: the encoder is box-only — it needs the
trained `ssl_encoder.pt` checkpoint and is explicitly "not in the portable Kaggle notebook".

---

## Real-Kaggle-validated vs proxy-only

Yacine's notes are careful to distinguish these, and the proxy is known-misleading
(standard d2 proxy saturates ~0.87 vs real ≈0.54). Status of each change:

| Change | On Kaggle? | Notes |
|---|---|---|
| 1. Multi-start registration (d2) | **Mechanism Kaggle-validated, magnitude proxy-only.** The *registered* d2 path is already amine's Kaggle-proven win; multi-start is a strict robustness superset (`restarts=1` == old). The *size* of the lift (0.654→0.806) is hard-proxy only. | Low risk — can only help or no-op. |
| 2. d1 dilation 3 | **Proxy-only** (+0.023, 3 seeds). | Real-data d1 untouched by the proxy simulator, so reasonably trustworthy, but unconfirmed. |
| 3. d3 registered tiebreak | **Proxy-only**, and even the proxy can't see the real shape prior. Notes say: validate via `split_submission` on Kaggle. | Tiebreak proxy 0.13→0.58 is large; end-to-end d3 delta unknown. |
| 4. SSL encoder → d1 | **Proxy-only and the riskiest.** Relies on the "d1 proxy is clean so it transfers" assumption; aug-optimism explicitly acknowledged for d2/d3. | Biggest single proxy gain (d1 0.70→0.87) but least validated; also non-portable. |

The headline 0.789 macro is a real Kaggle number, but the breakdown of which change contributed
how much is largely proxy-inferred. The safest interpretation: changes 1–3 are conservative
classical refinements of an already-Kaggle-proven pipeline; change 4 is a genuine bet.

---

## Actionable recommendations for amine (prioritised)

Ranked by impact × low-risk × low-effort. Changes 1–3 are **pure classical drop-ins** — the
base `rankers.py` already has `register_affine` / `rank_dense_mind_registered` to extend; change
4 needs the trained `ssl_encoder.pt` and is box-only.

1. **Adopt multi-start registration (change 1). Highest priority.** Port the `restarts` /
   `init_spread` args and the `_register_once` split into base `rankers.py::register_affine`,
   and thread `restarts` through `rank_dense_mind_registered`. Set `REG_RESTARTS=4`,
   `REG_ITERS=100`. **Risk is essentially zero** (`restarts=1` reproduces the current behaviour
   exactly; it can only escape bad minima), it targets d2 which `SOLUTION.md` already names as
   the dataset with the most headroom, and the registered path is already Kaggle-validated.
   Main cost is ~4× registration compute (acceptable; the knee is at 4).

2. **Adopt d1 dilation 3 (change 2). Second priority — trivial, monotone, multi-seed.** One-line
   change (`dilation=3` in the d1 `rank_dense_mind` call, e.g. via a `D1_DIL` env, default 3).
   +0.023 on every seed, lowest effort/risk of all four. Keep GRID=96 (Yacine's grid sweep shows
   finer hurts). Verify on Kaggle since real d1 is untouched by the proxy.

3. **Adopt the d3 registered tiebreak (change 3) — but validate on Kaggle first.** Drop-in: swap
   the d3 content term from `rank_mind` to `rank_dense_mind_registered`. Proxy says the tiebreak
   is ~4.6× better, but the end-to-end d3 number depends on the (proxy-invisible) shape prior, so
   submit and check the d3 split before trusting it. Costs registration on the d3 set.

4. **SSL encoder (change 4) — defer / treat as a research bet, not a portable adoption.** It
   needs `ssl_encoder.pt` (box-only, absent from the Kaggle notebook) and its single big gain
   (d1 0.70→0.87) rests on an unvalidated transfer assumption with acknowledged aug-optimism. If
   pursued: (a) retrain on the box, (b) submit `D1_METHOD=learned` and compare the **d1 split**
   against dense+dil3 on real Kaggle before committing, (c) keep a classical `D1_METHOD=dense`
   fallback for the portable notebook.

A conservative first step is to ship **1 + 2 only** (both near-zero-risk classical drop-ins),
measure the macro delta on Kaggle, then layer in 3 and 4 with split-level validation.

---

## Open questions / what to verify on Kaggle next

1. **Per-split Kaggle attribution.** Use `split_submission.py` to get per-dataset MRRs and
   confirm where the 0.703→0.789 actually came from. The proxy attribution is untrustworthy.
2. **Does the d1 SSL gain survive on real Kaggle?** This is the single largest proxy gain and the
   only one resting on a transfer assumption. Submit `D1_METHOD=learned` vs `dense` (dil=3) and
   compare the d1 split directly.
3. **Real-d2 magnitude of multi-start.** The hard-proxy lift (0.654→0.806) needs a Kaggle d2-split
   confirmation; real d2 ≈ 0.54 is the regime that matters.
4. **d3 end-to-end, not just tiebreak.** Confirm the registered tiebreak improves the real d3 split
   given the shape prior only pins ~36% of test queries.
5. **Untested next ideas from the notes:** sweep `REG_RESTARTS` 6–8 / `REG_ITERS` 100 if GPU budget
   allows (diminishing returns expected); apply multi-start registration to the ~64% of d3 the shape
   prior leaves tied; coarse-to-fine registration (downsample→register→upsample) for speed at higher
   restart counts.
6. **Portability.** The SSL route is box-only; decide whether the Kaggle-submission notebook ships
   classical-only (changes 1–3) and the encoder stays a box experiment.
