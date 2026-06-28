# Presentation Script — Cross-Modal Retrieval for 3D Medical Images
**EHL Paris 2026 · Inria · ~4 minutes · 9 slides**

> Delivery notes: speak to the *insight*, not the code. Pause on the numbers. The two
> moments that win the room are **Slide 4 (MIND)** and **Slide 6 (the D2 breakthrough)** —
> slow down there. Everything else is setup and payoff.

---

## Slide 1 — Title  *(~20s)*

"Hi, we worked on Inria's cross-modal retrieval challenge for 3D medical images.

In one sentence: given a brain MRI in one contrast, find the *same patient's* scan in a
*different* contrast — and we did it with **no training, no labels, and no data leak**.

Our system reaches **0.703 macro MRR**, up from the 0.455 baseline. Let me show you how."

*(Pause on 0.703.)*

---

## Slide 2 — The task  *(~35s)*

"First, what the task actually is — because it's easy to misread. This is **retrieval, not
classification**. We are *not* labelling tumors.

The query is a contrast-enhanced T1 scan. We're handed a gallery of T2 scans, and we have to
rank them so the **same individual's** T2 is at rank one.

Look at the two images — that's one patient, the same axial slice. On the left, ceT1; on the
right, T2. Same anatomy, but the intensities are *inverted and non-linear* — that's the
**modality gap**. On top of that the volumes are independently positioned and even deformed —
the **geometry gap**. And the patients are unseen, so there's no learning shortcut.

We're scored by mean reciprocal rank, averaged over three datasets."

---

## Slide 3 — Three regimes  *(~25s)*

"Those three datasets are really three difficulty regimes.

**D1** is aligned — the pairs share a common voxel grid. **D2** is the same data, but each
volume has been independently rotated, scaled, and warped. **D3** is the hardest: pre-operative
versus intra-operative scans, where the anatomy has *physically changed* — there's a resection.

One generic matcher would lose. So we made a deliberate choice: **a specialised matcher per
regime.**"

---

## Slide 4 — The key insight: MIND  *(~35s — slow down)*

"Here's the idea that unlocks the whole thing.

The problem with cross-modal matching is intensity: a voxel is bright in T1 and dark in T2.
So we stop looking at intensity. We use **MIND** — the Modality-Independent Neighbourhood
Descriptor. It describes each voxel by the **self-similarity of its local neighbourhood** — the
*texture pattern* around it, not its brightness.

The trick is this: intensities differ between contrasts, but the local structure does *not*.
So the same anatomy produces nearly the same descriptor field in both T1 and T2. We just
correlate those fields.

And crucially — **this needs zero training and zero labels.** It bridges the modality gap by
construction. That's the backbone of every matcher we built."

---

## Slide 5 — The pipeline  *(~25s)*

"So here's the system. Same MIND backbone, specialised per regime.

**D1**, aligned: we correlate the MIND fields voxel-by-voxel on the shared grid — dense MIND.
**D3**, pre-to-intra-op: we match on the original-array shape and break ties with global MIND.
And **D2** — the deformed one — gets the interesting treatment, which is the next slide.

Everything runs on GPU, everything is classical. Nothing to train, nothing to overfit, fully
reproducible from a single script."

---

## Slide 6 — The breakthrough: D2 registration  *(~40s — the climax)*

"D2 was our bottleneck, and fixing it was our single biggest lever.

The insight: **D2 is just D1 with the geometry scrambled** — each volume moved and warped. So
instead of inventing a new matcher, we *undo the deformation*. We register every volume back to
a canonical pose — a rigid-plus-scale transform — and we optimise it directly on the GPU, using
the modality-invariant MIND similarity itself as the objective. About 70 Adam iterations per
volume, no labels, no learned template. Then we just run the D1 matcher.

On our D2 proxy, that took the score from **0.15 to 0.72 — a 4.7× jump.** And it moved the whole
system: macro MRR went from **0.61 to 0.703**.

The lesson: we didn't need a bigger model. We needed to *cancel the nuisance variable.*"

---

## Slide 7 — Results  *(~30s)*

"Putting it together: D1 at 0.72, D2 around 0.60, D3 at 0.85 — macro **0.703**.

The journey: the provided baseline was 0.455. Our per-dataset classical blend with the shape
prior got us to 0.61. And the D2 canonical-pose registration pushed us to 0.703 — a **55%
relative improvement** over baseline.

We'll be honest: D2 still has the most headroom, and that's exactly where we'd push next."

---

## Slide 8 — Scientific rigor  *(~25s)*

"One thing we're proud of: we did the unglamorous work of *ruling things out.*

We suspected an identity leak — the data is in BraTS format — so we downloaded the full
BraTS-2021 set, 1,251 patients, and tested it. The patients simply aren't there: best match
around 0.45 where a true match would be 0.95. Leak is dead.

We also tried a learned contrastive 3D network. It scored 0.17 — worse than classical
everywhere. With only 350 labelled pairs, a global 3D net just collapses.

**Negative results are results.** Documenting the dead ends is *why* the pipeline that survived
is trustworthy."

---

## Slide 9 — Takeaway  *(~20s)*

"So the headline: in this low-data, cross-modal regime, **classical descriptors used well beat
learning.**

MIND bridges the modality gap with no training. A matcher per regime plus GPU registration took
us from 0.455 to 0.703. And it all reproduces in one command.

Happy to show the live demo — a ceT1 query and the ranked T2 gallery — and take your questions.
Thank you."

---

### Q&A — likely questions & crisp answers
- **"Why not deep learning?"** — 350 labelled pairs. We tried; a 3D CNN scored 0.17. Classical
  descriptors don't need data to generalise across modality.
- **"Is 0.703 good?"** — It's +55% over baseline. D1 and D3 are strong (0.72 / 0.85); D2 is the
  open frontier, and we have a concrete next step: a deformable (not just rigid) registration stage.
- **"How do you know there's no leak?"** — We actively tested it against 1,251 BraTS patients and
  it failed at ~0.45 similarity. We can show the probe.
- **"What's the runtime?"** — GPU, ~70 registration iterations per D2 volume; the rest is direct
  field correlation. The whole submission builds from one script.
- **"Clinical relevance?"** — Same-patient cross-modal retrieval supports linking studies across
  scanners/timepoints without identifiers — useful for de-identified data curation and follow-up.
