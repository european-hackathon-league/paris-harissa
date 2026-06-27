# Project Documentation

Cross-modal Content-based Retrieval for 3D Medical Images — EHL Paris 2026 Hackathon (Inria / Paris Brain Institute / PRAIRIE track).

## Index

| Doc | What's in it |
|-----|--------------|
| [challenge.md](challenge.md) | The problem, the 3 datasets/difficulty levels, evaluation metric, submission format, judging |
| [setup.md](setup.md) | Environment, Kaggle data access, the hybrid workflow, `entire` CLI session capture |
| [baseline.md](baseline.md) | Baseline architecture, the Kaggle adaptation + path-resolution fixes, reference results |
| [strategy.md](strategy.md) | Why the baseline fails on datasets 2/3 and the roadmap to beat it |
| [progress-log.md](progress-log.md) | Chronological checkpoint of what we've done so far |

## One-paragraph summary

Given a query brain MRI (contrast-enhanced T1) we must rank a gallery of T2 MRIs so the **same patient's** T2 is ranked first — a cross-modal retrieval problem. We train only on 350 perfectly-aligned pairs (dataset1) but are evaluated on three datasets of increasing difficulty (aligned → deformed → pre/post-surgery from a different hospital). The score is the macro-average of per-dataset Mean Reciprocal Rank. The core challenge is **generalization**: learning content-based matching that survives deformation and structural change without labels in those settings.

## Quick facts

- **Kaggle competition slug:** `ehl-paris-medical-image-retrieval`
- **Deadline:** 2026-06-28 10:30
- **Prize:** 500 · **Deliverables:** Pitch Deck + GitHub Repository + Live Demo
- **Metric:** `score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3`
- **Submission limit:** 100 / team / day
- **Official repo:** https://github.com/NicoStellwag/ehl-paris-2026-medical-retrieval
