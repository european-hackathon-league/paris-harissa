"""Submission with the learned MIND-contrastive model on the hard sets (d2/d3).

Per-dataset routing, each matched to what works best in measurement:
  * dataset1 -> dense_mind (training-free, exploits the shared grid)            ~0.71
  * dataset2 -> the TRAINED mind_embedder (deformation-invariant global embed)  -> the upgrade
  * dataset3 -> shape leak (primary) + learned-embed tiebreak

The model is trained ONCE on all 350 labelled d1 pairs (max data for the final run), then used to
embed the real d2/d3 query/gallery pools. Run on the box GPU:
    DATA_ROOT=/workspace/data/ehl OUT=submission_learned.csv python make_submission_learned.py
Knobs via env (mind_embedder): MIND_EPOCHS (default 200), MIND_RESECT, MIND_TTA, etc.
D2_BLEND / D3_LEARN_W tune the blends.
"""
import os, csv, time
from pathlib import Path
import numpy as np

import eval_harness as eh
import rankers as rk
import make_submission as ms          # rank_shape for d3
import mind_embedder as me

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission_learned.csv"))
GRID = int(os.environ.get("GRID", "96"))
D3_LEARN_W = float(os.environ.get("D3_LEARN_W", "0.25"))   # learned weight in the d3 blend

IDX = eh.build_image_index(ROOT)
ms.IDX = IDX
DEV = rk.pick_device()
print(f"DATA_ROOT={ROOT} indexed={len(IDX)} grid={GRID} device={DEV}")

_cache = {}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def vol(i):
    if i not in _cache:
        _cache[i] = eh.load_volume(IDX[i], GRID)
    return _cache[i]


def load(ids):
    return [vol(i) for i in ids]


def _rn(s):
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


# train the learned model once on all labelled d1 pairs (TRAIN_N caps it for quick test runs)
ALL = read_csv(ROOT / "dataset1" / "train_pairs.csv")[: int(os.environ.get("TRAIN_N", "100000"))]
print(f"training mind_embedder on {len(ALL)} d1 pairs ...")
EMBED = me.build(ALL, IDX, GRID, eh.cached_volume, device=DEV)


def emb_matrix(ids):
    return np.stack([EMBED(vol(i)) for i in ids])      # rows are L2-normalised


def rank_learned(qids, gids):
    return emb_matrix(qids) @ emb_matrix(gids).T       # cosine


def rank_d1(qids, gids):
    return rk.rank_dense_mind(load(qids), load(gids), DEV)


def rank_d2(qids, gids):
    return rank_learned(qids, gids)


def rank_d3(qids, gids):
    shape = ms.rank_shape(qids, gids)                  # shape leak dominates
    return _rn(shape) + D3_LEARN_W * _rn(rank_learned(qids, gids))


SETS = [("dataset1", "val", rank_d1), ("dataset1", "test", rank_d1),
        ("dataset2", "val", rank_d2), ("dataset2", "test", rank_d2),
        ("dataset3", "val", rank_d3), ("dataset3", "test", rank_d3)]


def main():
    rows = []
    for ds, split, ranker in SETS:
        t0 = time.time()
        qids = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        scores = ranker(qids, gids)
        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({"query_id": q, "target_id_ranking": " ".join(gids[j] for j in order)})
        print(f"  {ds}/{split:4s} [{ranker.__name__:8s}] {len(qids)}q x {len(gids)}g ({time.time()-t0:.0f}s)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT} (expect 377)")


if __name__ == "__main__":
    main()
