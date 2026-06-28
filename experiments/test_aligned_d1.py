"""Quantify the cheap, no-external-data ceiling on dataset1 by exploiting ALIGNMENT.

d1 query/target are voxel co-registered (same BraTS grid), only the modality differs. So an
aligned cross-modal similarity should retrieve the true match far better than the learned
embedder. We measure MRR on a labelled subset with three zero-training similarities:
  * raw intensity cosine          (baseline; weak across modality)
  * gradient-magnitude cosine     (edges align across modality)
  * mutual information            (the standard aligned cross-modal metric)
"""
import os, sys, csv
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh

ROOT = Path(os.environ.get("DATA_ROOT", "data"))
N = int(os.environ.get("N", "40"))
GRID = 96


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def grad_mag(v):
    gz, gy, gx = np.gradient(v)
    return np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)


def mutual_information(a, b, bins=32):
    m = (a > 0.02) & (b > 0.02)
    if m.sum() < 50:
        return 0.0
    h, _, _ = np.histogram2d(a[m], b[m], bins=bins, range=[[0, 1], [0, 1]])
    p = h / h.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def mrr_from_scores(score_fn, qs, gs):
    rr = []
    for i, q in enumerate(qs):
        scores = np.array([score_fn(q, g) for g in gs])
        order = np.argsort(-scores)
        rr.append(1.0 / (1 + int(np.where(order == i)[0][0])))
    return float(np.mean(rr))


def main():
    idx = eh.build_image_index(ROOT)
    pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
    print(f"loading {len(pairs)} d1 pairs at {GRID}^3 ...")
    qv = [eh.load_volume(idx[p["query_id"]], GRID) for p in pairs]
    gv = [eh.load_volume(idx[p["target_id"]], GRID) for p in pairs]
    qg = [grad_mag(v) for v in qv]
    gg = [grad_mag(v) for v in gv]

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print(f"\nMRR on {N} aligned d1 pairs (gallery = {N}):")
    print(f"  raw intensity cosine : {mrr_from_scores(cos, qv, gv):.4f}")
    print(f"  gradient cosine      : {mrr_from_scores(cos, qg, gg):.4f}")
    print(f"  mutual information   : {mrr_from_scores(mutual_information, qv, gv):.4f}")


if __name__ == "__main__":
    main()
