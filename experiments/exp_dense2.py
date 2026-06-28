"""Hunt a stronger dense cross-modal descriptor than 6-neighbour MIND (dense_mind = 0.71 on d1).

Whatever wins here lifts BOTH d1 (aligned) and d2 (registered-then-dense), because both are capped
by this same matcher. Tests on the 100-pair d1 gallery (the clean cross-modal aligned ceiling):
  * MIND face6 (baseline)         * MIND face6 multi-dilation {1,2,3}
  * MIND face+edge 18             * MIND-SSC (12 self-similarity pairs, Heinrich)
  * MIND-SSC multi-dilation

Run on the box GPU:  cd /shared-docker/amine && DATA_ROOT=/workspace/data/ehl python tools/exp_dense2.py
"""
import os, sys, csv, time, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh
import rankers as rk

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
IDX = eh.build_image_index(ROOT)
DEV = rk.pick_device()
N = int(os.environ.get("EXP_N", "100"))


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
qi = [p["query_id"] for p in pairs]
gi = [p["target_id"] for p in pairs]
true = np.arange(len(pairs))


def stack(ids):
    return torch.from_numpy(np.stack([eh.load_volume(IDX[i], 96) for i in ids]).astype("float32")).to(DEV).unsqueeze(1)


def mrr(qf, gf):
    return round(eh.mrr_from_scores((qf @ gf.t()).cpu().numpy(), true), 3)


FACE = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
EDGE = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
        (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]


def mind(x, dilations, offsets):
    feats = []
    for dil in dilations:
        for o in offsets:
            xs = torch.roll(x, tuple(d * dil for d in o), dims=(2, 3, 4))
            feats.append(F.avg_pool3d((x - xs) ** 2, 3, 1, 1))
    dp = torch.cat(feats, 1)
    var = dp.mean(1, keepdim=True).clamp_min(1e-6)
    m = torch.exp(-dp / var)
    m = m / m.amax(1, keepdim=True).clamp_min(1e-6)
    return F.normalize(m.reshape(x.shape[0], -1), dim=1)


def mind_ssc(x, dilations):
    sn = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]])
    pairs_ = [(i, j) for i in range(6) for j in range(i + 1, 6) if int((sn[i] - sn[j]).abs().sum()) == 2]
    feats = []
    for dil in dilations:
        for i, j in pairs_:
            oi = tuple(int(v) * dil for v in (sn[i] - 1))
            oj = tuple(int(v) * dil for v in (sn[j] - 1))
            a = torch.roll(x, oi, dims=(2, 3, 4))
            b = torch.roll(x, oj, dims=(2, 3, 4))
            feats.append(F.avg_pool3d((a - b) ** 2, 3, 1, 1))
    dp = torch.cat(feats, 1)
    var = dp.mean(1, keepdim=True).clamp_min(1e-6)
    m = torch.exp(-dp / var)
    m = m / m.amax(1, keepdim=True).clamp_min(1e-6)
    return F.normalize(m.reshape(x.shape[0], -1), dim=1)


def main():
    print(f"dense descriptor hunt on d1: N={len(pairs)}, device={DEV}\n")
    q = stack(qi)
    g = stack(gi)
    t0 = time.time()
    trials = {
        "MIND face6 dil2 (baseline)": lambda x: mind(x, [2], FACE),
        "MIND face6 dil{1,2,3}": lambda x: mind(x, [1, 2, 3], FACE),
        "MIND face+edge18 dil2": lambda x: mind(x, [2], FACE + EDGE),
        "MIND face+edge18 dil{1,2}": lambda x: mind(x, [1, 2], FACE + EDGE),
        "MIND-SSC dil2": lambda x: mind_ssc(x, [2]),
        "MIND-SSC dil{1,2,3}": lambda x: mind_ssc(x, [1, 2, 3]),
    }
    res = {name: mrr(fn(q), fn(g)) for name, fn in trials.items()}
    print(f"{'descriptor':32s}  d1-MRR")
    print("-" * 44)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"{k:32s}  {v:.3f}")
    print(f"\nbeat 0.714 to lift BOTH d1 and d2.  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
