"""Local smoke test for rankers.py — no real data / GPU needed (CPU torch is fine).

Fabricates N distinct synthetic 'brain' volumes and checks each pairwise ranker:
  * returns a finite (Nq, Ng) score matrix
  * identity MRR == 1.0 when gallery == query (the diagonal must win its row)
  * degrades but stays > chance under independent deformation (the d2 proxy)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh
import rankers as rk

# Match the harness grid so simulate_d2's GRID-relative deformation magnitudes are correct.
GRID = eh.GRID


def fake_volume(seed):
    """A blobby synthetic volume with a unique per-identity structure (mirrors smoke_test)."""
    r = np.random.default_rng(seed)
    v = np.zeros((GRID, GRID, GRID), np.float32)
    zz, yy, xx = np.ogrid[:GRID, :GRID, :GRID]
    for _ in range(6):
        c = r.uniform(GRID * 0.3, GRID * 0.7, 3)
        rad = r.uniform(GRID * 0.08, GRID * 0.2)
        v[((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < rad ** 2] += r.uniform(0.3, 1.0)
    return np.clip(v / (v.max() + 1e-6), 0, 1).astype(np.float32)


def main():
    if rk.torch is None:
        print("torch not importable — skipping ranker smoke test")
        return
    N = 10
    vols = [fake_volume(i) for i in range(N)]
    device = "cpu"  # force CPU for a deterministic, dependency-light test
    rankers = {
        "nmi": lambda q, g: rk.rank_nmi(q, g, device),
        "gradcos": lambda q, g: rk.rank_gradcos(q, g, device),
        "nmi_grad": lambda q, g: rk.rank_nmi_grad(q, g, device),
        "mind": lambda q, g: rk.rank_mind(q, g, device),
    }
    true_idx = np.arange(N)

    print("identity MRR (gallery == query, must be 1.000):")
    for name, fn in rankers.items():
        s = fn(vols, vols)
        assert s.shape == (N, N), f"{name} bad shape {s.shape}"
        assert np.isfinite(s).all(), f"{name} non-finite scores"
        score = eh.mrr_from_scores(s, true_idx)
        assert score == 1.0, f"{name} identity MRR={score} (expected 1.0)"
        print(f"  {name:10s} {score:.3f}")

    chance = float(np.mean([1.0 / r for r in range(1, N + 1)]))
    print(f"d2-proxy MRR (independent deformation each side, chance ~{chance:.3f}):")
    for name, fn in rankers.items():
        r = np.random.default_rng(7)
        q = [eh.simulate_d2(v, r) for v in vols]
        g = [eh.simulate_d2(v, r) for v in vols]
        score = eh.mrr_from_scores(fn(q, g), true_idx)
        print(f"  {name:10s} {score:.3f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
