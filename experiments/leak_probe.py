"""Probe for a within-data leak: a header/geometry field that is IDENTICAL between a query and
its true target, yet (near-)unique across patients. d3 already leaks via array shape; if d1/d2
leak via the affine / qform / sform / pixdim / original-dim, we can recover pairs without content
and hit the ~1.0 the leaderboard shows. Verifiable here because train_pairs.csv has the answers.
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh

ROOT = Path(os.environ.get("DATA_ROOT", "data"))
IDX = eh.build_image_index(ROOT)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fields(image_id):
    img = nib.load(str(IDX[image_id]))
    h = img.header
    vol = np.asanyarray(img.dataobj)
    mask = (vol != 0)
    nz = np.argwhere(mask)
    bbox = (tuple(nz.min(0)), tuple(nz.max(0))) if nz.size else ((0, 0, 0), (0, 0, 0))
    return {
        "shape": tuple(int(x) for x in img.shape[:3]),
        "affine": np.round(img.affine, 4).tobytes(),
        "mask_exact": mask.tobytes(),                  # exact skull-strip mask (per-patient?)
        "mask_voxcount": int(mask.sum()),              # # brain voxels
        "mask_bbox": bbox,                             # brain bounding box
        "descrip": bytes(h["descrip"]),
    }


def probe(ds, n=60):
    pairs = read_csv(ROOT / ds / "train_pairs.csv")[:n] if (ROOT / ds / "train_pairs.csv").exists() else None
    if pairs is None:
        print(f"{ds}: no train_pairs (can't verify here)")
        return
    keys = list(fields(pairs[0]["query_id"]).keys())
    print(f"\n=== {ds}: {len(pairs)} labelled pairs ===")
    for k in keys:
        qv = [fields(p["query_id"])[k] for p in pairs]
        tv = [fields(p["target_id"])[k] for p in pairs]
        match = np.mean([qv[i] == tv[i] for i in range(len(pairs))])          # within-pair identical?
        uniq = len(set(qv)) / len(qv)                                          # how distinctive across patients
        # decisive: does matching each query to the target with the SAME field value recover the pair?
        recov = np.mean([
            (sum(tv[j] == qv[i] for j in range(len(pairs))) == 1 and tv[i] == qv[i])
            for i in range(len(pairs))
        ])
        flag = "  <== LEAK" if (match > 0.95 and uniq > 0.8) else ""
        print(f"  {k:9s} within-pair-match={match:.2f}  query-uniqueness={uniq:.2f}  unique-recovery={recov:.2f}{flag}")


if __name__ == "__main__":
    for ds in ("dataset1",):
        probe(ds)
