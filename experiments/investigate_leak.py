"""Hunt for a same-patient fingerprint / leak in the NIfTI files.

We have ground-truth matches for dataset1's 350 train pairs. So for any cheap, non-anatomical
feature (array shape, voxel zooms, affine, dtype, header text, value count, ...), we can ask:
  (a) is it IDENTICAL / very close within a true pair?
  (b) is it UNIQUE enough across patients to retrieve the match?
If some metadata feature retrieves the d1 match near-perfectly, that's a candidate leak — then
we check whether the same feature varies usefully in dataset2/3 (where the real leaders score ~1).
"""
import os, csv, collections
from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(os.environ.get("DATA_ROOT", "data"))


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def index_images(root):
    idx = {}
    for p in root.glob("**/*.nii*"):
        n = p.name
        idx[n[:-7] if n.endswith(".nii.gz") else n[:-4]] = p
    return idx


IDX = index_images(ROOT)


def hdr_facts(image_id):
    img = nib.load(str(IDX[image_id]))
    h = img.header
    shape = tuple(int(x) for x in img.shape[:3])
    zooms = tuple(round(float(z), 4) for z in h.get_zooms()[:3])
    aff = np.round(img.affine, 3)
    return {
        "shape": shape,
        "zooms": zooms,
        "affine": aff,
        "origin": tuple(np.round(aff[:3, 3], 2)),
        "dtype": str(h.get_data_dtype()),
        "descrip": bytes(h["descrip"]).decode(errors="replace").strip("\x00").strip(),
        "db_name": bytes(h["db_name"]).decode(errors="replace").strip("\x00").strip(),
        "aux_file": bytes(h["aux_file"]).decode(errors="replace").strip("\x00").strip(),
    }


def dump_samples(pairs, n=4):
    print("\n=== sample d1 pairs (query  vs  target) ===")
    for p in pairs[:n]:
        q, t = hdr_facts(p["query_id"]), hdr_facts(p["target_id"])
        print(f"\npair {p['query_id']} -> {p['target_id']}")
        for k in ("shape", "zooms", "origin", "dtype", "descrip", "db_name", "aux_file"):
            same = "==" if q[k] == t[k] else "!="
            print(f"  {k:8s} {same}  q={q[k]!s:40s} t={t[k]!s}")
        print(f"  affine equal: {np.allclose(q['affine'], t['affine'])}")


def shape_mrr(pairs):
    """Retrieve target by matching the query's (shape+zooms+origin) vector. MRR on d1."""
    def vec(d):
        return np.array([*d["shape"], *d["zooms"], *d["origin"]], float)
    qs = [vec(hdr_facts(p["query_id"])) for p in pairs]
    gs = [vec(hdr_facts(p["target_id"])) for p in pairs]
    G = np.stack(gs)
    rr = []
    for i, q in enumerate(qs):
        d = np.linalg.norm(G - q, axis=1)
        order = np.argsort(d)
        rr.append(1.0 / (1 + int(np.where(order == i)[0][0])))
    return float(np.mean(rr))


def uniqueness(ids, label):
    shapes = [hdr_facts(i)["shape"] for i in ids]
    c = collections.Counter(shapes)
    print(f"\n[{label}] {len(ids)} imgs | {len(c)} distinct shapes | "
          f"most common: {c.most_common(3)}")
    return shapes


def bijection_check(qids, gids, label):
    """In a val split with no labels: do query shapes and gallery shapes pair up 1-to-1?"""
    qs = [hdr_facts(i)["shape"] for i in qids]
    gs = [hdr_facts(i)["shape"] for i in gids]
    gc = collections.Counter(gs)
    exact = sum(1 for s in qs if gc.get(s, 0) == 1)
    print(f"[{label}] queries whose shape hits EXACTLY ONE gallery image: "
          f"{exact}/{len(qs)}  (high => shape alone nearly solves it)")


def main():
    pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")
    dump_samples(pairs)

    print("\n=== metadata retrieval MRR on the 350 labelled d1 pairs ===")
    print(f"  shape+zooms+origin vector MRR = {shape_mrr(pairs):.4f}  (1.0 => perfect leak)")

    # uniqueness of shape across each pool
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qf = ROOT / ds / f"{split}_queries.csv"
            gf = ROOT / ds / f"{split}_gallery.csv"
            if not qf.exists():
                continue
            qids = [r["query_id"] for r in read_csv(qf)]
            gids = [r["target_id"] for r in read_csv(gf)]
            uniqueness(qids + gids, f"{ds}/{split} all")
            bijection_check(qids, gids, f"{ds}/{split}")


if __name__ == "__main__":
    main()
