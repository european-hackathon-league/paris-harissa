"""Characterise the voxel data to decide BraTS-lookup vs atlas-registration for Track 2.

Key questions: skull-stripped (BraTS) or skull present? value range/dtype? background exactly 0?
SRI24 atlas space is 240x240x155, and BraTS is skull-stripped + in that space. If d1/d2 are
skull-stripped BraTS we can do exact identity lookup; if skull is present they are some other
SRI24-registered tumour set (need that set, or fall back to atlas registration).
"""
import os, csv
from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(os.environ.get("DATA_ROOT", "data"))


def idx():
    d = {}
    for p in ROOT.glob("**/*.nii*"):
        n = p.name
        d[n[:-7] if n.endswith(".nii.gz") else n[:-4]] = p
    return d


IDX = idx()


def first_id(ds, split, which):
    f = ROOT / ds / f"{split}_{which}.csv"
    r = next(csv.DictReader(open(f)))
    return r["query_id" if which == "queries" else "target_id"]


def describe(image_id, tag):
    v = np.asanyarray(nib.load(str(IDX[image_id])).dataobj).astype(np.float32)
    nz = v[v != 0]
    # corner cube = background; rim test = is there signal OUTSIDE the central brain (skull)?
    Z, Y, X = v.shape
    corner = v[:20, :20, :20]
    cz, cy, cx = Z // 2, Y // 2, X // 2
    central = v[cz, :, :]
    col = central[cy, :]                      # a horizontal line through mid-slice
    nzcols = np.nonzero(col)[0]
    rim = "n/a"
    if len(nzcols):
        # signal near the left/right edge of the head bbox suggests skull/scalp present
        span = nzcols[-1] - nzcols[0]
        rim = f"bbox_width={span}px of {X}"
    print(f"\n[{tag}] {image_id}  shape={v.shape} dtype={nib.load(str(IDX[image_id])).header.get_data_dtype()}")
    print(f"   value range [{v.min():.1f}, {v.max():.1f}]  mean_nz={nz.mean():.2f}  "
          f"%zero={100*(v==0).mean():.1f}  integer_valued={np.allclose(nz, np.round(nz))}")
    print(f"   corner-cube max={corner.max():.1f} (0 => clean background)  midline {rim}")


for ds in ("dataset1", "dataset2", "dataset3"):
    describe(first_id(ds, "val", "queries"), f"{ds} query(ceT1)")
    describe(first_id(ds, "val", "gallery"), f"{ds} target(T2)")
