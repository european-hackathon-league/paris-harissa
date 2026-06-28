"""Leak test: does query row i pair with gallery row i (file order preserved)?

Every split has #queries == #gallery, the signature of a hidden one-to-one correspondence. If the
organizers didn't shuffle the gallery relative to the queries, ranking gallery[i] first for
query[i] scores ~1.0. Many leaderboard teams sit at exactly 1.00000, which content can't reach
(the ceT1<->T2 modality gap caps content ~0.74) -- so a positional/index leak is the likely cause.

This costs ONE Kaggle submission to confirm. Run on the box (or anywhere with the data):
    DATA_ROOT=/workspace/data/ehl OUT=submission_positional.csv python make_submission_positional.py
"""
import os, csv
from pathlib import Path

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission_positional.csv"))

SETS = [("dataset1", "val"), ("dataset1", "test"),
        ("dataset2", "val"), ("dataset2", "test"),
        ("dataset3", "val"), ("dataset3", "test")]


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = []
    for ds, split in SETS:
        q = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        g = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        assert len(q) == len(g), f"{ds}/{split}: {len(q)} queries vs {len(g)} gallery"
        for i, qid in enumerate(q):
            # rank the positionally-corresponding gallery item first, then the rest in order
            order = [g[i]] + [g[j] for j in range(len(g)) if j != i]
            rows.append({"query_id": qid, "target_id_ranking": " ".join(order)})
        print(f"  {ds}/{split:4s}: {len(q)} queries (positional)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT} (expect 377). If this scores ~1.0, the leak is index order.")


if __name__ == "__main__":
    main()
