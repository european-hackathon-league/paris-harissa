# Claude ↔ AMD Jupyter workflow

The simple loop for developing on the AMD MI300X box (the JupyterLab container).

## The cycle

1. **Claude writes/edits code locally** in this repo (so `entire` captures the AI session).
2. **Claude uploads it to the box** via the Jupyter API — port 80, no SSH needed:
   ```
   python tools/jupyter_put.py <file>        # e.g. kaggle_baseline.py  or  run_baseline.ipynb
   ```
   Files land in `/shared-docker` and appear in the JupyterLab file browser (hit ↻).
3. **You run it** in JupyterLab on the MI300X (open the notebook → Run All).
4. Repeat.

➡️ **Only when you want to submit**, say so and Claude makes a **Kaggle version** (paste-ready, Kaggle default paths) — see below.

## Box facts

| | |
|---|---|
| JupyterLab | `http://165.245.141.178/lab` (host + token in `.env`) |
| Jupyter file root | `/shared-docker` (uploads land here; explorer shows here) |
| Data | `/workspace/data/ehl` (1454 `.nii`; `dataset1/train_pairs.csv`, …) |
| Output | `/workspace/out` |
| Env | container already has ROCm torch; only `monai`/`nibabel` get pip-installed |

## One script, two modes (no divergence)

`kaggle_baseline.py` is the single source of truth. It switches by env var, so the
same file runs both places:

- **AMD / Jupyter:** `DATA_INPUT_ROOT=/workspace/data/ehl`, `WORK_DIR=/workspace/out`
  (the `run_baseline.ipynb` notebook sets these in a cell).
- **Kaggle:** set nothing → defaults to `/kaggle/input` + `/kaggle/working` and
  auto-pip-installs MONAI.

## Offline validation harness (measure before you submit)

We have labels only for dataset1's 350 pairs, the real val/test matches are hidden, and
Kaggle allows 100 submissions/day. So we **never** tune on the leaderboard blind.
`eval_harness.py` builds synthetic proxies of all three levels from held-out d1 pairs:

- **d1** = real ceT1→T2, untouched → tests the modality gap only.
- **d2** = + independent rigid + elastic warp on each side → deformation invariance.
- **d3** = + synthetic resection cavity + bias field + gamma shift → structural/domain shift.

Score = mean(d1, d2, d3) MRR — the same macro-average shape as the real leaderboard.
An *embedder* is any `volume(np.float32 HxWxD) -> L2-normalised vector`; add new methods to
`EMBEDDERS` and re-run. Ships with three reference embedders (`intensity`, `edges`,
`fingerprint`) that demonstrate the d2/d3 collapse and the modality-invariant fingerprint idea.

**Run it:** upload + open `run_eval.ipynb` on the box → Run All → results table renders
inline and is written to `eval_results.md`.

The box's Jupyter root (`/shared-docker`) is **shared** with a teammate, so our files live
in a personal `amine/` subdir. Upload into it with the optional remote-name arg, and manage
files server-side over HTTP (no SSH) with `tools/jupyter_fs.py`:

```
python tools/jupyter_put.py eval_harness.py amine/eval_harness.py
python tools/jupyter_put.py run_eval.ipynb  amine/run_eval.ipynb
python tools/jupyter_fs.py  ls amine                 # list
python tools/jupyter_fs.py  mkdir amine              # create dir
python tools/jupyter_fs.py  mv old.py amine/old.py   # move/rename
```

`run_eval.ipynb` cell 2 symlinks `data/` → `/workspace/data/ehl` and `out/` → `/workspace/out`
into `amine/`, so the inputs and outputs are browsable in the left file panel; its last cell
gives click-to-download links for `eval_results.md` / `submission.csv`.

Core logic is unit-tested locally (no data/GPU needed) via `python tools/smoke_test.py`.

## Submit mode (only on request)

When you say *"make a Kaggle version"*, Claude produces a paste-ready copy targeting
Kaggle defaults. You paste it into a Kaggle GPU notebook → Run All → download/submit
`submission.csv`. The AMD run and the Kaggle run produce the same `submission.csv`
format (377 rows).
