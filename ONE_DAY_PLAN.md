# One-day execution plan

Timings assume a laptop and a normal home connection. Do them in this order; the
pipeline is designed so that every stage produces a usable artefact, so if you
run out of time you still have something coherent to submit.

| Slot | Time | What | Notes |
|---|---|---|---|
| 1 | 0:00–0:20 | `pip install -r requirements.txt`, then `python scripts/smoke_test.py` | Proves the environment works before any download. If this fails, fix it now, not at hour six. |
| 2 | 0:20–1:20 | `python -m src.download --split dev --per-aviary 1500 --out data/raw` | ~2.6 GB. **Start it and let it run in a second terminal while you do slot 3.** If your connection is slow, drop to `--per-aviary 800`. |
| 3 | 0:20–1:20 | Read `report/TECHNICAL_REPORT.md` end to end and make it yours | Rewrite §1–§4 in your own voice while the download runs. This is the part that gets read most carefully. |
| 4 | 1:20–1:40 | `python -m src.detect …` then `python -m src.features …` | ~3–5 min of compute. Look at the printed feature table before moving on. |
| 5 | 1:40–2:10 | `python -m src.model --features results/features_dev.csv --out results/` | Read the output carefully. Whether you beat the species-mean baseline is the headline result either way. |
| 6 | 2:10–2:40 | `python -m src.eda …`, then open all four figures | Fig 1 is your central argument. If it doesn't show saturation, say so — that's a finding, not a failure. |
| 7 | 2:40–4:00 | Fill every `[FILL]` in the report; write §5 and §6 from what you actually saw | Do not fabricate. A negative result honestly analysed scores better here than a good number you can't defend. |
| 8 | 4:00–4:40 | *Optional:* `RUN_EVAL=1` inference on a few eval aviaries | Only if slots 1–7 are done. Skippable. |
| 9 | 4:40–5:10 | Push to GitHub, check the README renders, confirm `run_all.sh` works from a clean clone | Clone into a fresh folder and run the smoke test once more. |
| 10 | 5:10–5:30 | Export the report to PDF, write the submission email | Keep the email to four lines: repo link, what you built, the headline number, the one thing you'd do next. |

## If something breaks

- **Download too slow** → `--per-aviary 600`. The method degrades gracefully;
  just say in §4 what you used and note the increased sampling noise.
- **`pyarrow` missing** → use `--out results/clip_measurements_dev.csv`. The
  code falls back to CSV automatically.
- **A model number looks absurd** → check `results/features_dev.csv` first. A
  feature that is identical across all six aviaries means the band or the
  threshold is wrong, not the model.
- **You beat nothing** → that is a publishable result with 8 labels. Report the
  species-mean baseline as the honest state of the art on this development set,
  explain why, and put your effort into §6 and §7. Reviewers of a
  research-oriented challenge respond well to this and badly to a suspiciously
  clean number.
