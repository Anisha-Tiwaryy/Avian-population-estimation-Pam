# Avian Population Estimation from Passive Acoustic Monitoring

Submission for the Darukaa.earth challenge *"Avian Population Estimation using
Passive Acoustic Monitoring"*, built on the BioDCASE 2026 Bird Counting dataset.

**The task.** Given a collection of 3-second passive acoustic recordings from a
zoo aviary, output an integer estimate of how many individuals of a target
species live in that aviary.

**The short version of the approach.** A three-stage pipeline: unsupervised
per-clip acoustic measurement → per-aviary feature aggregation → a
four-parameter pooled log-linear regression, evaluated with leave-one-aviary-out
cross-validation. The design is driven by one uncomfortable fact about this
dataset, which is documented in full in `report/TECHNICAL_REPORT.md`:

> There are 140,899 development clips but only **8 labels** — and because
> `dev_aviary_5` and `dev_aviary_6` are the same population recorded twice,
> only **5 independent sites**. Almost any model you can think of will fit
> those 8 points exactly and generalise not at all.

Everything else in this repository follows from taking that seriously.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies are numpy / scipy / pandas / scikit-learn / soundfile /
matplotlib / tqdm / huggingface_hub. There is **no deep-learning dependency in
the default path** — the DSP measurement backend is pure scipy. BirdNET is
optional (`pip install birdnetlib tensorflow`) and adds a confidence column that
the feature stage will pick up automatically if present.

Verify the install without downloading anything (generates synthetic audio and
runs every stage):

```bash
python scripts/smoke_test.py
```

## 2. Get the data

The full HuggingFace repo is ~287 GB. Downloading it is neither necessary nor
sensible. `src/download.py` pulls a **temporally stratified random sample**
instead — uniformly across (day × hour-of-day) cells, so diel activity structure
is preserved:

```bash
python -m src.download --split dev  --per-aviary 1500 --out data/raw   # ~2.6 GB
python -m src.download --split eval --per-aviary 800  --out data/raw   # optional
```

Taking `chunk_000` instead would give you a few contiguous hours of one night
and would silently destroy the diel features. See `stratified_sample()`.

## 3. Run the pipeline

```bash
bash run_all.sh                      # everything below, in order
```

or step by step:

```bash
# Stage 1 - per-clip acoustic measurement (~35 clips/s/core)
python -m src.detect   --manifest data/raw/manifest_dev.csv \
                       --out results/clip_measurements_dev.parquet \
                       --backend dsp            # or: birdnet | both

# Stage 2 - aggregate to one feature vector per (aviary, species)
python -m src.features --clips results/clip_measurements_dev.parquet \
                       --out results/features_dev.csv

# Stage 3 - fit + leave-one-aviary-out evaluation
python -m src.model    --features results/features_dev.csv --out results/

# Stage 4 - figures for the report
python -m src.eda      --clips results/clip_measurements_dev.parquet \
                       --features results/features_dev.csv \
                       --predictions results/dev_predictions.csv --out figures/

# Stage 5 - inference on held-out aviaries
python -m src.detect   --manifest data/raw/manifest_eval.csv \
                       --out results/clip_measurements_eval.parquet
python -m src.features --clips results/clip_measurements_eval.parquet \
                       --out results/features_eval.csv
python -m src.predict  --features results/features_eval.csv \
                       --model results/model.json --out results/submission.csv
```

Total runtime on a 4-core laptop, excluding download: **under 10 minutes**.

## 4. Repository layout

```
src/config.py     Species definitions, analysis bands, ground truth, CV groups
src/download.py   Stratified subsampled download from HuggingFace
src/detect.py     Stage 1: per-clip acoustic measurement (DSP + optional BirdNET)
src/features.py   Stage 2: per-aviary abundance features
src/model.py      Stage 3: pooled log-linear model, baselines, nested LOAO CV
src/predict.py    Stage 4: integer estimates + 95% prediction intervals
src/eda.py        Figures for the report
scripts/smoke_test.py   Synthetic end-to-end test, no download needed
report/TECHNICAL_REPORT.md
```

## 5. Method in one page

**Stage 1 — measurement, not detection.** For every clip and every target
species' frequency band we compute seven quantities: in-band peak level, the
10th-percentile *floor* level, band occupancy, spectral-flux onset rate, in-band
spectral flatness, the mean number of concurrent spectral peaks, and an SNR
normalised against an out-of-band reference. This is deliberately unsupervised.
With 8 labels, training any per-clip classifier would teach it to recognise
aviaries, not birds.

**Stage 2 — features that don't saturate.** The obvious feature, "fraction of
clips containing a call", is pinned at 1.0 for every flamingo aviary above about
30 birds and therefore cannot distinguish 52 from 161. The features that keep
moving are (a) the **chorus floor**: in a large flock the gaps between calls are
filled by other individuals, so the *quiet* part of each clip gets louder; and
(b) **polyphony proxies**: spectral flatness and concurrent-peak count both grow
with the number of simultaneous callers. All level features are expressed
relative to each aviary's own 10th-percentile baseline so that recording gain
and microphone placement cancel out.

**Stage 3 — four parameters, and not one more.**

```
log N = alpha_species + beta · z(feature)
```

Three species intercepts (unpenalised — a quelea flock and an ibis pair really
are different scales) plus one shared, ridge-penalised, sign-constrained slope.
The log link handles a population range spanning 4 → 161. The shared slope means
each species' 2–4 points contribute to a single estimate of *how* acoustic
activity scales with abundance, instead of three hopeless separate fits.

**Evaluation — leave-one-aviary-out, with feature selection inside the loop.**
Aviaries 5 and 6 are held out together. Feature selection is nested: with 20
candidate features and 8 points, picking the best feature on all the data and
then reporting its CV score is meaningless. `src/model.py` prints both numbers so
the size of that gap is visible. The number to believe is `nested_selection`.

The reference point that matters is `baseline_species_mean` — predicting the
geometric mean count of that species from the training folds, using no audio at
all. A model that cannot beat it is not counting birds; it is exploiting the
species label you were already given.

## 6. Known limitations

Stated up front rather than buried: the frequency bands in `config.py` are
literature approximations and are hyper-parameters, not measurements. Pied
avocet has no development-set label, so its predictions fall back to a global
intercept and are flagged `extrapolated=1` in the submission. Subsampling 1,500
of ~20,000 clips per aviary adds sampling noise to every rate feature (roughly
±1–2% absolute on activity rate). And with 5 independent sites, every reported
error bar is itself uncertain — the honest conclusion of this work is as much
about what *cannot* be established from 8 labels as about what can.

Full discussion, results tables and failure analysis: `report/TECHNICAL_REPORT.md`.
