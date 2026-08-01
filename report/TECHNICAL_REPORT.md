# Avian Population Estimation from Passive Acoustic Monitoring

**Darukaa.earth technical challenge — BioDCASE 2026 Bird Counting task**
Anisha Tiwary · [DATE]

> **Before submitting:** every `[FILL: …]` marker below needs a number from your
> actual run (`results/metrics.json`, `results/model_log.txt`,
> `results/dev_predictions.csv`). Nothing else in this document depends on the
> results — the argument holds either way, which is deliberate.

---

## 1. Problem framing

The task is abundance estimation, not detection. Given ~20,000 three-second
clips recorded continuously in a zoo aviary, output an integer: how many
individuals of the target species live there. Ground truth comes from zoo
records, so the label is a property of the *collection*, not of any clip.

This distinction matters more than it first appears. Species detection is a
per-clip supervised problem with abundant labels. Abundance estimation here is a
**weakly supervised regression with eight labelled examples**, in which the
per-clip layer must be built without labels at all.

## 2. The central constraint: eight labels

The development set contains 140,899 clips and the following complete label set:

| Aviary | Target | Count | | Aviary | Target | Count |
|---|---|---|---|---|---|---|
| dev_aviary_1 | Red-billed quelea | 153 | | dev_aviary_4 | Greater flamingo | 161 |
| dev_aviary_2 | Greater flamingo | 107 | | dev_aviary_4 | Hadada ibis | 4 |
| dev_aviary_2 | Hadada ibis | 6 | | dev_aviary_5 | Greater flamingo | 52 |
| dev_aviary_3 | Red-billed quelea | 61 | | dev_aviary_6 | Greater flamingo | 52 |

Two things follow immediately.

First, `dev_aviary_5` and `dev_aviary_6` are two recording sessions of the *same
physical population at the same location*. Treating them as independent
observations inflates any cross-validation score, because a model can transfer
site-specific acoustics from one to the other. All cross-validation in this work
holds them out together, leaving **five independent sites and eight points**.

Second, per-species data is thinner still: 4 flamingo points (3 independent),
2 quelea, 2 ibis. Fitting three separate species models means fitting a
two-parameter line to two points — an exact interpolation with zero degrees of
freedom and no possible validation.

The engineering consequence is that model capacity, not detection accuracy, is
the binding constraint. A BirdNET or ARIA embedding gives 1024 dimensions per
clip; there is no honest way to consume them here. Every design decision below
is an answer to *"how do I extract abundance information without spending
degrees of freedom I do not have?"*

## 3. Method

### 3.1 Stage 1 — unsupervised per-clip measurement

For each clip and each target species' frequency band (`src/config.py`;
literature-derived, treated as hyper-parameters and re-checked in §6) the
pipeline computes seven quantities from a 64 ms / 16 ms STFT after decimation to
16 kHz:

| Quantity | Motivation |
|---|---|
| `peak_db` | in-band peak level — presence and intensity of the loudest call |
| `floor_db` | 10th-percentile in-band level — the **chorus floor** |
| `occupancy` | fraction of frames within 6 dB of the clip peak |
| `onset_rate` | in-band spectral-flux onsets per second — call rate proxy |
| `flatness` | Wiener entropy in band on active frames — polyphony proxy |
| `n_peaks` | mean count of prominent spectral peaks — polyphony proxy |
| `snr_db` | peak relative to a 100–400 Hz out-of-band reference — gain-invariant |

Nothing here is trained. That is the point: a supervised per-clip stage fitted
against eight aviary-level labels would learn aviary identity — reverberation,
background HVAC, microphone response — and present it as abundance.

### 3.2 Stage 2 — features that survive saturation

The natural feature, *fraction of clips containing a target call*, fails on
exactly the species that matter most. Above roughly 30 simultaneously vocalising
flamingos the rate is pinned at 1.0, and 52 birds look identical to 161. The
official baseline documentation names this same problem.

Two feature families are designed to keep moving past that ceiling:

**Chorus floor.** With few birds, the gaps between calls are quiet. With many,
the gaps are filled by other individuals, so the *quiet* percentile of each clip
rises even after the peak has saturated. `chorus_floor_p50/p90` and
`chorus_duty` measure this. It is a level feature, so it is expressed relative
to the aviary's own 10th-percentile baseline, cancelling recording gain and mic
placement.

**Polyphony.** More concurrent callers produce a flatter, denser in-band
spectrum. `poly_flatness`, `poly_npeaks` and their product with onset rate
(`poly_simultaneity`) target the number of voices overlapping at an instant
rather than the number of events.

Two supporting families are included: **bout structure** (`bout_count`,
`bout_mean_clips`, `bout_burstiness` — small populations call in isolated
bursts, large ones approach continuous chorus) and **diel structure**
(`diel_active_hours`, `diel_entropy` — a larger flock keeps the band occupied
across more of the day). Twenty features in total, per (aviary, species).

### 3.3 Stage 3 — a four-parameter model

```
log N = α_species + β · z(feature)
```

Three unpenalised species intercepts and one shared, ridge-penalised,
sign-constrained slope. Design notes:

- **Log link.** Populations span 4 → 161. Absolute error on a flock of 161 and
  on a pair of ibis are not comparable quantities; multiplicative error is.
- **Pooled slope.** Instead of three separate two-point fits, all eight points
  contribute to a single estimate of how acoustic activity scales with
  abundance. The species intercepts absorb the fact that a quelea flock and an
  ibis pair are different acoustic scales.
- **Sign constraint.** More in-band acoustic activity may not predict *fewer*
  birds. When the unconstrained fit produces β < 0 it is fitting noise; clipping
  to zero collapses the model to the species-mean baseline, which is the correct
  conservative behaviour.
- **Prediction intervals** come from the residual standard deviation in log
  space, so they are asymmetric in count space — appropriate for a positive,
  right-skewed quantity.

## 4. Experimental protocol

**Cross-validation.** Leave-one-group-out over five groups, with aviaries 5 and
6 in one group.

**Nested feature selection.** With 20 candidate features and 8 points, choosing
the best feature on all the data and then reporting its CV score is circular —
you can always find a feature that fits. Selection is therefore performed
*inside* each outer fold using an inner leave-one-out loop. `src/model.py`
reports both numbers so the gap is visible; the honest figure is
`nested_selection`.

**Baseline that matters.** `baseline_species_mean` predicts the geometric mean
count of that species from the training folds, using no audio whatsoever. It is
strong precisely because the species identity is given. Any audio-based model
that does not beat it under nested LOAO is not counting birds.

**Data subsampling.** 1,500 clips per aviary (≈8–12% of each collection),
sampled uniformly across (day × hour) cells rather than contiguously, so diel
structure survives. Sampling noise on rate features is ≈ ±1–2 percentage points
(binomial, n=1500).

## 5. Results

Fill from `results/metrics.json`:

| Model | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|
| Species-mean baseline (no audio) | [FILL] | [FILL] | [FILL] | [FILL] |
| Single feature: `activity_rate` | [FILL] | [FILL] | [FILL] | [FILL] |
| Best single feature, selected on all folds *(optimistic)* | [FILL] | [FILL] | [FILL] | [FILL] |
| **Pooled log-linear, nested selection *(honest)*** | **[FILL]** | [FILL] | [FILL] | [FILL] |

Per-point predictions (`results/dev_predictions.csv`):

| Aviary | Species | True | Predicted | Error |
|---|---|---|---|---|
| … | … | … | … | … |

Features selected per outer fold: [FILL — from `model_log.txt`; note whether the
same feature is chosen every time. Stability across folds is itself evidence;
instability means the selection is noise-driven and should be reported as such.]

Reference point: the official BioDCASE baseline (ARIA detections + per-species
regression) reports MAE 11.50 / MAPE 10.6% on the development set. Note that
this figure is *not* directly comparable to the nested number above — the two
use different validation protocols and mine deliberately does not select
features against the full label set.

**Figures.** `figures/fig1_saturation.png` (activity rate saturates, chorus
floor does not), `fig2_diel.png` (diel structure preserved by subsampling),
`fig3_predictions.png` (held-out predictions vs species-mean baseline),
`fig4_feature_corr.png` (feature/log-abundance correlation, n=8).

## 6. Failure analysis

**Greater flamingo — synchronous calling.** [FILL: your per-point errors.] The
expected failure is compression: predictions clustered near the species mean
regardless of true count, because 52, 107 and 161 birds all produce a saturated
band. Check whether `chorus_floor_p90` orders the four flamingo aviaries
correctly; if it does not, the polyphony route is the one worth pursuing.

**Red-billed quelea — texture, not events.** With only two points (61 and 153),
any slope is fitted to a single contrast. A large held-out error here is
statistically inevitable, not a model defect, and should be reported that way.

**Hadada ibis — the scale problem.** Counts of 4 and 6 mean that a ±2 error is a
50% relative error. This is where a fundamentally different method belongs:
individual-level clustering of call embeddings, feasible because ibis calls are
loud, impulsive and well separated. Not attempted here for time reasons; see §7.

**Band-assumption risk.** The frequency bands are literature approximations. The
sanity check is whether the flamingo band is more active in flamingo aviaries
than in quelea aviaries; if `fig2_diel.png` does not show that separation, the
bands are wrong and every downstream number is suspect. [FILL: state what you
observed.]

**Aviary 5 / 6 as a repeatability test.** Same population, different days. The
difference between their predictions is a direct measurement of the pipeline's
sensitivity to recording conditions alone. [FILL: report it — it is the single
most informative diagnostic in this dataset, and it costs nothing.]

## 7. Limitations and what I would do next

*Limitations.* Eight labels and five sites; every reported error bar is itself
uncertain. Frequency bands are assumptions, not measurements. Pied avocet has no
development label and its predictions fall back to a global intercept (flagged
`extrapolated=1` in the submission). Vocal activity is behaviour, not census: a
silent bird is invisible to any acoustic method, so all estimates are of
*calling* individuals and the mapping to true population is itself a modelling
assumption.

*Next steps, in order of expected value per unit effort:*

1. **Individual-level clustering for small populations.** Extract embeddings of
   isolated ibis and avocet calls, cluster them, and use the cluster count as a
   direct estimate. This is the only route that addresses the 4-vs-6 problem,
   and small populations are exactly where it is tractable.
2. **A calibrated mixing model instead of a regression.** Simulate chorus at
   known N by mixing single-caller recordings, measure how chorus floor and
   polyphony grow, and invert that curve. This replaces 8 real labels with an
   unlimited synthetic calibration set and turns the problem from statistics
   into physics. `scripts/smoke_test.py` already contains a crude version of the
   mixing generator.
3. **Detector-in-the-loop validation.** Run BirdNET or ARIA on a few thousand
   clips purely to *verify the frequency bands and confirm species presence*,
   rather than to produce features. High value, low degrees-of-freedom cost.
4. **Leave-one-species-out.** Fit on flamingo + quelea, predict ibis. If the
   pooled slope is real biology rather than a fitted artefact, it should
   transfer across two orders of magnitude. This is a much harder test than
   LOAO, and failing it would be informative.

## 8. Reproducibility

`bash run_all.sh` reproduces everything from a clean checkout; all randomness is
seeded (`--seed`, default 0). `python scripts/smoke_test.py` validates every
stage on synthetic audio without any download. Total runtime after download:
under 10 minutes on 4 cores. Dataset: BioDCASE 2026 Bird Counting (CC BY 4.0),
Argın, Härmä & Arslan-Dogan, Maastricht University.
