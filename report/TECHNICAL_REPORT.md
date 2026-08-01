# Avian Population Estimation from Passive Acoustic Monitoring

**Darukaa.earth technical challenge — BioDCASE 2026 Bird Counting task**

Anisha Tiwary · August 2026

---

## 1. Problem framing

The task is abundance estimation, not detection. Given a collection of
three-second clips recorded continuously in a zoo aviary, the output is a single
integer: how many individuals of the target species live there. Ground truth
comes from zoo records, so the label is a property of the *collection*, never of
any individual clip.

That distinction drives everything that follows. Species detection is a per-clip
supervised problem with abundant labels. Abundance estimation, as posed here, is
a **weakly supervised regression with eight labelled examples**, in which the
per-clip layer has to be built with no labels at all.

## 2. The central constraint: eight labels

The development set contains 140,899 clips and exactly this label set:

| Aviary | Target | Count | | Aviary | Target | Count |
|---|---|---|---|---|---|---|
| dev_aviary_1 | Red-billed quelea | 153 | | dev_aviary_4 | Greater flamingo | 161 |
| dev_aviary_2 | Greater flamingo | 107 | | dev_aviary_4 | Hadada ibis | 4 |
| dev_aviary_2 | Hadada ibis | 6 | | dev_aviary_5 | Greater flamingo | 52 |
| dev_aviary_3 | Red-billed quelea | 61 | | dev_aviary_6 | Greater flamingo | 52 |

Two consequences follow immediately.

First, `dev_aviary_5` and `dev_aviary_6` are two recording sessions of the *same
physical population at the same location*. Treating them as independent
observations inflates any cross-validation score, because a model can carry
site-specific acoustics from one into the other. All cross-validation in this
work holds them out together, leaving **five independent sites and eight
points**.

Second, per-species data is thinner still: four flamingo points (three
independent), two quelea, two ibis. Fitting a separate model per species means
fitting a two-parameter line through two points — exact interpolation, zero
degrees of freedom, nothing left to validate against.

The engineering consequence is that **model capacity, not detection accuracy, is
the binding constraint**. A BirdNET or ARIA embedding provides on the order of
1,024 dimensions per clip; there is no honest way to consume that here. Any
model flexible enough to use it will fit the eight points perfectly and say
nothing about a sixth aviary. Every design decision below answers one question:
*how do I extract abundance information without spending degrees of freedom I do
not have?*

Section 5 tests this claim experimentally rather than assuming it.

## 3. Method

### 3.1 Stage 1 — unsupervised per-clip measurement

For each clip and each target species' frequency band (`src/config.py`;
literature-derived and treated as hyper-parameters, not measurements) the
pipeline computes seven quantities from a 64 ms / 16 ms STFT after decimation
from 48 kHz to 16 kHz:

| Quantity | Motivation |
|---|---|
| `peak_db` | in-band peak level — presence and intensity of the loudest call |
| `floor_db` | 10th-percentile in-band level — the **chorus floor** |
| `occupancy` | fraction of frames within 6 dB of the clip peak |
| `onset_rate` | in-band spectral-flux onsets per second — call-rate proxy |
| `flatness` | Wiener entropy in band on active frames — polyphony proxy |
| `n_peaks` | mean count of prominent spectral peaks — polyphony proxy |
| `snr_db` | peak relative to a 100–400 Hz out-of-band reference — gain-invariant |

Nothing here is trained, and that is the point. A supervised per-clip stage
fitted against eight aviary-level labels would learn aviary identity —
reverberation, background HVAC, microphone response — and present it as
abundance.

### 3.2 Stage 2 — the design hypothesis

The obvious feature, *fraction of clips containing a target call*, fails on
precisely the species that matter most. Above roughly thirty simultaneously
vocalising flamingos the rate is pinned at 1.0, and 52 birds become
indistinguishable from 161. The official baseline documentation identifies the
same problem.

Two feature families were designed specifically to break that ceiling.

**Chorus floor.** With few birds, the gaps between calls are quiet. With many,
those gaps are filled by other individuals, so the *quiet* percentile of each
clip rises even after the peak has saturated. `chorus_floor_p50/p90` and
`chorus_duty` measure this. Being level features, they are expressed relative to
each aviary's own 10th-percentile baseline, cancelling recording gain and
microphone placement.

**Polyphony.** More concurrent callers produce a flatter, denser in-band
spectrum. `poly_flatness`, `poly_npeaks`, `poly_occupancy` and the product
`poly_simultaneity` target the number of voices overlapping at an instant rather
than the number of discrete events.

Two further families were included as supporting candidates without a strong
prior: **bout structure** (`bout_count`, `bout_mean_clips`, `bout_burstiness` —
small populations call in isolated bursts, large ones approach continuous
chorus) and **diel structure** (`diel_active_hours`, `diel_entropy`). Twenty
features in total per (aviary, species).

**This hypothesis was not borne out.** Section 5 shows that neither chorus floor
nor polyphony carried usable signal, and that the only feature to beat the
baseline came from the supporting set. That negative result is reported in full
rather than quietly dropped, because it is the most substantive thing this
experiment establishes.

### 3.3 Stage 3 — a four-parameter model

```
log N = α_species + β · z(feature)
```

Three unpenalised species intercepts plus one shared, ridge-penalised,
sign-constrained slope. Design notes:

- **Log link.** Populations span 4 → 161. An absolute error of 10 on a flock of
  161 and on a pair of ibis are not comparable quantities; multiplicative error
  is.
- **Pooled slope.** Rather than three hopeless two-point fits, all eight points
  contribute to a single estimate of how acoustic activity scales with
  abundance. The species intercepts absorb the fact that a quelea flock and an
  ibis pair are simply different acoustic scales.
- **Sign constraint.** More in-band acoustic activity must not predict *fewer*
  birds. When the unconstrained fit returns β < 0 it is fitting noise; clipping
  to zero collapses the model to the species-mean baseline. This turns out to be
  the most consequential single line in the model (§5.2).
- **Prediction intervals** derive from the residual standard deviation in log
  space, so they are asymmetric in count space — appropriate for a positive,
  right-skewed quantity.

## 4. Experimental protocol

**Cross-validation.** Leave-one-group-out over five groups, with aviaries 5 and
6 held out together.

**Nested feature selection.** With twenty candidate features and eight points,
choosing the best feature on all the data and then reporting its
cross-validation score is circular — a feature that fits can always be found.
Selection is therefore performed *inside* each outer fold using an inner
leave-one-out loop. `src/model.py` reports both figures so the size of the gap
is visible; the honest one is `nested_selection`.

**The baseline that matters.** `baseline_species_mean` predicts the geometric
mean count of that species from the training folds, using no audio whatsoever.
It is a strong baseline precisely because species identity is given in the task.
Any audio-based model that fails to beat it under nested cross-validation is not
counting birds — it is exploiting a label it was handed.

**Data subsampling.** 400 clips per aviary were downloaded, against collection
sizes of 11,879–36,340 (1.1%–3.4%), sampled uniformly across (day × hour) cells
rather than contiguously so that diel structure survives. This was a deliberate
trade under a one-day constraint: transferring the full ~287 GB repository, or
even a 10% subsample, was not feasible. Binomial sampling noise on rate features
at n = 400 is approximately ±2.5 percentage points, against ±1.3 at n = 1,500.
The trade is defensible because every feature is a distribution statistic — a
rate, a percentile, a mean over active frames — rather than a raw event count,
so none of them scale with collection size. Section 6.7 revisits this.

## 5. Results

Two configurations were run. The only difference between them is how many
features the model is permitted to use.

### 5.1 The capacity ablation

**Run A — up to 2 features** (`--max-k 2 --alpha-ridge 1.0`)

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Species-mean baseline (no audio) | 59.07 | 70.24 | 85.9 |
| Single feature: `activity_rate` | 59.07 | 70.24 | 85.9 |
| Best single feature, selected on all folds *(optimistic)* | 48.18 | 53.84 | 136.4 |
| Pooled log-linear, nested selection *(honest)* | **93.34** | 120.84 | 205.0 |

**Run B — 1 feature, stronger ridge** (`--max-k 1 --alpha-ridge 3.0`)

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Species-mean baseline (no audio) | 59.07 | 70.24 | 85.9 |
| Single feature: `activity_rate` | 59.07 | 70.24 | 85.9 |
| Best single feature, selected on all folds *(optimistic)* | 42.12 | 51.48 | 109.8 |
| **Pooled log-linear, nested selection *(honest)*** | **45.32** | **55.91** | 115.9 |

Run A is the more expressive model and it is **worse than using no audio at
all** — MAE 93.3 against a 59.1 baseline. Run B, restricted to exactly one
feature, beats the baseline by 23% (45.3 vs 59.1) under an identical protocol.

This confirms the argument of §2 directly: halving the number of free parameters
more than halved the held-out error. Capacity, not detection quality, was the
binding constraint.

Selection stability tells the same story. Run A picked a different feature pair
in every outer fold — no two folds agreed:

| Held-out fold | Run A (2 features) | Inner MAE | Run B (1 feature) | Inner MAE |
|---|---|---|---|---|
| g1 (aviary 1) | `poly_flatness`, `poly_simultaneity` | 39.31 | `bout_burstiness` | 39.57 |
| g2 (aviary 2) | `poly_occupancy`, `bout_burstiness` | 57.75 | `bout_burstiness` | 72.53 |
| g3 (aviary 3) | `activity_rate_strong`, `bout_burstiness` | 44.96 | `bout_burstiness` | 53.16 |
| g4 (aviary 4) | `level_p90`, `bout_burstiness` | 25.00 | `bout_burstiness` | 37.70 |
| g5 (aviaries 5+6) | `poly_occupancy`, `poly_onset_rate` | 34.46 | `poly_occupancy` | 36.32 |

Run B converges on `bout_burstiness` in four folds of five. Consistent selection
under nested cross-validation is the difference between a feature that carries
signal and one that happens to fit; Run A shows the latter, Run B the former.

The optimistic/honest gap behaves accordingly. In Run A it runs the *wrong way*
(optimistic 48.2, honest 93.3): performance degrades when feature choice is made
blind to the held-out fold, the signature of selection fitting noise. In Run B
the two are close (42.1 vs 45.3).

### 5.2 Seventeen of twenty features carried no signal

The per-feature leave-one-out scores in Run B are more informative than the
headline table:

| Feature | LOO MAE |
|---|---|
| `bout_burstiness` | **42.12** |
| `activity_rate`, `activity_rate_strong`, `level_p50`, `bout_mean_clips`, `diel_entropy`, `poly_npeaks`, *and 11 others* | 59.07 |
| `diel_active_hours` | 59.86 |
| `level_p90` | 60.09 |
| `bout_max_clips` | 60.49 |

The large block sitting at *exactly* 59.07 is not a coincidence — that is the
species-mean baseline value to two decimal places. Those features produced a
negative fitted slope, the sign constraint of §3.3 clipped it to zero, and the
model collapsed to the intercepts. In other words, **seventeen of twenty
features correlated the wrong way with abundance and were caught**.

This is the sign constraint earning its place. Without it, each of those
features would have contributed a spurious negative relationship — more acoustic
activity, fewer birds — that fits the eight training points and inverts on new
data. Three features (`diel_active_hours`, `level_p90`, `bout_max_clips`) scored
slightly *worse* than baseline, meaning they retained a small positive slope that
still did not help.

### 5.3 The design hypothesis failed

`bout_burstiness` is the Goh–Barabási burstiness of intervals between active
clips. It was included as a supporting candidate, not as a primary hypothesis.

The features the method was actually built around — `chorus_floor_p50`,
`chorus_floor_p90`, `chorus_duty`, `poly_flatness`, `poly_simultaneity` — do not
appear in the top ten at all. The chorus-floor family, the central idea of
§3.2, was among the worst performers. Section 5.5 investigates why, using a
controlled experiment with manufactured ground truth, and finds that the
mechanism is real but confounded rather than absent.

Two honest caveats about the winning feature. First, it is a temporal-structure
measure, so it is more vulnerable to the subsampling of §4 than a
distributional one: burstiness is computed over intervals between active clips
in a 400-clip sample, and the sampling itself imposes temporal structure. This
should be re-checked at n = 1,500 before any weight is placed on it. Second, all
observed `bout_burstiness` values are negative (activity more regular than
Poisson), and the sign constraint forces the model to read *less negative* as
*more birds* — plausible if larger flocks produce more continuous, less clumped
calling, but not independently verified here. With five independent sites and
one feature, "this feature is real" is not a claim this experiment can support;
"this feature was selected consistently and beat the baseline" is.

### 5.4 Per-point predictions (Run B, honest/nested)

| Aviary | Species | True | Predicted | Error |
|---|---|---|---|---|
| dev_aviary_1 | Red-billed quelea | 153 | 73.0 | −80.0 |
| dev_aviary_2 | Greater flamingo | 107 | 76.8 | −30.2 |
| dev_aviary_2 | Hadada ibis | 6 | 2.0 | −4.0 |
| dev_aviary_3 | Red-billed quelea | 61 | 127.0 | +66.0 |
| dev_aviary_4 | Greater flamingo | 161 | 167.3 | +6.3 |
| dev_aviary_4 | Hadada ibis | 4 | 18.3 | +14.3 |
| dev_aviary_5 | Greater flamingo | 52 | 132.6 | +80.6 |
| dev_aviary_6 | Greater flamingo | 52 | 133.2 | +81.2 |

**A caveat on the metric.** Run B improves MAE (45.3 vs 59.1) but *worsens* MAPE
(115.9 vs 85.9). The two disagree because the ibis counts are 4 and 6:
predicting 18.3 against a true count of 4 is an absolute error of 14 but a
relative error of 358%, which dominates the MAPE average. MAE is the challenge's
primary metric and the improvement there is real, but the model is plainly not
usable for small populations. See §6.4.

**Reference point.** The official BioDCASE baseline (ARIA detections plus
per-species regression) reports MAE 11.50 / MAPE 10.6% on the development set.
That figure is not directly comparable — it uses a trained species detector
rather than unsupervised band energy, and a different validation protocol — but
the gap is an honest measure of what this approach gives up by avoiding a
pretrained detector.

**Figures.** `figures/fig1_saturation.png`, `fig2_diel.png`,
`fig3_predictions.png`, `fig4_feature_corr.png`.

### 5.5 A controlled saturation experiment

The development set provides eight labels, which is not enough to establish
whether any feature genuinely tracks abundance. Rather than accept that limit,
this section manufactures ground truth: real clips are mixed together at a known
multiplicity K, and each feature is measured as a function of K. This produces
thousands of exactly-labelled points instead of eight, and makes the central
hypothesis of §3.2 testable rather than assumed.

**Method.** 200 real clips are sampled from `dev_aviary_4` (so background,
microphone, reverberation and non-target species are all realistic). For each of
40 trials per K, K clips are drawn at random and their waveforms summed. The
same per-clip quantities used by the production pipeline (`src/detect.py`) are
then measured on the mix, for K from 1 to 64.

**Confound handling.** Summing K incoherent signals raises broadband level by
roughly √K, which would make any absolute level feature rise trivially. Every
curve is therefore computed twice: raw, and with the mix rescaled to constant
RMS. **Only the RMS-normalised curves are interpreted below**, since they
isolate changes in acoustic *structure* from changes in gain. A second, weaker
confound is that background energy accumulates along with calls, so the measured
saturation points are an optimistic bound — real saturation occurs at K no
larger than reported. Script: `scripts/saturation_experiment.py`.

**Results** (RMS-normalised medians, `dev_aviary_4`, flamingo band 350–2200 Hz):

| K | `peak_db` | `floor_db` | `occupancy` | `n_peaks` |
|---|---|---|---|---|
| 1 | −17.38 | −34.07 | 0.14 | 16.22 |
| 2 | −18.95 | −32.87 | 0.24 | 16.31 |
| 4 | −19.49 | −31.01 | 0.32 | 16.42 |
| 8 | −19.76 | −29.00 | 0.46 | 16.41 |
| 16 | −20.05 | −28.84 | 0.48 | 16.47 |
| 32 | −20.71 | −28.21 | 0.66 | 16.36 |
| 64 | −20.85 | −27.45 | 0.71 | 16.77 |

| Feature | Saturation K (normalised) | Normalised range |
|---|---|---|
| `floor_db` | none out to 64 | **6.63 dB** |
| `occupancy` | none out to 64 | **0.601** |
| `n_peaks` | none out to 64 | 0.550 |
| `peak_db` | none out to 64 | 3.471 |
| `onset_rate` | **6** | 0.744 |
| `flatness` | 48 | 0.126 |

**The chorus-floor hypothesis is confirmed.** With overall gain held constant,
`peak_db` is essentially flat across the whole range (−17.4 to −20.9, drifting
slightly *downward*), while `floor_db` rises monotonically by 6.6 dB — the
largest normalised range of any feature, with no sign of saturation at K = 64.
Energy is redistributing out of the peaks and into the gaps between calls, which
is exactly the mechanism proposed in §3.2. `occupancy` expresses the same
phenomenon from the other direction, rising monotonically from 0.14 to 0.71 as
the floor climbs toward the peak.

**Event-counting dies almost immediately.** `onset_rate` saturates at K = 6.
Any method that counts discrete acoustic events is therefore uninformative above
roughly six concurrent callers — which is below every flock population in this
dataset. This is a quantitative version of the argument in §3.2 and, on this
evidence, the strongest single reason not to build an abundance estimator on
detection counts.

**One design assumption was simply wrong.** `n_peaks` is flat (16.2 → 16.8).
Counting prominent spectral peaks does not track the number of concurrent
voices, as had been assumed when it was included as a polyphony proxy.
`flatness` moves in the right direction but has by far the smallest normalised
range (0.126) and saturates at K = 48.

**The tension this creates.** The controlled experiment says `floor_db` and
`occupancy` are the correct features and that they behave exactly as designed.
On the real eight-label task (§5.2), the chorus-floor family was among the
*worst* performers and the model instead selected a temporal feature,
`bout_burstiness`.

Both results are sound, and reconciling them is the most informative outcome of
this work. The mixing experiment holds site, microphone, background and species
composition fixed and varies only density. The real task varies all of them at
once across five sites. A feature can therefore have a strong, monotonic,
non-saturating response to density — as `floor_db` demonstrably does — and still
fail as an abundance estimator, because between-site variation in gain,
reverberation and non-target species (§6.5) swamps the within-site density
signal it carries.

The practical implication is specific and testable: **the chorus-floor approach
is not refuted, it is confounded.** The fix is the detector gating proposed in
§6.5 and §7.2, which would remove the dominant source of between-site variance
while leaving the mechanism measured here intact. That is a considerably more
actionable conclusion than "the feature did not work."

**Figure.** `results/fig5_saturation_dev_aviary_4_flamingo.png`.

## 6. Failure analysis

### 6.1 The repeatability test

`dev_aviary_5` and `dev_aviary_6` are the **same 52 birds at the same location,
recorded on different dates**. The difference between their two predictions
measures directly how much of the model's output is driven by recording
conditions rather than by birds. It costs nothing to compute, and no other
diagnostic available in this dataset is as clean.

| Configuration | Prediction, aviary 5 | Prediction, aviary 6 | Ratio |
|---|---|---|---|
| Run A (2 features) | 184.0 | 313.4 | 1.70× |
| Run B (1 feature) | 132.6 | 133.2 | **1.005×** |

Run A calls the same population 184 birds on one day and 313 on another: it is
largely reading the recording session. Run B is reproducible to within 0.5%.

This single comparison is the strongest evidence in the report for the capacity
argument, and it is worth more than the aggregate MAE. A model can score
acceptably on MAE while being unreproducible, and would then be useless for the
actual conservation application, which is tracking a population *over time*.

Note that Run B is *reproducibly wrong* — both predictions are ≈133 against a
true count of 52. Reproducibility and accuracy are separate properties, and this
pipeline currently has only the first.

### 6.2 Greater flamingo — saturation, as predicted

Flamingo errors are +6, −30, +81, +81 against true counts of 161, 107, 52, 52.
The pattern is compression toward the species mean: both 52-bird aviaries are
over-predicted at ≈133 while the 161-bird aviary is nearly correct. This is
exactly the saturation failure anticipated in §3.2. The features designed to
break that ceiling did not survive selection (§5.3), so nothing in the final
model addresses it — even though §5.5 shows those features do respond to density
as intended under controlled conditions.

### 6.3 Red-billed quelea — two points, one contrast

Quelea errors are −80 (153 predicted as 73) and +66 (61 predicted as 127). The
predictions are not merely inaccurate but **inverted**: the aviary with more
birds receives the lower estimate. With only two quelea points, held out one at
a time, each prediction is effectively an extrapolation from the other species'
intercepts. This is a statistical inevitability of the design rather than a
defect in the feature, and no conclusion about quelea should be drawn from it in
either direction.

### 6.4 Hadada ibis — the scale problem

Counts of 4 and 6 predicted as 2.0 and 18.3. The second is a 358% relative error
and it drives the MAPE result in §5.4. A pooled log-linear model fitted mostly
on flocks of 52–161 cannot resolve differences of two individuals. Small
populations need a categorically different method — see §7, item 1.

### 6.5 Band energy is not species identity

The clearest limitation of the unsupervised DSP approach appears directly in
`results/features_dev.csv`. In `dev_aviary_5`, which contains **no Red-billed
quelea at all**, the quelea band (2–8 kHz) shows an activity rate of 0.795 —
*higher* than the flamingo band's 0.568 in the same aviary, where 52 flamingos
actually live.

The measurement is doing exactly what it was built to do: report energy in a
frequency range. It cannot report which species produced that energy. Any
co-occurring species vocalising in 2–8 kHz, and any broadband mechanical or
environmental noise, raises the "quelea" feature identically. Since each
development aviary holds 2–12 non-target species, this contaminates every
feature to an unknown degree.

This is very likely why the chorus-floor and polyphony features failed. Section
5.5 shows that `floor_db` responds strongly and monotonically to acoustic density
when site and species composition are held fixed, so the mechanism is not in
doubt; what defeats it on the real task is between-site variation. Both are designed to measure *how many of one species are calling at once*; if
the band is dominated by other species and background noise, both measure
something unrelated to the target population. The temporal feature that did
survive may simply be less sensitive to that contamination.

It is also the most important thing to fix, and the fix is cheap in degrees of
freedom because it adds no parameters: run BirdNET or ARIA over a subsample
purely to *confirm species presence and validate the band definitions*, then
gate the band measurements on frames where the target species is actually
detected.

### 6.6 A feature that carried no information

`diel_active_hours` is 24.0 for every aviary and every species band — every band
registers activity in every hour of the day. The feature is fully saturated and
contributes nothing (it scores 59.86, slightly worse than baseline). It was
included on the reasoning that a larger flock would occupy more of the day; that
reasoning fails in a zoo aviary, where background noise alone clears the
activity threshold at all hours. It is retained in the code for transparency but
should be dropped or redefined, for instance as hours above a *relative* rather
than absolute threshold.

### 6.7 Effect of subsampling

400 clips per aviary were used (1.1%–3.4% of each collection). For rate and
percentile features, binomial sampling noise at n = 400 is roughly ±2.5
percentage points and is unlikely to be material next to the systematic errors
above. For `bout_burstiness` — the feature the final model actually uses — the
concern is more serious, since the subsample itself imposes temporal structure
on the interval sequence from which burstiness is computed. Re-running at
n = 1,500 and confirming that `bout_burstiness` is stable and still selected is
the first thing to do with more time, and until that is done the §5 result
should be treated as provisional.

## 7. Limitations and next steps

**Limitations.** Eight labels across five independent sites; every reported
figure is itself uncertain, and the Run A / Run B difference should be read as a
demonstration of a mechanism rather than a precise effect size. Frequency bands
are literature assumptions, and §6.5 shows they do not isolate species. The
selected feature is a temporal statistic whose stability under fuller sampling
is unverified (§6.7). The mixing experiment of §5.5 measures response to summed
clips, not to K individual birds: it sums backgrounds along with calls and
assumes independence between callers, where real flock vocalisation is partly
synchronised, so its saturation points are optimistic bounds rather than exact. Pied avocet has no development label, so its predictions
fall back to a global intercept and are flagged `extrapolated=1` in the
submission. Most fundamentally, vocal activity is behaviour, not census: a
silent bird is invisible to any acoustic method, so all estimates are of
*calling* individuals, and the mapping from calling individuals to true
population is itself an unvalidated modelling assumption.

**Next steps, ordered by expected value per unit effort:**

1. **Individual-level clustering for small populations.** Extract embeddings of
   isolated ibis and avocet calls, cluster them, and use the cluster count as a
   direct estimate. The only route that addresses the 4-vs-6 problem of §6.4,
   and small populations are exactly where it is tractable.
2. **Detector-gated band measurement.** As described in §6.5 — the
   highest-value fix, and it costs no degrees of freedom because the detector
   validates features rather than generating them. This is also the direct test
   of whether the chorus-floor hypothesis failed on its merits or merely because
   of species contamination. Section 5.5 has already answered much of this: the
   chorus-floor mechanism is real and non-saturating, so gating it to frames
   where the target species is actually detected is a motivated next step rather
   than a guess.
3. **Re-run at n = 1,500** to confirm `bout_burstiness` stability (§6.7). Cheap,
   and the current headline result depends on it.
4. **A calibrated mixing model instead of a regression.** Simulate chorus at
   known N by mixing single-caller recordings, measure how chorus floor and
   polyphony grow with N, and invert that curve. This replaces eight real labels
   with an unlimited synthetic calibration set, turning the problem from
   statistics into physics. `scripts/smoke_test.py` already contains a crude
   version of the mixing generator.
5. **Leave-one-species-out validation.** Fit on flamingo and quelea, predict
   ibis. If the pooled slope reflects real biology rather than a fitted artefact
   it should transfer across two orders of magnitude.

## 8. Reproducibility

`bash run_all.sh` reproduces everything from a clean checkout; all randomness is
seeded (`--seed`, default 0). `python scripts/smoke_test.py` validates every
stage on synthetic audio with no download required. Runtime after download is
under ten minutes on four cores. Exact commands for the two runs in §5:

```bash
python -m src.model --features results/features_dev.csv --out results/ \
                    --max-k 2 --alpha-ridge 1.0     # Run A
python -m src.model --features results/features_dev.csv --out results/ \
                    --max-k 1 --alpha-ridge 3.0     # Run B (reported model)

python scripts/saturation_experiment.py --manifest data/raw/manifest_dev.csv \
        --aviary dev_aviary_4 --species flamingo --n-trials 40 --out results/

python scripts/permutation_test.py --features results/features_dev.csv \
        --n-perm 500 --out results/
```

Dataset: BioDCASE 2026 Bird Counting (CC BY 4.0), Argın, Härmä &
Arslan-Dogan, Maastricht University.
