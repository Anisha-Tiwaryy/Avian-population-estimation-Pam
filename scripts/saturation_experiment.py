"""
Controlled saturation experiment: how do the features respond to a KNOWN number
of concurrent vocalisations?

Motivation
----------
The development set provides 8 labels, which is not enough to establish whether
any feature tracks abundance (see report S2, S5). This script sidesteps that
limitation by manufacturing ground truth: real clips are mixed together at a
known multiplicity K, and each feature is measured as a function of K.

This gives thousands of exactly-labelled points instead of eight, and answers
the question the development set cannot:

    For each feature, at what K does it saturate?

That is the empirical version of the hypothesis in report S3.2, which claimed
that `activity_rate` saturates early while `chorus_floor` and the polyphony
measures keep growing. Here that claim is testable rather than assumed.

Method
------
1. Sample real clips from one aviary (the acoustic background, microphone,
   reverberation and non-target species are therefore all realistic).
2. For each trial, draw K clips at random and sum their waveforms. K concurrent
   clips is a proxy for K times the density of concurrent vocalisations.
3. Measure the same per-clip quantities used by the real pipeline (src.detect).
4. Repeat n_trials times per K, and plot the median response with an
   interquartile band.

What this IS and IS NOT
-----------------------
It is a controlled measurement of how each feature responds to increasing
acoustic density, using real recordings.

It is NOT a simulation of K individual birds. Summing K clips also sums K copies
of the background, and real flock calling is partly synchronised rather than
independent. Two specific confounds and their handling:

  amplitude   Summing K incoherent signals raises broadband level by ~sqrt(K),
              which would make any absolute level feature rise trivially. Every
              curve is therefore reported in BOTH raw and level-normalised form
              (the mix is rescaled to constant RMS). Only the normalised curves
              support conclusions about structure rather than gain.

  background  Background energy accumulates along with calls, so measured
              saturation points are an OPTIMISTIC bound: real saturation
              occurs at K at most this large, not larger.

Usage:
    python scripts/saturation_experiment.py --manifest data/raw/manifest_dev.csv \
        --aviary dev_aviary_4 --species flamingo --n-trials 40 --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from src.config import SPECIES                               # noqa: E402
from src.detect import _load, _spectrogram, _band_measurements  # noqa: E402

K_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

# Features reported. "relative" ones are ratios/fractions and are largely
# immune to the amplitude confound; "level" ones are not, and are shown in both
# raw and RMS-normalised form.
LEVEL_KEYS = ["peak_db", "floor_db"]
RELATIVE_KEYS = ["occupancy", "flatness", "n_peaks", "onset_rate"]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix(waves, normalise):
    n = min(len(w) for w in waves)
    m = np.sum([w[:n] for w in waves], axis=0)
    if normalise:
        m = m * (rms(waves[0][:n]) / rms(m))
    return m


def run(manifest, aviary, species_key, n_trials, out, pool_size=200, seed=0):
    rng = np.random.default_rng(seed)
    man = pd.read_csv(manifest)
    man = man[man.aviary_id == aviary]
    if man.empty:
        raise SystemExit(f"no clips for {aviary} in {manifest}")

    paths = man["path"].tolist()
    rng.shuffle(paths)
    paths = paths[:pool_size]
    print(f"loading a pool of {len(paths)} clips from {aviary}...")

    pool = []
    for p in paths:
        try:
            pool.append(_load(p))
        except Exception:
            continue
    print(f"  loaded {len(pool)}")
    if len(pool) < 8:
        raise SystemExit("pool too small")

    band = SPECIES[species_key].band_hz
    rows = []
    for K in K_VALUES:
        if K > len(pool):
            break
        for t in range(n_trials):
            idx = rng.choice(len(pool), size=K, replace=False)
            waves = [pool[i] for i in idx]
            for norm in (False, True):
                m = mix(waves, normalise=norm)
                f, P = _spectrogram(m)
                meas = _band_measurements(f, P, band)
                if meas is None:
                    continue
                rows.append({"K": K, "trial": t, "normalised": int(norm), **meas})
        print(f"  K={K:3d} done")

    df = pd.DataFrame(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"saturation_{aviary}_{species_key}.csv"
    df.to_csv(csv, index=False)

    summary = analyse(df, aviary, species_key, out)
    with open(out / f"saturation_{aviary}_{species_key}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {csv}")
    return df, summary


def saturation_k(ks, med):
    """Smallest K beyond which the curve moves less than 5% of its total range."""
    med = np.asarray(med, dtype=float)
    rng_ = med.max() - med.min()
    if rng_ <= 1e-9:
        return None
    for i in range(len(ks) - 1):
        if np.all(np.abs(med[i + 1:] - med[i]) < 0.05 * rng_):
            return int(ks[i])
    return None


def analyse(df, aviary, species_key, out):
    keys = LEVEL_KEYS + RELATIVE_KEYS
    summary = {"aviary": aviary, "species": species_key, "features": {}}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, key in zip(axes.ravel(), keys):
        entry = {}
        for norm, colour, lab in [(0, "#d1495b", "raw"),
                                  (1, "#00798c", "RMS-normalised")]:
            sub = df[df.normalised == norm]
            if sub.empty:
                continue
            g = sub.groupby("K")[key]
            ks = np.array(sorted(sub.K.unique()))
            med = g.median().reindex(ks).to_numpy()
            lo = g.quantile(0.25).reindex(ks).to_numpy()
            hi = g.quantile(0.75).reindex(ks).to_numpy()
            ax.plot(ks, med, "-o", ms=3, color=colour, label=lab)
            ax.fill_between(ks, lo, hi, color=colour, alpha=0.18)
            sk = saturation_k(ks, med)
            entry["saturation_K_raw" if norm == 0 else "saturation_K_norm"] = sk
            entry["range_raw" if norm == 0 else "range_norm"] = \
                float(np.nanmax(med) - np.nanmin(med))
            if sk is not None:
                ax.axvline(sk, color=colour, ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (concurrent clips mixed)")
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=.3)
        summary["features"][key] = entry
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Feature response to known acoustic density — "
                 f"{aviary}, {SPECIES[species_key].common_name} band "
                 f"({band_str(species_key)})\n"
                 f"dotted line = saturation point (curve moves <5% of range "
                 f"thereafter)", fontsize=11)
    fig.tight_layout()
    p = Path(out) / f"fig5_saturation_{aviary}_{species_key}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"wrote {p}")

    print("\n" + "=" * 68)
    print(f"{'feature':14s} {'sat K (raw)':>12s} {'sat K (norm)':>13s} "
          f"{'range (norm)':>14s}")
    print("=" * 68)
    for k, v in summary["features"].items():
        print(f"{k:14s} {str(v.get('saturation_K_raw')):>12s} "
              f"{str(v.get('saturation_K_norm')):>13s} "
              f"{v.get('range_norm', float('nan')):14.3f}")
    print("=" * 68)
    print("A feature that saturates at low K cannot distinguish large flocks.")
    print("A feature with a small normalised range carries little information")
    print("about density independent of overall gain.")
    return summary


def band_str(species_key):
    lo, hi = SPECIES[species_key].band_hz
    return f"{lo}-{hi} Hz"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/manifest_dev.csv")
    ap.add_argument("--aviary", default="dev_aviary_4")
    ap.add_argument("--species", default="flamingo", choices=list(SPECIES))
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.manifest, a.aviary, a.species, a.n_trials, a.out, a.pool_size, a.seed)
