"""
Stage 2 - aggregate per-clip measurements into one feature vector per
(aviary, species).

Design constraints that drove every choice here
-----------------------------------------------
1. There are 8 labelled points in the development set. Features must therefore
   be few, interpretable, and monotonically motivated by acoustics -- not
   selected by searching a large space against the labels.
2. Recording gain, microphone placement and aviary reverberation differ between
   sites. Every level-based feature is expressed *relative to that aviary's own
   quiet baseline*, so absolute gain cancels out.
3. The dominant failure mode for flock species is saturation: once ~30 flamingos
   are calling, "fraction of clips containing a flamingo call" is pinned at 1.0
   and carries no further information. Features are chosen to keep varying past
   that point (chorus floor, polyphony, simultaneity).

Feature groups
--------------
activity_*     How often the band is active at all. Saturating.
level_*        How loud, relative to the aviary's own baseline. Semi-saturating.
chorus_*       Elevation of the *between-call* floor. Keeps growing with flock
               size -- the workhorse feature for flamingo and quelea.
poly_*         Polyphony / simultaneity proxies (flatness, spectral peak count).
bout_*         Temporal structure: bouts per hour, mean bout length, and the
               burstiness index. Small populations produce sparse, isolated
               bouts; large ones produce near-continuous chorus.
diel_*         Fraction of the day with activity, and dawn-peak concentration.

Usage:
    python -m src.features --clips results/clip_measurements_dev.parquet \
                           --out results/features_dev.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SPECIES


def _read(path):
    p = str(path)
    if p.endswith(".parquet"):
        try:
            return pd.read_parquet(p)
        except Exception:
            p = p.replace(".parquet", ".csv")
    return pd.read_csv(p)


def _burstiness(x: np.ndarray) -> float:
    """Goh & Barabasi burstiness of inter-event intervals. +1 bursty, -1 regular."""
    if len(x) < 3:
        return 0.0
    m, s = float(np.mean(x)), float(np.std(x))
    if m + s == 0:
        return 0.0
    return (s - m) / (s + m)


def aviary_species_features(g: pd.DataFrame) -> dict:
    """g = all clips of one aviary for one species band, sorted by time."""
    g = g.sort_values(["day", "tod_s"]).reset_index(drop=True)
    n = len(g)
    out = {"n_clips": n}

    # --- aviary-internal reference level -------------------------------------
    # The 10th-percentile clip is "this aviary when nothing much is happening".
    # Everything level-based is measured against it, cancelling recording gain.
    base_peak = np.percentile(g["peak_db"], 10)
    base_floor = np.percentile(g["floor_db"], 10)

    rel_peak = g["peak_db"].to_numpy() - base_peak
    rel_floor = g["floor_db"].to_numpy() - base_floor

    # --- activity ------------------------------------------------------------
    # A clip is "active" if its in-band peak stands >=8 dB above the aviary's own
    # quiet baseline. Unsupervised, no labels involved.
    active = rel_peak > 8.0
    out["activity_rate"] = float(active.mean())
    out["activity_rate_strong"] = float((rel_peak > 16.0).mean())

    # Degeneracy guard. In a permanently noisy aviary every clip clears the
    # absolute threshold (rate -> 1.0) and in a very quiet one none do
    # (rate -> 0.0). Either way the bout/diel features below would collapse to
    # constants and silently stop carrying information. When that happens we
    # fall back to a within-aviary quantile split so temporal structure remains
    # measurable. `activity_rate` itself is left at its true (saturated) value,
    # because that saturation is a real signal we want the model to see.
    out["activity_degenerate"] = float(out["activity_rate"] in (0.0, 1.0))
    if out["activity_rate"] < 0.02 or out["activity_rate"] > 0.98:
        active = rel_peak >= np.percentile(rel_peak, 75)

    # --- level ---------------------------------------------------------------
    out["level_p50"] = float(np.percentile(rel_peak, 50))
    out["level_p90"] = float(np.percentile(rel_peak, 90))
    out["level_snr_p90"] = float(np.percentile(g["snr_db"], 90))

    # --- chorus floor (does not saturate) ------------------------------------
    out["chorus_floor_p50"] = float(np.percentile(rel_floor, 50))
    out["chorus_floor_p90"] = float(np.percentile(rel_floor, 90))
    # how much of the time is the floor lifted, i.e. continuous chorus present
    out["chorus_duty"] = float((rel_floor > 6.0).mean())

    # --- polyphony -----------------------------------------------------------
    act = g[active] if active.sum() >= 10 else g
    out["poly_flatness"] = float(act["flatness"].mean())
    out["poly_npeaks"] = float(act["n_peaks"].mean())
    out["poly_occupancy"] = float(act["occupancy"].mean())
    out["poly_onset_rate"] = float(act["onset_rate"].mean())
    # simultaneity index: high onset rate AND high flatness => many callers
    out["poly_simultaneity"] = out["poly_onset_rate"] * out["poly_flatness"]

    # --- temporal / bout structure ------------------------------------------
    bouts, cur = [], 0
    for a in active:
        if a:
            cur += 1
        elif cur:
            bouts.append(cur)
            cur = 0
    if cur:
        bouts.append(cur)
    out["bout_count"] = len(bouts)
    out["bout_mean_clips"] = float(np.mean(bouts)) if bouts else 0.0
    out["bout_max_clips"] = float(np.max(bouts)) if bouts else 0.0

    idx = np.flatnonzero(active)
    out["bout_burstiness"] = _burstiness(np.diff(idx)) if len(idx) > 2 else 0.0

    # --- diel structure ------------------------------------------------------
    hours = (g["tod_s"].to_numpy() // 3600)
    hact = pd.Series(active).groupby(hours).mean()
    out["diel_active_hours"] = float((hact > 0.1).sum())
    out["diel_peak_rate"] = float(hact.max()) if len(hact) else 0.0
    p = hact.to_numpy() + 1e-9
    p = p / p.sum()
    out["diel_entropy"] = float(-(p * np.log(p)).sum() / np.log(max(len(p), 2)))

    # --- optional BirdNET agreement -----------------------------------------
    if "bn_conf" in g.columns and g["bn_conf"].notna().any():
        c = g["bn_conf"].fillna(0.0).to_numpy()
        out["bn_det_rate"] = float((c > 0.25).mean())
        out["bn_mean_conf"] = float(c.mean())
        out["bn_p90_conf"] = float(np.percentile(c, 90))
    return out


def build(clips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (aviary, sp), g in clips.groupby(["aviary_id", "species"]):
        if len(g) < 50:
            print(f"skip {aviary}/{sp}: only {len(g)} clips")
            continue
        r = {"aviary_id": aviary, "species": sp}
        r.update(aviary_species_features(g))
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values(["aviary_id", "species"]).reset_index(drop=True)


FEATURE_COLS = [
    "activity_rate", "activity_rate_strong",
    "level_p50", "level_p90", "level_snr_p90",
    "chorus_floor_p50", "chorus_floor_p90", "chorus_duty",
    "poly_flatness", "poly_npeaks", "poly_occupancy", "poly_onset_rate",
    "poly_simultaneity",
    "bout_count", "bout_mean_clips", "bout_max_clips", "bout_burstiness",
    "diel_active_hours", "diel_peak_rate", "diel_entropy",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = _read(a.clips)
    feats = build(clips)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(a.out, index=False)
    print(feats.to_string(index=False))
    print(f"\nwrote {a.out}")
