"""
Stage 4 - inference: integer population estimate per (aviary, target species),
with a 95% prediction interval.

Usage:
    python -m src.predict --features results/features_eval.csv \
                          --model results/model.json \
                          --out results/submission.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EVAL_TARGETS, SPECIES


def load_model(path):
    with open(path) as fh:
        return json.load(fh)


def predict(features_csv, model_json, out_csv, targets=None):
    m = load_model(model_json)
    feats = pd.read_csv(features_csv)
    targets = targets or EVAL_TARGETS

    rows = []
    for aviary, sp_key in targets.items():
        r = feats[(feats.aviary_id == aviary) & (feats.species == sp_key)]
        if r.empty:
            print(f"no features for {aviary}/{sp_key} - skipping")
            continue
        x = r[m["features"]].to_numpy(dtype=float)[0]
        z = (x - np.array(m["mu"])) / np.array(m["sd"])
        log_n = m["intercepts"].get(sp_key, m["global_intercept"]) \
            + float(np.dot(z, m["beta"]))

        # Species not present in the development labels (Pied avocet) have no
        # fitted intercept. We fall back to the global intercept and flag the
        # prediction as extrapolated - see report §7.
        extrapolated = sp_key not in m["intercepts"]

        n = float(np.exp(log_n))
        lo = float(np.exp(log_n - 1.96 * m["sigma_log"]))
        hi = float(np.exp(log_n + 1.96 * m["sigma_log"]))
        rows.append({
            "aviary_id": aviary,
            "species": SPECIES[sp_key].common_name,
            "scientific_name": SPECIES[sp_key].scientific_name,
            "predicted_count": int(max(1, round(n))),
            "point_estimate": round(n, 1),
            "ci95_low": int(max(1, round(lo))),
            "ci95_high": int(round(hi)),
            "extrapolated": int(extrapolated),
        })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="results/model.json")
    ap.add_argument("--out", default="results/submission.csv")
    a = ap.parse_args()
    predict(a.features, a.model, a.out)
