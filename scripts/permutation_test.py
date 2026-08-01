"""
Permutation test: is the Run B improvement distinguishable from chance?

The headline result of this work is that a one-feature pooled log-linear model
scores MAE 45.32 under nested leave-one-aviary-out cross-validation, against
59.07 for a species-mean baseline that uses no audio at all.

With 8 labelled points and 20 candidate features, that comparison alone does not
establish anything. The relevant question is: *how often would a random
assignment of counts to aviaries produce an improvement this large, given the
same features, the same model and the same selection procedure?*

Method
------
Permute the eight count labels across the eight (aviary, species) points,
holding the feature matrix fixed, and re-run the complete nested pipeline —
including inner-loop feature selection, so the selection procedure's own
capacity to overfit is included in the null. Repeat n_perm times and record the
resulting MAE each time.

Because the species intercepts carry most of the model's predictive power, two
null hypotheses are tested separately:

  unrestricted   Counts shuffled freely across all 8 points. This breaks both
                 the species-count relationship and the feature-count
                 relationship, so it mostly measures whether species identity
                 matters -- which we already know it does. Reported for
                 completeness.

  within-species Counts shuffled only among points of the same species. Species
                 intercepts are preserved exactly; only the pairing between
                 features and counts within a species is destroyed. THIS is the
                 test that matters: it isolates the contribution of the audio
                 features from the contribution of knowing the species.

The within-species null is severely limited by this dataset: quelea and ibis
have 2 points each (2 permutations) and flamingo has 4 (24 permutations), giving
2 x 2 x 24 = 96 distinct label assignments in total, one of which is the
observed one. The smallest attainable p-value is therefore ~1/96 = 0.010, and
the test has very little power. That limit is itself worth reporting.

Usage:
    python scripts/permutation_test.py --features results/features_dev.csv \
                                       --n-perm 1000 --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations as iter_permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_COLS                       # noqa: E402
from src.model import (SpeciesMean, load_dataset, loo_predict,   # noqa: E402
                       metrics, nested_feature_selection)


def run_once(df, feats, alpha_ridge, max_k):
    """Nested MAE for one label assignment."""
    preds, _ = nested_feature_selection(df, feats, max_k=max_k,
                                        alpha_ridge=alpha_ridge)
    ok = ~np.isnan(preds)
    if ok.sum() < len(df):
        return np.nan
    return metrics(df["count"], preds)["MAE"]


def baseline_mae(df, feats):
    p = loo_predict(df, feats[:1], lambda: SpeciesMean())
    return metrics(df["count"], p)["MAE"]


def shuffle_unrestricted(counts, rng):
    return rng.permutation(counts)


def shuffle_within_species(counts, species, rng):
    out = np.array(counts, dtype=float)
    for s in set(species):
        idx = np.flatnonzero(np.array(species) == s)
        out[idx] = rng.permutation(out[idx])
    return out


def n_distinct_within_species(species):
    """Total number of distinct within-species label assignments."""
    from math import factorial
    total = 1
    for s in set(species):
        total *= factorial(sum(1 for x in species if x == s))
    return total


def main(features_csv, outdir, n_perm=1000, alpha_ridge=3.0, max_k=1, seed=0):
    df = load_dataset(features_csv)
    feats = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().all()]
    for c in df.columns:
        if c.startswith("bn_") and df[c].notna().all():
            feats.append(c)

    true_counts = df["count"].to_numpy(dtype=float)
    species = df["species"].tolist()

    observed = run_once(df, feats, alpha_ridge, max_k)
    base = baseline_mae(df, feats)
    print(f"{len(df)} points, {len(feats)} candidate features, "
          f"max_k={max_k}, alpha_ridge={alpha_ridge}")
    print(f"observed nested MAE : {observed:.2f}")
    print(f"species-mean baseline: {base:.2f}")
    print(f"improvement          : {base - observed:.2f} "
          f"({100 * (base - observed) / base:.1f}%)\n")

    n_distinct = n_distinct_within_species(species)
    print(f"distinct within-species label assignments: {n_distinct} "
          f"-> smallest attainable p = {1 / n_distinct:.3f}\n")

    rng = np.random.default_rng(seed)
    results = {"observed_mae": observed, "baseline_mae": base,
               "n_perm": n_perm, "max_k": max_k, "alpha_ridge": alpha_ridge,
               "n_distinct_within_species": n_distinct}

    for label, fn in [("unrestricted", lambda c: shuffle_unrestricted(c, rng)),
                      ("within_species",
                       lambda c: shuffle_within_species(c, species, rng))]:
        null = []
        d = df.copy()
        for i in range(n_perm):
            d["count"] = fn(true_counts)
            m = run_once(d, feats, alpha_ridge, max_k)
            if not np.isnan(m):
                null.append(m)
            if (i + 1) % max(1, n_perm // 10) == 0:
                print(f"  {label}: {i + 1}/{n_perm}")
        null = np.array(null)
        # one-sided: how often does a random labelling do at least as well?
        p = float((np.sum(null <= observed) + 1) / (len(null) + 1))
        results[label] = {
            "p_value": p,
            "null_mae_mean": float(null.mean()),
            "null_mae_median": float(np.median(null)),
            "null_mae_p05": float(np.percentile(null, 5)),
            "n_valid": int(len(null)),
        }
        print(f"\n{label} null:")
        print(f"  null MAE  mean {null.mean():7.2f}   median "
              f"{np.median(null):7.2f}   5th pct {np.percentile(null, 5):7.2f}")
        print(f"  p = {p:.4f}\n")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "permutation_test.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("=" * 60)
    pw = results["within_species"]["p_value"]
    if pw <= 0.05:
        verdict = ("The within-species null is rejected at the 5% level: the "
                   "audio features carry information beyond species identity.")
    elif pw <= 0.20:
        verdict = ("Suggestive but not significant. The observed improvement is "
                   "in the upper tail of the null but cannot be distinguished "
                   "from chance at conventional levels, which is the expected "
                   "outcome given only 8 points.")
    else:
        verdict = ("The within-species null is NOT rejected. A random "
                   "reassignment of counts within each species reproduces this "
                   "level of performance often enough that the improvement "
                   "cannot be attributed to the audio features.")
    print(verdict)
    print("=" * 60)
    print(f"wrote {outdir}/permutation_test.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features_dev.csv")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--alpha-ridge", type=float, default=3.0)
    ap.add_argument("--max-k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.features, a.out, a.n_perm, a.alpha_ridge, a.max_k, a.seed)
