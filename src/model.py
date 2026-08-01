"""
Stage 3 - abundance estimation and honest evaluation.

The statistical situation, stated plainly
-----------------------------------------
The development set has 140,899 audio clips but only **8 labels**:

    dev_aviary_1  quelea   153        dev_aviary_4  flamingo 161
    dev_aviary_2  flamingo 107        dev_aviary_4  ibis       4
    dev_aviary_2  ibis       6        dev_aviary_5  flamingo  52
    dev_aviary_3  quelea    61        dev_aviary_6  flamingo  52

and aviaries 5 and 6 are the same population recorded twice, so there are 5
independent sites, not 6. Any model with more than a handful of free parameters
will interpolate those 8 points perfectly and generalise not at all. That is the
central engineering constraint of this task, and it dictates:

  * a *pooled* model across species (shared slope) rather than 3 separate fits,
  * a log link (populations span 4 -> 161, two orders of magnitude),
  * at most 1-2 features, chosen inside the CV loop, not before it,
  * a monotonicity constraint (more acoustic activity must not predict fewer
    birds) so the fit cannot exploit noise,
  * leave-one-*group*-out CV that keeps aviaries 5 and 6 in the same fold.

Model
-----
    log N = alpha_species + beta * z(feature)

3 species intercepts + 1 shared slope = 4 parameters over 8 points. Intercepts
are unpenalised (they encode real biology: a quelea flock and an ibis pair
simply are different scales); the slope is ridge-penalised and sign-constrained.

Usage:
    python -m src.model --features results/features_dev.csv --out results/
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEV_LABELS, CV_GROUPS
from .features import FEATURE_COLS


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class PooledLogLinear:
    """log N = alpha_species + sum_k beta_k * z_k, ridge on beta, sign-constrained."""

    def __init__(self, alpha_ridge: float = 1.0, enforce_positive: bool = True):
        self.alpha_ridge = alpha_ridge
        self.enforce_positive = enforce_positive

    def fit(self, X, species, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ < 1e-9] = 1.0
        Z = (X - self.mu_) / self.sd_

        self.species_ = sorted(set(species))
        S = np.zeros((len(y), len(self.species_)))
        for i, s in enumerate(species):
            S[i, self.species_.index(s)] = 1.0

        A = np.hstack([S, Z])
        b = np.log(y)
        # ridge on the slope columns only
        k = Z.shape[1]
        pen = np.zeros((k, A.shape[1]))
        pen[:, len(self.species_):] = np.sqrt(self.alpha_ridge) * np.eye(k)
        A_aug = np.vstack([A, pen])
        b_aug = np.concatenate([b, np.zeros(k)])

        coef, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        self.intercepts_ = dict(zip(self.species_, coef[:len(self.species_)]))
        self.beta_ = coef[len(self.species_):]

        if self.enforce_positive and np.any(self.beta_ < 0):
            # Monotonicity prior: acoustic activity cannot be negatively related
            # to abundance. Zeroing a negative slope collapses to species-mean,
            # which is the correct conservative behaviour with n=8.
            self.beta_ = np.clip(self.beta_, 0.0, None)
            self._refit_intercepts(Z, species, b)

        self.global_intercept_ = float(np.mean(list(self.intercepts_.values())))
        # residual sd in log space -> used for prediction intervals
        pred = self._predict_log(Z, species)
        dof = max(len(y) - (len(self.species_) + k), 1)
        self.sigma_ = float(np.sqrt(np.sum((b - pred) ** 2) / dof))
        return self

    def _refit_intercepts(self, Z, species, b):
        adj = b - Z @ self.beta_
        for s in self.species_:
            m = np.array([sp == s for sp in species])
            if m.any():
                self.intercepts_[s] = float(adj[m].mean())

    def _predict_log(self, Z, species):
        a = np.array([self.intercepts_.get(s, self.global_intercept_) for s in species])
        return a + Z @ self.beta_

    def predict(self, X, species):
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.exp(self._predict_log(Z, species))

    def predict_interval(self, X, species, z=1.96):
        mu = self._predict_log((np.asarray(X, float) - self.mu_) / self.sd_, species)
        return np.exp(mu - z * self.sigma_), np.exp(mu + z * self.sigma_)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class SpeciesMean:
    """Predict the geometric mean count of that species in the training folds.

    This is the baseline that matters. Any feature-based model that cannot beat
    it under leave-one-aviary-out is not extracting abundance information from
    the audio -- it is exploiting the species identity you already knew.
    """

    def fit(self, X, species, y):
        y = np.asarray(y, float)
        self.by_sp_, self.global_ = {}, float(np.exp(np.mean(np.log(y))))
        for s in set(species):
            m = np.array([sp == s for sp in species])
            self.by_sp_[s] = float(np.exp(np.mean(np.log(y[m]))))
        return self

    def predict(self, X, species):
        return np.array([self.by_sp_.get(s, self.global_) for s in species])


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.maximum(y_true, 1)) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def loo_predict(df, feats, model_factory):
    """Leave-one-group-out predictions. Groups keep aviaries 5+6 together."""
    groups = df["group"].to_numpy()
    preds = np.full(len(df), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(set(df.loc[tr, "species"])) == 0 or tr.sum() < 3:
            continue
        m = model_factory().fit(df.loc[tr, feats].to_numpy(),
                                df.loc[tr, "species"].tolist(),
                                df.loc[tr, "count"].to_numpy())
        preds[te] = m.predict(df.loc[te, feats].to_numpy(),
                              df.loc[te, "species"].tolist())
    return preds


def nested_feature_selection(df, candidate_feats, max_k=2, alpha_ridge=1.0):
    """Choose features INSIDE each outer fold, then score. No label leakage.

    Reporting a score from features picked on the full 8 points would be
    meaningless -- with 20 candidates and 8 points you can always find one that
    fits. The nested number below is the one to believe.
    """
    groups = df["group"].to_numpy()
    combos = []
    for k in range(1, max_k + 1):
        combos.extend(combinations(candidate_feats, k))

    preds = np.full(len(df), np.nan)
    chosen = []
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        inner = df.loc[tr].reset_index(drop=True)
        best, best_mae = None, np.inf
        for c in combos:
            p = loo_predict(inner, list(c),
                            lambda: PooledLogLinear(alpha_ridge))
            ok = ~np.isnan(p)
            if ok.sum() < 3:
                continue
            mae = np.mean(np.abs(p[ok] - inner.loc[ok, "count"].to_numpy()))
            if mae < best_mae:
                best_mae, best = mae, c
        if best is None:
            best = (candidate_feats[0],)
        chosen.append((g, best, round(best_mae, 2)))
        m = PooledLogLinear(alpha_ridge).fit(
            inner[list(best)].to_numpy(), inner["species"].tolist(),
            inner["count"].to_numpy())
        preds[te] = m.predict(df.loc[te, list(best)].to_numpy(),
                              df.loc[te, "species"].tolist())
    return preds, chosen


def load_dataset(features_csv):
    feats = pd.read_csv(features_csv)
    lab = pd.DataFrame(DEV_LABELS, columns=["aviary_id", "species", "count"])
    df = lab.merge(feats, on=["aviary_id", "species"], how="left")
    missing = df[df["n_clips"].isna()]
    if len(missing):
        print("WARNING: no features for:\n", missing[["aviary_id", "species"]])
        df = df.dropna(subset=["n_clips"])
    df["group"] = df["aviary_id"].map(CV_GROUPS)
    return df.reset_index(drop=True)


def main(features_csv, outdir, alpha_ridge=1.0, max_k=2):
    df = load_dataset(features_csv)
    avail = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().all()]
    for c in df.columns:
        if c.startswith("bn_") and df[c].notna().all():
            avail.append(c)
    print(f"{len(df)} labelled points, {len(avail)} usable features\n")

    results = {}

    # --- baselines -----------------------------------------------------------
    p = loo_predict(df, avail[:1], lambda: SpeciesMean())
    results["baseline_species_mean"] = metrics(df["count"], p)
    df["pred_species_mean"] = p

    for f in ["activity_rate", "bn_det_rate"]:
        if f in df.columns:
            p = loo_predict(df, [f], lambda: PooledLogLinear(alpha_ridge))
            results[f"single_{f}"] = metrics(df["count"], p)

    # --- optimistic (features picked on all data) - reported for contrast -----
    single_scores = {}
    for f in avail:
        p = loo_predict(df, [f], lambda: PooledLogLinear(alpha_ridge))
        single_scores[f] = metrics(df["count"], p)["MAE"]
    ranked = sorted(single_scores.items(), key=lambda kv: kv[1])
    print("Per-feature LOO MAE (optimistic - feature chosen using all folds):")
    for f, v in ranked[:10]:
        print(f"   {f:24s} {v:8.2f}")
    best_feat = ranked[0][0]
    p_opt = loo_predict(df, [best_feat], lambda: PooledLogLinear(alpha_ridge))
    results["optimistic_best_single"] = metrics(df["count"], p_opt)
    results["optimistic_best_single"]["feature"] = best_feat

    # --- honest nested selection --------------------------------------------
    p_nest, chosen = nested_feature_selection(df, avail, max_k, alpha_ridge)
    results["nested_selection"] = metrics(df["count"], p_nest)
    df["pred_nested"] = p_nest
    print("\nFeatures chosen per outer fold (nested):")
    for g, c, mae in chosen:
        print(f"   held out {g}: {c}  (inner MAE {mae})")

    # --- final model on all data --------------------------------------------
    final_feats = list(chosen[0][1])
    final = PooledLogLinear(alpha_ridge).fit(
        df[final_feats].to_numpy(), df["species"].tolist(), df["count"].to_numpy())

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "dev_predictions.csv", index=False)
    with open(outdir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(outdir / "model.json", "w") as fh:
        json.dump({
            "features": final_feats,
            "mu": final.mu_.tolist(), "sd": final.sd_.tolist(),
            "beta": final.beta_.tolist(),
            "intercepts": final.intercepts_,
            "global_intercept": final.global_intercept_,
            "sigma_log": final.sigma_,
            "alpha_ridge": alpha_ridge,
        }, fh, indent=2)

    print("\n" + "=" * 62)
    print(f"{'model':32s} {'MAE':>8s} {'RMSE':>8s} {'MAPE%':>8s}")
    print("=" * 62)
    for k, v in results.items():
        print(f"{k:32s} {v['MAE']:8.2f} {v['RMSE']:8.2f} {v['MAPE_%']:8.1f}")
    print("=" * 62)
    print("\nPer-point (honest, nested):")
    show = df[["aviary_id", "species", "count", "pred_nested", "pred_species_mean"]]
    print(show.round(1).to_string(index=False))
    print(f"\nwrote {outdir}/metrics.json, model.json, dev_predictions.csv")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--alpha-ridge", type=float, default=1.0)
    ap.add_argument("--max-k", type=int, default=2)
    a = ap.parse_args()
    main(a.features, a.out, a.alpha_ridge, a.max_k)
