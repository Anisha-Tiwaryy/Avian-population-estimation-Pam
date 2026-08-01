"""
Exploratory analysis + the four figures used in the technical report.

    python -m src.eda --clips results/clip_measurements_dev.parquet \
                      --features results/features_dev.csv \
                      --predictions results/dev_predictions.csv \
                      --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEV_LABELS, SPECIES
from .features import FEATURE_COLS, _read

COLORS = {"flamingo": "#d1495b", "quelea": "#00798c", "ibis": "#edae49",
          "avocet": "#66a182"}


def fig_saturation(feats, out):
    """The core argument: activity saturates, the chorus floor does not."""
    lab = pd.DataFrame(DEV_LABELS, columns=["aviary_id", "species", "count"])
    df = lab.merge(feats, on=["aviary_id", "species"])
    pairs = [("activity_rate", "Clip activity rate (saturating)"),
             ("chorus_floor_p90", "Chorus floor elevation, dB (non-saturating)"),
             ("poly_npeaks", "Mean concurrent spectral peaks (polyphony)")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4))
    for ax, (col, title) in zip(axes, pairs):
        if col not in df:
            continue
        for sp, g in df.groupby("species"):
            ax.scatter(g[col], g["count"], s=70, label=SPECIES[sp].common_name,
                       color=COLORS.get(sp, "grey"), edgecolor="k", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel("True population")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Why detection rate alone cannot count a flock", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig1_saturation.png", dpi=160)
    plt.close(fig)


def fig_diel(clips, out):
    """Activity by hour of day per aviary - sanity check that the subsample
    preserved diel structure, and a look at species-specific rhythms."""
    c = clips.copy()
    c["hour"] = (c["tod_s"] // 3600).astype("Int64")
    aviaries = sorted(c.aviary_id.unique())
    fig, axes = plt.subplots(len(aviaries), 1, figsize=(9, 1.6 * len(aviaries)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, av in zip(axes, aviaries):
        sub = c[c.aviary_id == av]
        for sp, g in sub.groupby("species"):
            base = np.percentile(g["peak_db"], 10)
            act = (g["peak_db"] - base) > 8
            r = act.groupby(g["hour"]).mean()
            ax.plot(r.index, r.values, label=sp, color=COLORS.get(sp, "grey"))
        ax.set_ylabel(av.replace("dev_", ""), fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("Diel activity by band, per aviary", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig2_diel.png", dpi=160)
    plt.close(fig)


def fig_pred(preds, out):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [1, max(preds["count"].max(), preds["pred_nested"].max()) * 1.4]
    ax.plot(lim, lim, "k--", lw=1, zorder=1)
    for sp, g in preds.groupby("species"):
        ax.scatter(g["count"], g["pred_nested"], s=80,
                   color=COLORS.get(sp, "grey"), edgecolor="k",
                   label=SPECIES[sp].common_name, zorder=3)
        if "pred_species_mean" in g:
            ax.scatter(g["count"], g["pred_species_mean"], s=40, marker="x",
                       color=COLORS.get(sp, "grey"), alpha=.6, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True population"); ax.set_ylabel("Predicted (held-out)")
    ax.set_title("Leave-one-aviary-out predictions\n(x = species-mean baseline)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(Path(out) / "fig3_predictions.png", dpi=160)
    plt.close(fig)


def fig_corr(feats, out):
    lab = pd.DataFrame(DEV_LABELS, columns=["aviary_id", "species", "count"])
    df = lab.merge(feats, on=["aviary_id", "species"])
    cols = [c for c in FEATURE_COLS if c in df.columns]
    r = [np.corrcoef(df[c], np.log(df["count"]))[0, 1] for c in cols]
    order = np.argsort(r)
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(cols) + 1.5))
    ax.barh([cols[i] for i in order], [r[i] for i in order],
            color=["#00798c" if r[i] > 0 else "#d1495b" for i in order])
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Pearson r with log(population), n=8")
    ax.set_title("Feature correlation with abundance\n"
                 "(n=8: treat as a sanity check, not evidence)", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / "fig4_feature_corr.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips")
    ap.add_argument("--features", required=True)
    ap.add_argument("--predictions")
    ap.add_argument("--out", default="figures/")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(a.features)
    fig_saturation(feats, a.out)
    fig_corr(feats, a.out)
    if a.clips:
        fig_diel(_read(a.clips), a.out)
    if a.predictions:
        fig_pred(pd.read_csv(a.predictions), a.out)
    print(f"figures written to {a.out}")
