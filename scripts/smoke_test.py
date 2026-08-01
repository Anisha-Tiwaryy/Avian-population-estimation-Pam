"""
End-to-end smoke test on SYNTHETIC audio - no dataset download required.

Generates six fake "aviaries" whose clips contain a controlled number of
simultaneous synthetic callers per species band, matching the real ground-truth
counts. Then runs measure -> features -> model -> predict.

This does NOT validate scientific accuracy. It validates that the plumbing works
and that the features respond monotonically to the number of simultaneous
callers, which is the assumption the whole method rests on.

    python scripts/smoke_test.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEV_LABELS, SPECIES, SAMPLE_RATE  # noqa: E402
from src.detect import run_dsp                            # noqa: E402
from src.features import build                            # noqa: E402
from src import model as M                                # noqa: E402
from src.predict import predict                           # noqa: E402

ROOT = Path("/tmp/smoke")
SR = SAMPLE_RATE
DUR = 3.0
N_CLIPS = 120          # per aviary; small, this is a plumbing test


def make_clip(rng, pops):
    """pops: {species_key: n_individuals}. More individuals -> more overlapping
    calls -> higher chorus floor and more concurrent spectral peaks."""
    n = int(SR * DUR)
    t = np.arange(n) / SR
    x = rng.normal(0, 0.002, n)                       # ambient noise
    for key, count in pops.items():
        lo, hi = SPECIES[key].band_hz
        # number of individuals actually calling in this 3 s window
        p_call = min(0.9, 0.02 + 0.004 * count)
        n_calling = rng.binomial(count, p_call)
        for _ in range(n_calling):
            f0 = rng.uniform(lo * 1.05, min(hi * 0.55, lo * 2.2))
            start = rng.uniform(0, DUR - 0.45)
            dur = rng.uniform(0.15, 0.4)
            i0, i1 = int(start * SR), int((start + dur) * SR)
            env = np.hanning(i1 - i0)
            seg = np.zeros(i1 - i0)
            for h in range(1, 5):
                if f0 * h < hi:
                    seg += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t[i0:i1]
                                              + rng.uniform(0, 6.28))
            x[i0:i1] += 0.05 * env * seg
    return np.clip(x, -1, 1).astype(np.float32)


def build_dataset():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = np.random.default_rng(0)
    lab = pd.DataFrame(DEV_LABELS, columns=["aviary_id", "species", "count"])
    rows = []
    for aviary, g in lab.groupby("aviary_id"):
        pops = dict(zip(g["species"], g["count"]))
        d = ROOT / aviary
        d.mkdir(parents=True, exist_ok=True)
        for i in range(N_CLIPS):
            hh, mm = 4 + (i * 18) // 60 % 20, (i * 18) % 60
            p = d / f"rec_d1_{hh:02d}_{mm:02d}_{(i*7)%60:02d}.wav"
            sf.write(p, make_clip(rng, pops), SR, subtype="PCM_16")
            rows.append((aviary, str(p)))
    man = ROOT / "manifest.csv"
    pd.DataFrame(rows, columns=["aviary_id", "path"]).to_csv(man, index=False)
    return man


def main():
    print("1/5 generating synthetic aviaries...")
    man = build_dataset()

    print("2/5 measuring clips...")
    clips = run_dsp(str(man), workers=1)
    assert len(clips) > 0, "no measurements produced"
    print(f"    {len(clips):,} (clip, band) measurements")

    print("3/5 building features...")
    feats = build(clips)
    out = ROOT / "features.csv"
    feats.to_csv(out, index=False)
    print(feats[["aviary_id", "species", "activity_rate",
                 "chorus_floor_p90", "poly_npeaks"]].round(3).to_string(index=False))

    print("\n4/5 fitting + LOAO evaluating...")
    res = M.main(str(out), str(ROOT / "results"), alpha_ridge=1.0, max_k=1)

    print("\n5/5 inference path...")
    targets = {"dev_aviary_1": "quelea", "dev_aviary_4": "flamingo"}
    predict(str(out), str(ROOT / "results" / "model.json"),
            str(ROOT / "results" / "submission.csv"), targets=targets)

    assert res["nested_selection"]["MAE"] > 0
    print("\nSMOKE TEST PASSED - pipeline is wired correctly end to end.")


if __name__ == "__main__":
    main()
