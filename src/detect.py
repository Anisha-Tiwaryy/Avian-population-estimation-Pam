"""
Stage 1 - per-clip acoustic measurement.

For every 3-second clip we compute, for every target species' analysis band, a
small set of measurements that downstream aggregation turns into abundance
features. Deliberately *not* just "did the detector fire": see report §3.

Measurements per (clip, species band)
-------------------------------------
peak_db      Max over frames of in-band energy. Tracks presence/intensity of the
             loudest call in the clip. Saturates quickly with flock size.
floor_db     10th percentile over frames of in-band energy -- the "chorus floor".
             In a large flock the between-call silence is filled by other
             individuals, so the floor rises even after peak_db has saturated.
             This is the single most informative feature we found for flock
             species (report §5).
occupancy    Fraction of frames whose in-band energy is within 6 dB of the clip
             peak. High for continuous chattering, low for isolated calls.
onset_rate   In-band spectral-flux onsets per second. Proxy for call rate.
flatness     Wiener entropy inside the band on active frames. Many overlapping
             voices -> flatter (more noise-like) spectrum. Polyphony proxy.
n_peaks      Mean number of distinct spectral peaks in the band on active
             frames. Second polyphony proxy; more robust than flatness to
             broadband noise.
snr_db       peak_db minus an out-of-band reference level. Makes the measurement
             robust to overall gain / distance differences between aviaries.

The DSP backend is unsupervised and has no learned parameters, which matters a
lot here: with 8 labelled data points, any per-clip supervised training would
memorise aviary identity rather than learn abundance.

An optional BirdNET backend adds a per-clip species confidence column; the
downstream code uses it if present and falls back to the DSP measurements if
not. `--backend both` is the recommended run.

Usage:
    python -m src.detect --manifest data/raw/manifest_dev.csv \
                         --out results/clip_measurements_dev.parquet --backend dsp
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from .config import SPECIES, SAMPLE_RATE
from .download import parse_filename

warnings.filterwarnings("ignore", category=RuntimeWarning)

WORK_SR = 16_000          # decimate 48k -> 16k; covers all bands (max 8 kHz)
NFFT = 1024               # 64 ms window
HOP = 256                 # 16 ms hop
EPS = 1e-12
OUT_OF_BAND = (100, 400)   # low-frequency reference used for SNR normalisation


def _load(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != WORK_SR:
        if sr % WORK_SR == 0:
            x = signal.decimate(x, sr // WORK_SR, ftype="fir", zero_phase=True)
        else:
            n = int(round(len(x) * WORK_SR / sr))
            x = signal.resample(x, n)
    return np.asarray(x, dtype=np.float32)


def _spectrogram(x: np.ndarray):
    f, _, Z = signal.stft(x, fs=WORK_SR, nperseg=NFFT, noverlap=NFFT - HOP,
                          window="hann", padded=False, boundary=None)
    P = (np.abs(Z) ** 2).astype(np.float32)
    return f, P


def _band_measurements(f, P, band):
    """All per-band measurements for one clip. P is power [freq, time]."""
    lo, hi = band
    sel = (f >= lo) & (f <= hi)
    if sel.sum() < 3 or P.shape[1] < 4:
        return None
    B = P[sel]                                    # [band_bins, frames]
    energy = B.sum(axis=0) + EPS                  # per-frame in-band energy
    e_db = 10.0 * np.log10(energy)

    peak_db = float(np.max(e_db))
    floor_db = float(np.percentile(e_db, 10))
    med_db = float(np.median(e_db))
    occupancy = float(np.mean(e_db > peak_db - 6.0))

    # out-of-band reference for gain normalisation
    ref_sel = (f >= OUT_OF_BAND[0]) & (f <= OUT_OF_BAND[1])
    ref_db = float(np.median(10 * np.log10(P[ref_sel].sum(axis=0) + EPS))) \
        if ref_sel.sum() else 0.0
    snr_db = peak_db - ref_db

    # onsets: half-wave-rectified spectral flux inside the band
    flux = np.diff(e_db, prepend=e_db[0])
    flux = np.maximum(flux, 0.0)
    thr = flux.mean() + 2.0 * flux.std()
    peaks, _ = signal.find_peaks(flux, height=max(thr, 1.0),
                                 distance=max(1, int(0.08 * WORK_SR / HOP)))
    onset_rate = len(peaks) / (P.shape[1] * HOP / WORK_SR)

    # polyphony proxies, measured only on frames that actually contain energy
    active = e_db > (med_db + 3.0)
    if active.sum() < 2:
        active = e_db >= np.percentile(e_db, 75)
    Ba = B[:, active] + EPS
    gm = np.exp(np.mean(np.log(Ba), axis=0))
    am = np.mean(Ba, axis=0)
    flatness = float(np.mean(gm / am))

    n_peaks = []
    for j in range(Ba.shape[1]):
        col = 10 * np.log10(Ba[:, j])
        pk, _ = signal.find_peaks(col, prominence=6.0)
        n_peaks.append(len(pk))
    n_peaks = float(np.mean(n_peaks)) if n_peaks else 0.0

    return dict(peak_db=peak_db, floor_db=floor_db, med_db=med_db,
                occupancy=occupancy, snr_db=snr_db, onset_rate=onset_rate,
                flatness=flatness, n_peaks=n_peaks)


def measure_clip(args):
    aviary_id, path = args
    try:
        x = _load(path)
    except Exception as exc:                       # corrupt / truncated file
        return [{"aviary_id": aviary_id, "path": path, "error": str(exc)}]
    if len(x) < NFFT * 2:
        return [{"aviary_id": aviary_id, "path": path, "error": "too short"}]

    f, P = _spectrogram(x)
    broadband_db = float(10 * np.log10(P.sum() / P.shape[1] + EPS))
    parsed = parse_filename(Path(path).name)
    day, tod = parsed if parsed else ("d?", np.nan)

    rows = []
    for key, sp in SPECIES.items():
        m = _band_measurements(f, P, sp.band_hz)
        if m is None:
            continue
        rows.append({"aviary_id": aviary_id, "path": path, "day": day,
                     "tod_s": tod, "species": key,
                     "broadband_db": broadband_db, **m})
    return rows


def run_dsp(manifest_csv: str, workers: int = 0) -> pd.DataFrame:
    man = pd.read_csv(manifest_csv)
    jobs = list(man[["aviary_id", "path"]].itertuples(index=False, name=None))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    out = []
    from tqdm import tqdm
    if workers == 1:
        for j in tqdm(jobs, desc="measuring"):
            out.extend(measure_clip(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in tqdm(ex.map(measure_clip, jobs, chunksize=32),
                             total=len(jobs), desc="measuring"):
                out.extend(rows)
    df = pd.DataFrame(out)
    if "error" in df.columns:
        bad = df["error"].notna()
        if bad.any():
            print(f"warning: {int(bad.sum())} clips failed to read and were skipped")
        df = df[~bad].drop(columns=["error"])
    return df


def run_birdnet(manifest_csv: str, min_conf: float = 0.05) -> pd.DataFrame:
    """Optional: per-clip BirdNET confidence for each target species.

    Requires `pip install birdnetlib tensorflow`. Kept fully optional so the
    pipeline runs end-to-end on a laptop with no model download.
    """
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    from tqdm import tqdm

    analyzer = Analyzer()
    alias_to_key = {}
    for key, sp in SPECIES.items():
        for a in (sp.common_name, sp.scientific_name, *sp.detector_aliases):
            alias_to_key[a.lower()] = key

    man = pd.read_csv(manifest_csv)
    rows = []
    for aviary_id, path in tqdm(
            man[["aviary_id", "path"]].itertuples(index=False, name=None),
            total=len(man), desc="birdnet"):
        try:
            rec = Recording(analyzer, path, min_conf=min_conf)
            rec.analyze()
        except Exception:
            continue
        best = {}
        for d in rec.detections:
            for name in (d.get("common_name", ""), d.get("scientific_name", "")):
                k = alias_to_key.get(str(name).lower())
                if k:
                    best[k] = max(best.get(k, 0.0), float(d["confidence"]))
        for key in SPECIES:
            rows.append({"aviary_id": aviary_id, "path": path, "species": key,
                         "bn_conf": best.get(key, 0.0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["dsp", "birdnet", "both"], default="dsp")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df = None
    if a.backend in ("dsp", "both"):
        df = run_dsp(a.manifest, a.workers)
    if a.backend in ("birdnet", "both"):
        bn = run_birdnet(a.manifest)
        df = bn if df is None else df.merge(
            bn, on=["aviary_id", "path", "species"], how="left")

    if a.out.endswith(".parquet"):
        try:
            df.to_parquet(a.out, index=False)
        except Exception:
            a.out = a.out.replace(".parquet", ".csv")
            df.to_csv(a.out, index=False)
    else:
        df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  rows={len(df):,}")
