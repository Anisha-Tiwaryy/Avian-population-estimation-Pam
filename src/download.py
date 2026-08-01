"""
Subsampled download of the BioDCASE 2026 Bird Counting dataset.

The full repo is ~287 GB, which is not a sensible thing to pull for a
time-boxed experiment. This module downloads a *temporally stratified random
sample* of clips per aviary instead.

Why stratified-random rather than "the first chunk":
    Clips are consecutive segments of continuous recordings and the chunk index
    is monotonic in time. Taking chunk_000 would give you a few contiguous hours
    (typically the middle of one night), which destroys the diel activity
    structure that the counting features depend on. We therefore enumerate every
    file, parse day + time-of-day from the filename, and sample uniformly across
    (day x hour) cells.

Downloads run in a thread pool. Each file is only ~528 KB, so a serial download
spends most of its wall-clock time on per-request connection setup rather than
on transfer; 16 concurrent workers typically gives a 6-10x speedup on a home
connection. Files already present locally are skipped, so an interrupted run can
simply be restarted.

Usage:
    python -m src.download --split dev --per-aviary 400 --out data/raw
    python -m src.download --split dev --per-aviary 400 --out data/raw --workers 24
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import HF_REPO_ID

# Silence the per-file progress bars; we render one bar for the whole job.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

FNAME_RE = re.compile(
    r"rec_(?P<day>d\d+)_(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2}(?:\.\d+)?)\.wav$"
)


def parse_filename(name: str):
    """rec_d1_19_05_02.500000.wav -> ('d1', 68702.5). Returns None if unparsable."""
    m = FNAME_RE.search(name)
    if not m:
        return None
    tod = int(m["h"]) * 3600 + int(m["m"]) * 60 + float(m["s"])
    return m["day"], tod


def list_aviary_files(repo_files, aviary_id):
    return [f for f in repo_files if f.startswith(f"{aviary_id}/") and f.endswith(".wav")]


def stratified_sample(files, n, seed=0):
    """Sample n files spread evenly over (day, hour-of-day) cells."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for f in files:
        parsed = parse_filename(Path(f).name)
        if parsed is None:
            continue
        day, tod = parsed
        cells[(day, int(tod // 3600))].append(f)

    if not cells:                       # fall back to plain random
        return rng.sample(files, min(n, len(files)))

    for v in cells.values():
        rng.shuffle(v)

    keys = sorted(cells)
    picked, i = [], 0
    # round-robin over cells so every hour of every day is represented
    while len(picked) < n and any(cells[k] for k in keys):
        k = keys[i % len(keys)]
        if cells[k]:
            picked.append(cells[k].pop())
        i += 1
    return picked


def _fetch(remote_path: str, out: str, retries: int = 3):
    """Download one file, skipping it if already present. Returns local path."""
    from huggingface_hub import hf_hub_download

    local_guess = Path(out) / remote_path
    if local_guess.exists() and local_guess.stat().st_size > 0:
        return str(local_guess)

    last = None
    for attempt in range(retries):
        try:
            return hf_hub_download(HF_REPO_ID, remote_path, repo_type="dataset",
                                   local_dir=out)
        except Exception as exc:                     # transient network error
            last = exc
    print(f"  failed after {retries} tries: {remote_path} ({last})")
    return None


def download(split="dev", per_aviary=400, out="data/raw", seed=0,
             aviaries=None, workers=16):
    from huggingface_hub import HfApi
    from tqdm import tqdm

    api = HfApi()
    print(f"Listing files in {HF_REPO_ID} (several minutes, be patient)...")
    repo_files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")

    prefix = "dev_aviary_" if split == "dev" else "eval_aviary_"
    found = sorted({f.split("/")[0] for f in repo_files if f.startswith(prefix)},
                   key=lambda s: int(s.rsplit("_", 1)[1]))
    if aviaries:
        found = [a for a in found if a in aviaries]

    # metadata is small - always grab it all, serially
    print("Fetching metadata...")
    for f in [f for f in repo_files if f.startswith("metadata/")]:
        _fetch(f, out)

    # Build the full job list first so one progress bar covers the whole run.
    jobs = []
    for aviary in found:
        files = list_aviary_files(repo_files, aviary)
        picked = stratified_sample(files, per_aviary, seed=seed)
        print(f"{aviary}: {len(files):,} available -> sampling {len(picked):,}")
        jobs.extend((aviary, f) for f in picked)

    print(f"\nDownloading {len(jobs):,} clips with {workers} workers...")
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, f, out): (aviary, f) for aviary, f in jobs}
        with tqdm(total=len(futs), unit="clip", smoothing=0.05) as bar:
            for fut in as_completed(futs):
                aviary, _ = futs[fut]
                local = fut.result()
                if local:
                    manifest_rows.append((aviary, local))
                bar.update(1)

    man = Path(out) / f"manifest_{split}.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w") as fh:
        fh.write("aviary_id,path\n")
        for a, p in sorted(manifest_rows):
            fh.write(f"{a},{p}\n")
    print(f"\nWrote {man} ({len(manifest_rows):,} clips)")
    if len(manifest_rows) < len(jobs):
        print(f"note: {len(jobs) - len(manifest_rows)} clips failed and were "
              f"omitted from the manifest; re-run to retry them")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--per-aviary", type=int, default=400)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aviaries", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    download(a.split, a.per_aviary, a.out, a.seed, a.aviaries, a.workers)
