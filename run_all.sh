#!/usr/bin/env bash
# End-to-end reproduction. Assumes `pip install -r requirements.txt` has run.
set -euo pipefail

PER_AVIARY_DEV=${PER_AVIARY_DEV:-1500}
PER_AVIARY_EVAL=${PER_AVIARY_EVAL:-800}
BACKEND=${BACKEND:-dsp}          # dsp | birdnet | both

mkdir -p data/raw results figures

echo "=== [1/6] downloading stratified subsample (dev) ==="
[ -f data/raw/manifest_dev.csv ] || \
  python -m src.download --split dev --per-aviary "$PER_AVIARY_DEV" --out data/raw

echo "=== [2/6] per-clip acoustic measurement ==="
python -m src.detect --manifest data/raw/manifest_dev.csv \
                     --out results/clip_measurements_dev.parquet --backend "$BACKEND"

echo "=== [3/6] per-aviary features ==="
python -m src.features --clips results/clip_measurements_dev.parquet \
                       --out results/features_dev.csv

echo "=== [4/6] model fit + leave-one-aviary-out evaluation ==="
python -m src.model --features results/features_dev.csv --out results/ \
                    | tee results/model_log.txt

echo "=== [5/6] figures ==="
python -m src.eda --clips results/clip_measurements_dev.parquet \
                  --features results/features_dev.csv \
                  --predictions results/dev_predictions.csv --out figures/

echo "=== [6/6] evaluation-set inference (optional) ==="
if [ "${RUN_EVAL:-0}" = "1" ]; then
  python -m src.download --split eval --per-aviary "$PER_AVIARY_EVAL" --out data/raw
  python -m src.detect   --manifest data/raw/manifest_eval.csv \
                         --out results/clip_measurements_eval.parquet --backend "$BACKEND"
  python -m src.features --clips results/clip_measurements_eval.parquet \
                         --out results/features_eval.csv
  python -m src.predict  --features results/features_eval.csv \
                         --model results/model.json --out results/submission.csv
else
  echo "skipped (set RUN_EVAL=1 to enable)"
fi

echo "=== done. see results/metrics.json and figures/ ==="
