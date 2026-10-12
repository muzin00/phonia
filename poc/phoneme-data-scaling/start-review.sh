#!/bin/sh
set -eu
task_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_root=$(CDPATH= cd -- "$task_script_dir/../.." && pwd)
task_run="$task_root/artifacts/phoneme-data-scaling/review100-20261012-v1"
export PHONIA_REVIEW_DATASET="$task_run/review-dataset.json"
export PHONIA_REVIEW_MEDIA_ROOT="$task_run/media"
export PHONIA_REVIEW_OUTPUT="$task_run/review-records.jsonl"
unset PHONIA_REVIEW_MODE
if [ ! -f "$PHONIA_REVIEW_DATASET" ]; then
  echo "Prepare the fixed review dataset first; see review.md." >&2
  exit 1
fi
cd "$task_root/tools/phoneme-review-ui"
exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5179 --strictPort
