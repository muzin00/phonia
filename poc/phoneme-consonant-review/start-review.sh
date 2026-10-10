#!/bin/sh
set -eu

task_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_root=$(CDPATH= cd -- "$task_script_dir/../.." && pwd)

export PHONIA_REVIEW_DATASET="$task_script_dir/data/review/review-dataset.json"
export PHONIA_REVIEW_OUTPUT="$task_script_dir/data/review/review-records.jsonl"
export PHONIA_REVIEW_MEDIA_ROOT="$task_root/artifacts/phoneme-consonant-review/mn-100-20261010-v1/media"

if [ ! -f "$PHONIA_REVIEW_DATASET" ] || [ ! -d "$PHONIA_REVIEW_MEDIA_ROOT" ]; then
  echo "Run: python3 $task_script_dir/sample_review.py" >&2
  exit 1
fi

cd "$task_root/tools/phoneme-review-ui"
exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort
