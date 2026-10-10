#!/bin/sh
set -eu

task_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_root=$(CDPATH= cd -- "$task_script_dir/../.." && pwd)

case "${1:-jvs}" in
  jvs) task_corpus=jvs; task_data=jvs-review; task_port=5176 ;;
  common-voice) task_corpus=common-voice; task_data=common-voice-review; task_port=5177 ;;
  *) echo "Usage: sh $0 [jvs|common-voice]" >&2; exit 1 ;;
esac

export PHONIA_REVIEW_DATASET="$task_script_dir/data/$task_data/review-dataset.json"
export PHONIA_REVIEW_OUTPUT="$task_script_dir/data/$task_data/review-records.jsonl"
export PHONIA_REVIEW_MEDIA_ROOT="$task_root/artifacts/phoneme-s-review/s-200-20261010-v1/$task_corpus/media"

if [ ! -f "$PHONIA_REVIEW_DATASET" ] || [ ! -d "$PHONIA_REVIEW_MEDIA_ROOT" ]; then
  echo "Run: poc/phoneme-verification-evaluation/.venv/bin/python $task_script_dir/prepare_review.py" >&2
  exit 1
fi

cd "$task_root/tools/phoneme-review-ui"
exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port "$task_port" --strictPort
