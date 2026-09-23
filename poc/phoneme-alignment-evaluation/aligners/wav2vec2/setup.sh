#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"
model_revision="2c733782da5604684829819a5eb744c193fe9398"
cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
data_base="${XDG_DATA_HOME:-${HOME}/.local/share}"
source_dir="${PHONIA_WAV2VEC2_SOURCE_DIR:-${cache_base}/phonia/wav2vec2/source-${model_revision}}"
runtime_dir="${PHONIA_WAV2VEC2_RUNTIME_DIR:-${data_base}/phonia/wav2vec2/${model_revision}/onnx}"
source_sha256="04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551"
onnx_sha256="fdfecb377f173a45d92502d698558521bb0107c118ef151d6896a2f76987c65a"

uv run --project "${project_dir}" --group wav2vec2 \
  hf download facebook/wav2vec2-xlsr-53-espeak-cv-ft \
  --revision "${model_revision}" \
  --local-dir "${source_dir}"

printf '%s  %s\n' "${source_sha256}" "${source_dir}/pytorch_model.bin" \
  | shasum -a 256 --check

if [[ ! -f "${runtime_dir}/model.onnx" ]]; then
  mkdir -p "${runtime_dir}"
  uv run --project "${project_dir}" --group wav2vec2 \
    optimum-cli export onnx \
    --model "${source_dir}" \
    --task automatic-speech-recognition \
    --dtype fp32 \
    "${runtime_dir}"
fi

printf '%s  %s\n' "${onnx_sha256}" "${runtime_dir}/model.onnx" \
  | shasum -a 256 --check

echo "Wav2Vec2 source: ${source_dir}"
echo "Wav2Vec2 ONNX: ${runtime_dir}/model.onnx"
