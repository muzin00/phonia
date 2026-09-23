#!/usr/bin/env bash
set -euo pipefail

JULIUS_VERSION="4.6"
JULIUS_COMMIT="3b7174d0d4091f5e6ebb917769822032d079996f"
KIT_COMMIT="e0e8bbaf98e27d19dfc6fe8312be607ad03592ad"
MODEL_SHA256="58952ccfe60f283c7efb1c22f9e012c7d297429700b7eeb4b60f7651a6e65940"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/phonia/julius"
INSTALL_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/phonia/julius"
SOURCE_DIR="${CACHE_ROOT}/source-v${JULIUS_VERSION}"
INSTALL_DIR="${INSTALL_ROOT}/${JULIUS_VERSION}"
KIT_DIR="${INSTALL_ROOT}/segmentation-kit"
JULIUS_BIN="${INSTALL_DIR}/bin/julius"
PATCH_FILE="${SCRIPT_DIR}/patches/julius-4.6-apple-silicon.patch"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup script currently supports Apple Silicon macOS only." >&2
  exit 1
fi

for command in git make gcc perl shasum; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command was not found: ${command}" >&2
    exit 1
  fi
done

mkdir -p "${CACHE_ROOT}" "${INSTALL_ROOT}"

if [[ ! -x "${JULIUS_BIN}" ]]; then
  if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --depth 1 --branch "v${JULIUS_VERSION}" \
      https://github.com/julius-speech/julius.git "${SOURCE_DIR}"
  fi
  if [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" != "${JULIUS_COMMIT}" ]]; then
    echo "Unexpected Julius source commit in ${SOURCE_DIR}" >&2
    exit 1
  fi
  if git -C "${SOURCE_DIR}" apply --unidiff-zero --check "${PATCH_FILE}"; then
    git -C "${SOURCE_DIR}" apply --unidiff-zero "${PATCH_FILE}"
  elif ! git -C "${SOURCE_DIR}" apply --unidiff-zero --reverse --check "${PATCH_FILE}"; then
    echo "Apple Silicon patch cannot be applied cleanly." >&2
    exit 1
  fi
  (
    cd "${SOURCE_DIR}"
    ./configure
    make -C libsent -j4
    make -C libjulius -j4
    make -C julius -j4
  )
  mkdir -p "${INSTALL_DIR}/bin"
  cp "${SOURCE_DIR}/julius/julius" "${JULIUS_BIN}"
  chmod 755 "${JULIUS_BIN}"
fi

if [[ ! -d "${KIT_DIR}/.git" ]]; then
  git clone https://github.com/julius-speech/segmentation-kit.git "${KIT_DIR}"
  git -C "${KIT_DIR}" checkout "${KIT_COMMIT}"
fi
if [[ "$(git -C "${KIT_DIR}" rev-parse HEAD)" != "${KIT_COMMIT}" ]]; then
  echo "Unexpected segmentation-kit commit in ${KIT_DIR}" >&2
  exit 1
fi

MODEL_FILE="${KIT_DIR}/models/hmmdefs_monof_mix16_gid.binhmm"
ACTUAL_MODEL_SHA256="$(shasum -a 256 "${MODEL_FILE}" | awk '{print $1}')"
if [[ "${ACTUAL_MODEL_SHA256}" != "${MODEL_SHA256}" ]]; then
  echo "Monophone model checksum does not match." >&2
  exit 1
fi

ln -sfn "../../${JULIUS_VERSION}/bin/julius" "${KIT_DIR}/bin/julius-4.3.1"
VERSION_OUTPUT="$("${JULIUS_BIN}" -version 2>&1 || true)"
if [[ "${VERSION_OUTPUT}" != *"JuliusLib rev.${JULIUS_VERSION}"* ]]; then
  echo "Julius version verification failed." >&2
  exit 1
fi

echo "Julius ${JULIUS_VERSION} is ready: ${JULIUS_BIN}"
echo "Segmentation kit is ready: ${KIT_DIR}"
