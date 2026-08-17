#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/Users/pjb/git/vllm-metal-m5max-qualification-20260817-123234}"
MODEL_DIR="${MODEL_DIR:-/Users/pjb/Models/AX-Qwen3.6-27B-6bit-native-mtp}"
EXPECTED_HEAD="b50405c9bd9e4f40366c672cec421c30df98f478"
BUNDLE_REF="0f3ccc1ca11e85381a096368a7b547ce1a93aee3"
BUNDLE_SHA256="2aa3cde1756c589717cecbbc1ea224ab8980f06d442fb2a6398d088048747e83"
RAW_BASE="https://raw.githubusercontent.com/PhilipJohnBasile/vllm-metal/${BUNDLE_REF}/tools/qwen36-parity"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This diagnostic requires an Apple Silicon Mac." >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" ]]; then
  echo "ERROR: Qualification checkout not found: $REPO" >&2
  exit 2
fi
CURRENT_HEAD="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: Wrong qualification checkout HEAD." >&2
  echo "Expected: $EXPECTED_HEAD" >&2
  echo "Found:    $CURRENT_HEAD" >&2
  exit 2
fi
if [[ ! -x "$REPO/.venv-m5max-qwen36/bin/python" ]]; then
  echo "ERROR: Existing qualification venv not found:" >&2
  echo "  $REPO/.venv-m5max-qwen36" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/model-mtp.safetensors" ]]; then
  echo "ERROR: Adopted native-MTP model not found or incomplete:" >&2
  echo "  $MODEL_DIR" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d /tmp/qwen36-mtp-parity.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
B64_FILE="$TMP_DIR/qwen36-mtp-parity-ladder.b64"
ZIP_FILE="$TMP_DIR/qwen36-mtp-parity-ladder.zip"
: > "$B64_FILE"

for part in 00 01 02 03; do
  echo "Downloading parity bundle part $part..."
  curl --fail --location --silent --show-error \
    "$RAW_BASE/bundle.part-$part.b64" >> "$B64_FILE"
done

base64 -D "$B64_FILE" > "$ZIP_FILE"
ACTUAL_SHA256="$(shasum -a 256 "$ZIP_FILE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$BUNDLE_SHA256" ]]; then
  echo "ERROR: Parity bundle checksum mismatch." >&2
  echo "Expected: $BUNDLE_SHA256" >&2
  echo "Found:    $ACTUAL_SHA256" >&2
  exit 2
fi

echo "Installing the verified parity ladder into: $REPO"
unzip -oq "$ZIP_FILE" -d "$REPO"
chmod +x "$REPO/scripts/run_qwen36_mtp_parity_ladder.sh"

echo "Using existing model: $MODEL_DIR"
echo "No model download will occur."
echo

cd "$REPO"
export MODEL_DIR
set +e
scripts/run_qwen36_mtp_parity_ladder.sh
STATUS=$?
set -e

LATEST_RESULT="$(ls -dt "$HOME"/vllm-metal-results/qwen36-mtp-parity-* 2>/dev/null | head -1 || true)"
echo
echo "============================================================"
echo "One-shot parity diagnostic finished with status: $STATUS"
echo "Latest evidence directory: ${LATEST_RESULT:-not produced}"
echo "============================================================"
if [[ -n "$LATEST_RESULT" && -f "$LATEST_RESULT/parity_summary.md" ]]; then
  echo
  cat "$LATEST_RESULT/parity_summary.md"
fi
exit "$STATUS"
