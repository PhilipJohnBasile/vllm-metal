#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MLX_LM_MTP_SHA="${MLX_LM_MTP_SHA:-a9fd7ef1032419a584ead9a38bdb66635f2d85c3}"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-m5max-qwen36}"
MODEL_DIR=""
OUTPUT_DIR=""
PREPARE_OFFICIAL=0
Q_BITS=6
SKIP_INSTALL=0
ALLOW_NON_M5MAX=0
REQUIRE_SPEEDUP=1
REPEATS=3

usage() {
  cat <<'EOF'
Usage:
  scripts/run_m5max_qwen36_qualification.sh --model-dir PATH [options]

Required:
  --model-dir PATH          Native MLX model directory containing mtp.* tensors.

Options:
  --prepare-official        Convert Qwen/Qwen3.6-27B into MODEL_DIR first.
  --q-bits N                Quantization bits for conversion (default: 6).
  --output-dir PATH         Evidence directory (default: ~/vllm-metal-results/...).
  --repeats N               Timed repetitions per workload/launch (default: 3).
  --skip-install            Reuse the existing qualification virtualenv.
  --allow-non-m5max         Debug the harness on another arm64 Mac.
  --no-require-speedup      Return success on functional pass even if <1.05x.
  -h, --help                Show this help.

The positive claim gate requires exact output parity, clean lifecycle evidence,
positive MTP acceptance, <=15% CV/drift, >=1.05x geometric-mean speedup, and no
workload below 0.95x baseline.
EOF
}

while (($#)); do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --prepare-official)
      PREPARE_OFFICIAL=1
      shift
      ;;
    --q-bits)
      Q_BITS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --allow-non-m5max)
      ALLOW_NON_M5MAX=1
      shift
      ;;
    --no-require-speedup)
      REQUIRE_SPEEDUP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "--model-dir is required" >&2
  usage >&2
  exit 64
fi
if [[ ! "$Q_BITS" =~ ^(4|6|8)$ ]]; then
  echo "--q-bits must be 4, 6, or 8" >&2
  exit 64
fi
if [[ ! "$REPEATS" =~ ^[2-9][0-9]*$ ]]; then
  echo "--repeats must be an integer >= 2" >&2
  exit 64
fi

MODEL_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$MODEL_DIR")"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$HOME/vllm-metal-results/qwen36-mtp-m5max-$(date -u +%Y%m%dT%H%M%SZ)"
fi
OUTPUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This qualification requires an arm64 Mac." >&2
  exit 2
fi
CHIP="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/^[[:space:]]*Chip:/{print $2; exit}')"
if [[ "$CHIP" != *"M5 Max"* && "$ALLOW_NON_M5MAX" -ne 1 ]]; then
  echo "Expected Apple M5 Max; detected '${CHIP:-unknown}'." >&2
  echo "Use --allow-non-m5max only to debug the harness." >&2
  exit 2
fi

echo "Repository: $REPO_ROOT"
echo "Model:      $MODEL_DIR"
echo "Evidence:   $OUTPUT_DIR"
echo "Chip:       ${CHIP:-unknown}"
echo "MLX-LM:     $MLX_LM_MTP_SHA"

if [[ "$SKIP_INSTALL" -ne 1 ]]; then
  python3.12 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip uv 'huggingface_hub[hf_xet]'
  python -m pip install "$VLLM_WHEEL"
  VLLM_METAL_BUILD_FROM_SOURCE=1 python -m pip install -e '.[dev]'
  python -m pip install --force-reinstall --no-deps \
    "git+https://github.com/PhilipJohnBasile/mlx-lm@${MLX_LM_MTP_SHA}"
  source scripts/lib.sh
  ensure_metal_toolchain
  build_native_artifacts
else
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "--skip-install requested but $VENV_DIR does not exist" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [[ "$PREPARE_OFFICIAL" -eq 1 ]]; then
  if [[ -e "$MODEL_DIR" ]]; then
    if [[ ! -f "$MODEL_DIR/config.json" ]]; then
      echo "Refusing to convert into non-model path: $MODEL_DIR" >&2
      exit 2
    fi
    echo "Model directory already exists; validating it instead of reconverting."
  else
    parent="$(dirname "$MODEL_DIR")"
    mkdir -p "$parent"
    free_kb="$(df -Pk "$parent" | awk 'NR==2{print $4}')"
    free_gib=$((free_kb / 1024 / 1024))
    if ((free_gib < 100)); then
      echo "WARNING: only ${free_gib} GiB free. Official BF16 download plus the" >&2
      echo "6-bit output can require roughly 100 GiB of temporary/free space." >&2
    fi
    echo "Converting official Qwen/Qwen3.6-27B with native MTP tensors..."
    python -m mlx_lm.convert \
      --hf-path Qwen/Qwen3.6-27B \
      --mlx-path "$MODEL_DIR" \
      -q \
      --q-bits "$Q_BITS" \
      --q-group-size 64 \
      --dtype bfloat16
  fi
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "No model found at $MODEL_DIR." >&2
  echo "Pass --prepare-official or point --model-dir at a completed native-MTP MLX conversion." >&2
  exit 2
fi

ulimit -n 4096 || true
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
export VLLM_METAL_USE_PAGED_ATTENTION=1
export VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-auto}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"

{
  echo "qualification_branch=$(git branch --show-current)"
  echo "qualification_head=$(git rev-parse HEAD)"
  echo "mlx_lm_mtp_sha=$MLX_LM_MTP_SHA"
  echo "model_dir=$MODEL_DIR"
  echo "model_config_sha256=$(shasum -a 256 "$MODEL_DIR/config.json" | awk '{print $1}')"
  sw_vers
  uname -a
  system_profiler SPHardwareDataType
} | tee "$OUTPUT_DIR/revisions-and-hardware.txt"

command=(
  python benchmarks/qwen_mtp_m5max_qualification.py
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --repeats "$REPEATS"
)
if [[ "$ALLOW_NON_M5MAX" -eq 1 ]]; then
  command+=(--allow-non-m5max)
fi
if [[ "$REQUIRE_SPEEDUP" -eq 1 ]]; then
  command+=(--require-speedup)
fi

set +e
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
status=${PIPESTATUS[0]}
set -e

if [[ -f "$OUTPUT_DIR/qualification_summary.md" ]]; then
  echo
  cat "$OUTPUT_DIR/qualification_summary.md"
fi

echo
echo "Evidence written to: $OUTPUT_DIR"
case "$status" in
  0)
    echo "Qualification command passed."
    ;;
  2)
    echo "Functional qualification failed. Inspect run.log and qualification_summary.json." >&2
    ;;
  3)
    echo "Functional qualification passed, but the positive >=1.05x performance claim gate failed." >&2
    ;;
  *)
    echo "Qualification stopped with exit status $status." >&2
    ;;
esac
exit "$status"
