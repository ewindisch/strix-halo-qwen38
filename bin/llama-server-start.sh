#!/usr/bin/env bash
# Layout-aware llama-server launcher for Qwen3.8-Flash-Next on Strix Halo.
# Designed to run as the ExecStart of a user-level systemd unit (foreground exec).
#
# Layout detection: GTT mode (no BIOS carveout, gtt >= 100 GiB) requires
# GGML_HIP_ENABLE_UNIFIED_MEMORY=1; carveout layouts must NOT set it.
set -euo pipefail

MODEL_ROOT="${QWEN38_MODEL_ROOT:-$HOME/.local/share/qwen38/models}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-8081}"
CTX="${QWEN38_CTX:-786432}"
YARN_SCALE="${QWEN38_YARN_SCALE:-3}"
YARN_ORIG_CTX="${QWEN38_YARN_ORIG_CTX:-262144}"
CONTAINER="${QWEN38_CONTAINER:-llama-rocm-10.0}"
MODEL="${QWEN38_MODEL:-}"

# --- find the amdgpu card (cardN numbering shifts between memory layouts) ---
GT=""
for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
  [ -f "$f" ] && GT="$f" && break
done
if [ -z "$GT" ]; then
  echo "no amdgpu mem_info_gtt_total under /sys/class/drm — GPU not ready?" >&2
  exit 1
fi
gtt_gib=$(awk '{printf "%d", $1/1073741824}' "$GT")

ENV_ARGS=()
if [ "$gtt_gib" -ge 100 ]; then
  echo "GTT mode detected (${gtt_gib} GiB) — enabling GGML_HIP_ENABLE_UNIFIED_MEMORY=1"
  ENV_ARGS+=(--env GGML_HIP_ENABLE_UNIFIED_MEMORY=1)
else
  echo "Carveout layout detected (GTT ${gtt_gib} GiB < 100) — unified memory NOT set"
fi

# --- locate model shard 1 ---
if [ -z "$MODEL" ]; then
  MODEL=$(find "$MODEL_ROOT" -name '*Qwen3.8-Flash-Next*00001-of-*.gguf' 2>/dev/null | head -1)
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "no Qwen3.8-Flash-Next GGUF under $MODEL_ROOT — run install.sh or set QWEN38_MODEL" >&2
  exit 1
fi

podman start "$CONTAINER" >/dev/null 2>&1 || true

exec podman exec "${ENV_ARGS[@]}" "$CONTAINER" llama-server \
  -m "$MODEL" \
  --alias qwen3.8-flash-next --host "$HOST" --port "$PORT" \
  -ngl 99 -c "$CTX" -fit off --jinja -fa on -ctk q8_0 -ctv q8_0 \
  --rope-scaling yarn --rope-scale "$YARN_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX" \
  -b 8192 -ub 2048 --parallel 1 --no-webui --no-warmup
