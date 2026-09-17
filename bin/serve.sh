#!/usr/bin/env bash
# Manual/experimental launcher (NOT for boot — use the systemd unit for that).
#   MODE=stock (default)  stock llama.cpp in the toolbox container
#   MODE=fork             EngramHalo fork + MTP draft (see docs/RUNBOOK.md "fork mode")
# GTT mode required for fork; layout assertion included.
set -euo pipefail

MODE="${MODE:-stock}"
HOST="${HOST:-${QWEN38_HOST:-0.0.0.0}}"
PORT="${PORT:-${QWEN38_PORT:-8081}}"
MODEL_ROOT="${QWEN38_MODEL_ROOT:-$HOME/.local/share/qwen38/models}"
CONTAINER="${QWEN38_CONTAINER:-llama-rocm-10.0}"
CTX="${QWEN38_CTX:-786432}"
YARN_SCALE="${QWEN38_YARN_SCALE:-3}"
YARN_ORIG_CTX="${QWEN38_YARN_ORIG_CTX:-262144}"

GT=""
for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
  [ -f "$f" ] && GT="$f" && break
done
gtt_gib=$(awk '{printf "%d", $1/1073741824}' "${GT:-/dev/null}" 2>/dev/null || echo 0)
if [ "${gtt_gib:-0}" -lt 100 ]; then
  echo "ABORT: GTT ${gtt_gib:-0} GiB < 100 — carveout layout (GTT mode required here)" >&2
  exit 1
fi
echo "GTT: ${gtt_gib} GiB confirmed"

pkill -f "^llama-server -m" 2>/dev/null || true
sleep 2

if [ "$MODE" = "fork" ]; then
  RECIPE="${QWEN38_RECIPE_DIR:-$HOME/.local/share/qwen38/qwen-3.8-next-flash-amd-strix-halo}"
  ENGINE="$RECIPE/engine/build-hip/bin/llama-server"
  [ -x "$ENGINE" ] || { echo "fork engine missing — see docs/RUNBOOK.md 'fork mode' (build-engine.sh)" >&2; exit 1; }
  cd "$RECIPE"
  MODEL_ROOT="$MODEL_ROOT" CTX="$CTX" SERVER_HOST="$HOST" SERVER_PORT="$PORT" \
    LLAMA_SERVER="$ENGINE" \
    setsid nohup ./recipe/serve-fast.sh > serve-fork.log 2>&1 < /dev/null &
else
  podman start "$CONTAINER" >/dev/null 2>&1 || true
  setsid nohup podman exec "$CONTAINER" env GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
    llama-server \
    -m "$(find "$MODEL_ROOT" -name '*Qwen3.8-Flash-Next*00001-of-*.gguf' | head -1)" \
    --alias qwen3.8-flash-next --host "$HOST" --port "$PORT" \
    -ngl 99 -c "$CTX" -fit off --jinja -fa on -ctk q8_0 -ctv q8_0 \
    --rope-scaling yarn --rope-scale "$YARN_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX" \
    -b 8192 -ub 2048 --parallel 1 --no-webui --no-warmup \
    > "$HOME/.local/share/qwen38/serve-stock.log" 2>&1 < /dev/null &
fi
disown
echo "launched MODE=$MODE — poll: curl -s http://localhost:$PORT/health"
