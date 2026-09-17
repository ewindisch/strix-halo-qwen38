#!/usr/bin/env bash
# Install/upgrade the strix-halo-qwen38 service on this machine.
# Idempotent: safe to re-run. Large assets (model GGUF) are downloaded from
# Hugging Face, never stored in the repo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARE_DIR="$HOME/.local/share/qwen38"
MODEL_ROOT="${QWEN38_MODEL_ROOT:-$SHARE_DIR/models}"
CONTAINER="${QWEN38_CONTAINER:-llama-rocm-10.0}"
IMAGE="docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0"
CTX="${QWEN38_CTX:-786432}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-8081}"
SKIP_MODELS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

echo "==> model root: $MODEL_ROOT"

# 1. container engine: toolbox (Fedora) or distrobox, else plain podman
TOOL="toolbox"
command -v toolbox >/dev/null 2>&1 || { command -v distrobox >/dev/null 2>&1 && TOOL="distrobox"; }

if ! podman image exists "$IMAGE" 2>/dev/null; then
  echo "==> pulling $IMAGE"
  podman pull "$IMAGE"
fi

if ! podman container exists "$CONTAINER" 2>/dev/null; then
  echo "==> creating toolbox container '$CONTAINER'"
  if [ "$TOOL" = "toolbox" ]; then
    toolbox create "$CONTAINER" --image "$IMAGE" -y >/dev/null
  elif [ "$TOOL" = "distrobox" ]; then
    distrobox create --name "$CONTAINER" --image "$IMAGE" >/dev/null
  else
    podman run -dt --name "$CONTAINER" \
      --device /dev/dri --device /dev/kfd \
      --group-add video --group-add render \
      --security-opt seccomp=unconfined --security-opt label=disable \
      "$IMAGE" sleep infinity
  fi
else
  echo "==> container '$CONTAINER' already exists"
fi

# 2. launcher scripts
echo "==> installing scripts to $SHARE_DIR/bin"
mkdir -p "$SHARE_DIR/bin" "$MODEL_ROOT" "$HOME/.config/qwen38"
install -m 0755 "$REPO_DIR/bin/llama-server-start.sh" "$SHARE_DIR/bin/"
install -m 0755 "$REPO_DIR/bin/serve.sh"              "$SHARE_DIR/bin/" 2>/dev/null || true

# 3. model download (large asset — from Hugging Face, not this repo)
if [ "$SKIP_MODELS" = 0 ]; then
  have_model=$(find "$MODEL_ROOT" -name '*Qwen3.8-Flash-Next*00001-of-*.gguf' 2>/dev/null | head -1 || true)
  if [ -n "$have_model" ]; then
    echo "==> model already present: $have_model"
  else
    echo "==> downloading unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ3_XXS (~78 GB) — this takes a while"
    if command -v hf >/dev/null 2>&1; then
      hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-IQ3_XXS/*" --local-dir "$MODEL_ROOT"
      mv "$MODEL_ROOT"/UD-IQ3_XXS/*.gguf "$MODEL_ROOT"/ 2>/dev/null || true
      rmdir "$MODEL_ROOT/UD-IQ3_XXS" 2>/dev/null || true
    else
      echo "    'hf' CLI not found — install it (pip install -U huggingface_hub)" >&2
      echo "    then re-run with the command above, or use any HF downloader." >&2
      exit 1
    fi
  fi
fi

# 4. config override file (only written if absent — never clobbers)
if [ ! -f "$HOME/.config/qwen38/env" ]; then
  cat > "$HOME/.config/qwen38/env" <<EOF
QWEN38_MODEL_ROOT=$MODEL_ROOT
QWEN38_CTX=$CTX
QWEN38_HOST=$HOST
QWEN38_PORT=$PORT
QWEN38_CONTAINER=$CONTAINER
EOF
  echo "==> wrote ~/.config/qwen38/env"
fi

# 5. systemd user unit (backs up any previous version)
mkdir -p "$HOME/.config/systemd/user"
if [ -f "$HOME/.config/systemd/user/llama-server.service" ]; then
  cp "$HOME/.config/systemd/user/llama-server.service" \
     "$HOME/.config/systemd/user/llama-server.service.bak.$(date +%s)"
fi
install -m 0644 "$REPO_DIR/systemd/llama-server.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload

# 6. linger + enable
loginctl enable-linger 2>/dev/null || echo "!! run 'loginctl enable-linger' (may need your password) so the unit starts at boot"
systemctl --user enable --now llama-server.service

echo
echo "==> done. waiting for health (first model load ~30 s, cold cache a few min)..."
for i in $(seq 1 100); do
  if curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo "==> healthy: http://localhost:$PORT/  (OpenAI-compatible /v1)"
    exit 0
  fi
  sleep 3
done
echo "!! not healthy yet — check: journalctl --user -u llama-server -n 50" >&2
exit 1
