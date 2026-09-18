# strix-halo-qwen38

Deploy [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (125B-A6B MoE, 768K context) on AMD Ryzen AI MAX+ 395 ("Strix Halo", gfx1151) machines.

What you get: an OpenAI-compatible `llama-server` on port 8081, running inside the
[kyuz0 Strix Halo toolbox](https://github.com/kyuz0/amd-strix-halo-toolboxes) container,
under a **user-level systemd unit** (starts at boot, auto-restarts on crash), serving over
all interfaces (put your wg/LAN interface in a trusted firewall zone).

Large assets (the ~78 GB GGUF) are **fetched from Hugging Face** at install time — nothing
big lives in this repo.

## Requirements

- Strix Halo APU (Ryzen AI MAX 385/395), Linux with rootless podman
- Fedora-style host: `toolbox` CLI (Debian/Ubuntu: `distrobox` works too)
- Optional but recommended: `hf` CLI (`pip install -U huggingface_hub` or the
  [installer](https://hf.co/cli/install.sh)); otherwise any HF downloader producing GGUF files
- BIOS: dedicated VRAM/carveout **removed or small** (GTT layout, ~124 GB RAM to the OS).
  The launcher is layout-aware and also runs on carveout layouts, but GTT mode is the
  tested configuration.
- Recommended kernel params (already standard for this platform):
  `amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856`

## Quick start

```bash
git clone <this-repo> && cd strix-halo-qwen38
./install.sh                          # image + container + scripts + unit + model download
# first boot downloads ~78 GB; later boots start in ~30 s
curl http://localhost:8081/health
```

Override anything with flags or a config file:

```bash
./install.sh --model-root /data/models --ctx 524288 --port 8081 --host 10.99.0.5
# persistent overrides land in ~/.config/qwen38/env (read by the unit at start)
```

## Using it

```bash
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Hello"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Management:

```bash
systemctl --user status llama-server          # supervises container + server
journalctl --user -u llama-server -f          # server logs
```

## Configuration

Everything is environment-driven; persistent overrides go in `~/.config/qwen38/env`:

| var | default | meaning |
|---|---|---|
| `QWEN38_MODEL_ROOT` | `~/.local/share/qwen38/models` | where GGUF shards live |
| `QWEN38_MODEL` | *(auto-discovered)* | full path to shard `-00001-of-*.gguf` |
| `QWEN38_CTX` | `786432` | context tokens (see cost table below) |
| `QWEN38_YARN_SCALE` | `3` | YaRN factor (native 262144 × scale = ctx) |
| `QWEN38_HOST` / `QWEN38_PORT` | `0.0.0.0` / `8081` | bind |
| `QWEN38_CONTAINER` | `llama-rocm-10.0` | toolbox container name |
| `QWEN38_MMPROJ` | *(unset = text-only)* | path to `mmproj-F16.gguf` to enable vision (+900 MB, ~4 GB RAM) |

**Context costs ~41 KB/token** (KV at q8_0 + sparse-attention metadata), measured on a
128 GB machine: 32K → 42 GB free · 512K → 20 GB · 768K → 11 GB · 1M → 1.4 GB (do not).
Default 768K = YaRN ×3 from the native 262,144.

## Notes & gotchas (details in docs/RUNBOOK.md)

- **Never set `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` on a carveout layout** (VRAM carveout in
  BIOS) — it inverts memory placement. **Do set it in GTT mode** (no carveout). The launcher
  detects the layout (GTT ≥ 100 GiB) and does the right thing.
- Keep mmap on, `-fit off`, `--parallel 1`; KV at q8_0, never bf16 on this arch.
- Thinking is ON by default (`reasoning_effort` defaults to xhigh); send
  `chat_template_kwargs: {"enable_thinking": false}` or `reasoning_effort: "medium"` for
  snappy answers.
- GPU card number can shift between layouts (`card0`/`card1`) — the launcher globs, so
  nothing to do.
- The endpoint is unauthenticated; keep it behind a trusted-zone interface or add an API key.

## Benchmarks (128 GB Strix Halo, MS-S1 MAX, 2026-09)

Stock llama.cpp 0.4.0, GTT mode: **~24.8 tok/s decode flat** across prose/code/structured,
TTFT ~91 ms (short prompts), prefill ~250 tok/s sustained. Fork engine + MTP draft claims
66–82 tok/s on code — see `docs/RUNBOOK.md` for the current state of that path
(`MODE=fork ./bin/serve.sh`).

## Licenses

- This repo: MIT (see LICENSE)
- Model: Qwen Community License 1.0 (has MaaS/commercial carve-outs — read it)
- Engine: kyuz0 toolbox images + llama.cpp (upstream / Aristo94 EngramHalo fork for MTP)
