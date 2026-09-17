# Runbook — Qwen3.8-Flash-Next on Strix Halo

Hard-won operational knowledge from bringing this up on a 128 GB Meigao MS-S1 MAX
(Fedora 43, kernel 7.1.8, rootless podman). Read before debugging.

## 1. Memory layouts — the single most important thing

Two mutually exclusive layouts, set by BIOS carveout ("UMA Frame Buffer Size" / "Dedicated
GPU memory"):

| | GTT mode (recommended) | Carveout mode |
|---|---|---|
| BIOS | Auto / minimum | e.g. 96 GB |
| OS RAM | ~124 GB | ~31 GB on a 128 GB box |
| `mem_info_gtt_total` | ~124 GiB | ~half of OS RAM |
| **`GGML_HIP_ENABLE_UNIFIED_MEMORY`** | **=1 REQUIRED** | **must NOT be set** (inverts placement — measured 76.9 GB anon / 0.2 GiB GPU) |
| status | canonical for kyuz0 images | exotic; see §2 |

The launcher in `bin/llama-server-start.sh` detects this via `mem_info_gtt_total >= 100 GiB`
and sets the env var accordingly. Do not override.

## 2. Known load-path failures (carveout-era, 2026-09-16/17)

All observed on a 96 GiB-carveout box; **GTT mode loads the same builds fine**:

| engine | load mode | result |
|---|---|---|
| EngramHalo fork (host build) | mmap (+/- lazy table) | hang post-weight-copy; thread stuck `kfd_wait_on_events` |
| EngramHalo fork | `-lm none` (ram) | same hang |
| stock 0.4.0 | `-lm mmap` | ~1-core spin, never ready |
| stock 0.4.0 | `-ot per_layer_token_embd.weight=ROCm0` | same kfd hang (do NOT pin the table to GPU) |
| stock 0.4.0 | **`-lm auto` (default)** | **loads in ~26 s — the only working config there** |

Side findings: the card number can shift (`card0`→`card1`) when the layout changes — never
hardcode; the table tensor is `per_layer_token_embd.weight` (26.8 GiB in the IQ3_XXS build);
HIP builds name their buffer `ROCm0` (not `CUDA0`).

## 3. Standing traps

1. `-fit off` always — auto-fit mis-sizes qwen4exp.
2. `--parallel 1` — multi-slot trips this arch's cache handling.
3. KV q8_0 (`-ctk q8_0 -ctv q8_0`); never bf16 on this attention shape (head-dim 256).
4. Thinking defaults ON with `reasoning_effort=xhigh` — send
   `chat_template_kwargs:{"enable_thinking":false}` or `reasoning_effort:"medium"`.
5. `amdgpu.gttsize=126976 ttm.pages_limit=32505856 amd_iommu=off` on the kernel cmdline
   (GTT mode tuning; `amd_iommu=off` disables the NPU and DMA isolation).
6. Endpoint is unauthenticated — keep it on a trusted-zone interface (wg0 in firewalld
   `trusted` works) or add `--api-key`.
7. Rootless podman + user systemd: `loginctl enable-linger` is required for boot start.
8. GPU sysfs may lag at early boot — the launcher exits non-zero rather than guess; the
   unit's restart cap (5/600 s) stops hammering; `systemctl --user restart` once the GPU is up.

## 4. Context cost (measured, 128 GB GTT)

~41 KB/token total (13 KB KV@q8_0 + ~28 KB ctx-scaling allocation, linear):

| ctx (yarn ×N from 262144) | used | avail |
|---|---|---|
| 32K | 81 GiB | 42 GiB |
| 512K (×2) | 103 GiB | 20 GiB |
| 768K (×3) — default | 112 GiB | 11 GiB |
| 1M (×4) | 123 GiB | 1.4 GiB — allocates but first real prompt OOMs |

GUF carries no rope-scaling keys (native 262144, freq_base 10M, mrope [11,11,10,0]) —
the CLI `--rope-scaling yarn` flags are the single source of truth.

## 5. Fork mode (max performance path) — status: untested on GTT

The [abliter8 recipe](https://github.com/abliter8-ai/qwen-3.8-next-flash-amd-strix-halo)
(EngramHalo.cpp fork, HIP-only) claims 66–82 tok/s code / ~22 prose with MTP draft + ngram
speculation. Its author ran GTT mode. On a carveout box the engine hung in every load mode (§2).

Setup (GTT machines only, experimentally):

```bash
git clone https://github.com/abliter8-ai/qwen-3.8-next-flash-amd-strix-halo
cd qwen-3.8-next-flash-amd-strix-halo
./recipe/build-engine.sh                       # host needs ROCm 7.x devel (hipcc, hip-lang)
./recipe/export-mtp-sidecar.sh Qwen/Qwen3.8-Flash-Next <MODEL_ROOT>
    # pinned converter rejects '--no-ple'; run with just: python3 convert_hf_to_gguf.py \
    #   --remote Qwen/Qwen3.8-Flash-Next --mtp --outtype bf16 --outfile <MODEL_ROOT>/mtp-...-BF16.gguf
    # then: llama-quantize <BF16> <MODEL_ROOT>/mtp-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0
MODE=fork ./bin/serve.sh                      # port 8080
```

kyuz0 toolbox images ship HIP runtime but **no `hip-lang` CMake package** (dnf shows 0 rocm
packages; /opt/rocm is baked in) — container rebuilds don't work; build on a ROCm host.

## 6. MTP draft on stock builds — dead end at b10937

`--spec-type draft-mtp` exists in stock 0.4.0 but no compatible draft file does: the
upstream converter at that commit hard-rejects `--mtp` for Qwen4Exp; unsloth's MTP files
(blk.48 naming / shared-embedding variant) don't match the loader either (needs blk.0 +
token_embd + output_hc_* + fused attn_qkv). Files remain staged under `models/MTP/` for
when a newer build ships. `ngram-mod` works but measured ≈ no-spec on novel text.
