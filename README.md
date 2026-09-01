# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## About this fork

This fork is an experimental long-context serving branch built around a shared,
paged KV cache. It keeps the upstream `llama.cpp` interface while adding the
pieces needed to run long, concurrent Qwen requests on memory-constrained GPUs.

The main additions are:

- paged KV pools for CUDA and Vulkan, including quantized paged attention;
- shared physical KV pages across server slots instead of a fixed KV partition
  per slot;
- multi-sequence scheduling, physical-budget admission control, CPU/GPU pools,
  and dynamic page growth;
- CUDA Flash Decode / DFlash2 support for paged attention;
- optional SnapKV page scoring, selective retention, per-sequence score state,
  eviction timing, and a repeatable quality benchmark;
- page-eviction guards in the paged attention kernels (`physical_block < 0` is
  skipped/masked instead of dereferenced), so evicted pages never corrupt KV
  reads during prefill or decode;
- a per-sequence `n_prompt`/`n_decoded` allocator invariant so cached tokens
  and decoded tokens keep `n_prompt + n_decoded == previous_max + 1` while
  pages are being released and reused;
- SnapKV state kept per sequence and correctly sized against the *logical*
  per-slot context even under `--kv-paged-dynamic` with multiple parallel
  slots (the fix uses `cparams.n_ctx` instead of `n_ctx_seq`, which previously
  under-sized the score window and crashed with more than one slot);
- `--kv-paged-prealloc-max`, which measures the usable GPU budget, runs one
  synthetic long prefill before the health endpoint becomes ready, and freezes
  the largest validated page pool for the lifetime of the server.

The last mode avoids allocations and page-table migrations in live prefill. A
server started with `--ctx-size 0` therefore publishes the physical context
that was actually calibrated, rather than a larger optimistic value that can
fail later under load. Slots draw from one shared pool: `--parallel` limits
simultaneous sequences but does not divide the pool into fixed per-slot quotas.
Pages released by a completed request can be reused by requests that remain.

### RTX 4070 12 GB results (2026-09-01)

Current production profile, measured on a GeForce RTX 4070 12 GB with
Qwen3.8-27B `UD-IQ3_XXS`, paged `q4_0` K/V cache, CUDA Flash Attention, one
server slot, MTP speculative decoding and dynamic paged KV. All pool
parameters are explicit, so startup skips the calibration pass and the server
is ready in about 12 s:

```sh
llama-server \
  --model Qwen3.8-27B-UD-IQ3_XXS.gguf \
  --ctx-size 40960 --parallel 1 \
  --batch-size 128 --ubatch-size 128 \
  --device CUDA0 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --kv-paged --kv-paged-dynamic --n-gpu-blocks 2560 \
  --n-gpu-blocks-initial 64 --n-gpu-blocks-growth 64 \
  --kv-paged-admission-blocks 2560 --fit off --kv-block-size 16 \
  --cache-ram 0 --ctx-checkpoints 1 --metrics \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.80 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
```

`--ctx-size 40960` matches the confirmed operational limit: a request of
36,866 prompt + 4,094 output tokens (40,960 total) is the maximum that
completes. The nominal 43,008 calibration is not reachable on this hardware;
beyond 40,960 the first KV page growth fails with a `cudaMalloc` OOM of about
46 MiB and the request errors with `Failed to reserve paged KV cache`.

`--ctx-checkpoints 1` is required for prompt caching on hybrid models.
Qwen3.8-27B-UD has 16 attention layers and 48 recurrent layers, so the hybrid
paged memory reports the rolling recurrent tail as the sequence minimum and
the server would otherwise force a full prompt re-process on every request
(zero cached tokens). Context checkpoints restore the attention and draft
state: an identical 1,651-token prompt drops from 9.8 s to 0.3 s, and the
maximum 36,867-token prompt reuses 36,863 cached tokens.

Measured OpenAI-compatible completions at the maximum gate (`ignore_eos`,
temperature 0):

| Profile | Prefill | Decode | Wall time |
| --- | ---: | ---: | ---: |
| P1 MTP nmax2, cold 36,867 + 4,091 | 631.1 tok/s | 32.6 tok/s | 183.9 s |
| P1 MTP nmax2, warm (36,863 cached) | - | ~32.6 tok/s | 160.2 s |
| P1 MTP nmax2, 2026-08-26 prealloc-max gate | 544.4 tok/s | 38.0 tok/s | 175.6 s |

The 2026-08-26 row is the `--kv-paged-prealloc-max --ctx-size 0 --fit off
--fit-target 643` profile on the same gate (36,864 + 4,096), kept for
reference. Peak VRAM during the maximum request is 11,899 / 12,282 MiB
(GPU-only, no spill). `nmax=3` OOMs repeatedly and DFlash2 stays below
target-only, so MTP `nmax=2` remains the promoted speculative profile.

CPU spill (`--n-cpu-blocks N` with N > 1) removes the reserve failure but
migrates whole attention layers to CPU, where the paged attention reference
kernel drops decode to about 3.6 tok/s. It stays disabled in production.

High-capacity profile without MTP (shared paged pool, `--parallel 3`, logical
context 77,824; startup calibration with `--kv-paged-prealloc-max` selects
4,032 pages of 16 tokens = 64,512 physical tokens in about 133 s):

| Workload | Prefill | Decode | Wall time |
| --- | ---: | ---: | ---: |
| 30,052 + 256, first request | 1,103.9 tok/s | 27.0 tok/s | 36.7 s |
| distinct 30,052 + 256, next request | 1,100.2 tok/s | 27.0 tok/s | 36.8 s |
| 64,511 + 1 | 1,100.5 tok/s | - | 58.6 s |
| 60,416 + 4,096 | 1,098.4 tok/s | 21.3 tok/s | 247.1 s |
| 62,052 + 4,096 (76K logical) | 535.3 tok/s | 22.3 tok/s | complete |

`--parallel 4` aborts the long gate at 57,344 tokens with a `cuMemCreate`
OOM; admission control defers overflow requests but does not replace the
FlashAttention workspace budget. Paged KV does not beat plain KV at equal
context - it wins by shared capacity, dynamic growth and eviction, not raw
decode speed (about 1% at short context, 6-7% slower decode at 40-48K).

SnapKV is available but not promoted for production: above the physical pool
it accepts long prompts but evicts pages and loses needle facts (80K logical
/ 48K physical missed 3/3 facts). Results are hardware-, model-, build-, and
cache-type-specific; the automatic calibration is intended to determine the
corresponding safe value on another system.

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
