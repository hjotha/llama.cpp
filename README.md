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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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

### Recent integration status (2026-09-04)

The current fork `master` contains the upstream merge through
`86b351fd6`, followed by the experimental SnapKV paged-retention merges. The
recent work adds streaming SnapKV capture, per-head page scores, selective
eviction, CPU-to-GPU page migration, and the repeatable
`tools/snapkv-40k-mtp-bench.py` harness.

The normal KV and traditional MTP paths remain the upstream implementation.
The added code is selected only for paged KV configurations. SnapKV remains
experimental: it is a long-context capacity and quality experiment, not the
default production profile.

### RTX 4070 12 GB calibration and production profile (2026-09-04)

The complete, model-specific calibration record is in
[docs/kv-calibration-findings.md](docs/kv-calibration-findings.md). Results
below use CUDA0, `q4_0` K/V, `parallel=1`, CUDA Flash Attention, and the
tested RTX 4070 12 GB unless stated otherwise.

The promoted profile is traditional KV with the ISTA MTP GGUF. It avoids the
paged-attention throughput penalty while retaining a validated 54K request
budget:

```sh
llama-server \
  --model Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  --ctx-size 54272 --parallel 1 \
  --batch-size 64 --ubatch-size 64 \
  --device CUDA0 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --fit off --cache-ram 4096 --ctx-checkpoints 1 --metrics \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.80 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
```

| Configuration | Validated boundary | Observed throughput | Operational result |
| --- | --- | --- | --- |
| ISTA traditional MTP | 50,168 prompt + 4,096 output | 510.54 prompt tok/s, 46.98 decode tok/s | promoted production profile |
| ISTA paged MTP | 60,160 total tokens | 471.00 prompt tok/s, 30.11 decode tok/s | more capacity, not a speed replacement |
| ISTA traditional no-MTP | 97,280 total tokens | 456.42 prompt tok/s, 21.19 decode tok/s | highest tested traditional capacity |
| ISTA paged no-MTP | 105,584 total tokens | 439.02 prompt tok/s, 17.27 decode tok/s | highest tested paged capacity |

For a literal 4,096-token MTP completion, `54,264` is the highest validated
traditional-KV context. At `54,272`, the server completes 4,094 tokens at the
exact boundary; `54,273` OOMs during the request. This two-token reserve is a
server boundary behavior, not a client-side cache hit or throughput issue.

Paged KV expands the memory ceiling, but should not be selected for raw
throughput at equal context. At `ctx-size=54,272`, paged MTP measured 485.01
prompt tok/s and 32.26 decode tok/s versus 549.13 and 46.66 for traditional
KV: 11.7% lower prefill and 30.9% lower decode. The current paged attention
path is still experimental and is expected to improve with backend work.

The calibration is intentionally model- and configuration-specific. The
larger Unsloth UD GGUF has different free VRAM and different usable limits;
do not reuse the ISTA thresholds for it. Run the automatic probe and then a
maximum-shaped request on the target model, batch/ubatch, cache type, and GPU.

Traditional MTP was also checked against a CUDA `sm_89` stock-upstream build.
At 128, 8,192, and 50,168 prompt tokens, the stock decode delta was 0.053%,
0.104%, and 0.532%, respectively, with identical draft counts and acceptance.
This is not a material regression; MTP acceptance and GPU clock variation are
larger sources of sub-1% run-to-run differences.

Sustained requests with varying prompt shapes exposed an upstream CUDA graph
recapture OOM at this tight VRAM boundary. The local fix requires four stable
graph calls before capture, preserving decode graphs while avoiding transient
prefill captures. The exact reproduction, upstream A/B, instrumentation, and
ten-request validation are recorded in
[docs/oom-reproduction-trace.md](docs/oom-reproduction-trace.md).

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
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
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
