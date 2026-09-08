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

This fork follows official `ggml-org/llama.cpp` and adds context-based model
routing, CUDA memory fixes, GPU governors, and experimental paged KV serving.
The feature list below distinguishes source additions from the configuration
actually used by the GOKAYA production server.

### Upstream and production status (2026-09-08)

Official upstream is integrated through `64e9bceb2` (21 new upstream commits).
The validated server build is `de57d0269`, including a follow-up fix for two
cold-autoload races found during integration: eviction before the first proxy
starts, and simultaneous requests competing for the same model slot.

GOKAYA port `8090` runs this build and its matching shared libraries from
`/home/hjotha/llama-releases/de57d0269/build/bin`, under
`llama-server-root.service`. Source hashes match the fork for all
1,712 audited files in `common`, `src`, `include`, `ggml`, and
`tools/server` (documentation, tests, assets and vendor directories excluded).
There is no pending runtime-source delta within that scope. The README and
operational-plan update follows the runtime build as a documentation commit.
The separate Vulkan compactor on port `8092` retains its existing build.

Both production profiles use the same `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
and public model name `qwen-3.8-27b`. The preset is
`/home/hjotha/prod-two-tier.ini`; `models-max=1` keeps one child loaded at a time.

| Profile | Context | Batch / ubatch | Validated input + output |
| --- | ---: | ---: | ---: |
| MTP | 61,184 | 64 | 57,088 + 4,094 |
| No MTP | 98,304 | 512 | 94,208 + 4,096 |

The route threshold is 61,184 **prompt plus requested output tokens**; the
larger budget selects the 98,304-token no-MTP profile. Existing conversation
pinning prevents demotion of a conversation already on the larger profile.
Cold-start routing still uses its existing byte estimate until a tokenizer is
available. Common settings are traditional KV, `parallel=1`, `q4_0` K/V,
`flash-attn=on`, `fit=off`, `load-mode=none`, `cache-ram=2048`, and one context checkpoint.
The native buffered load mode avoids full-file mmap population: rapid tier
reloads with mmap triggered a host-RAM OOM during the first deployment canary.
The same routing and cache canaries passed with buffered loading.
The MTP profile uses two draft tokens and `spec-draft-p-min=0.80`.
Slot-state reuse requires a compatible continuing prefix, including the previous
response. Removing or rewriting saved tokens can require prompt recomputation
for the hybrid/recurrent model even when the state file was restored successfully.

### Fork additions and what is not active in production

| Addition beyond official upstream | Status on GOKAYA `8090` |
| --- | --- |
| [Context route groups and conversation pinning](tools/server/README.md) | Enabled: one public name selects the two profiles above. |
| Slot-state transfer across a tier swap | Enabled with `--slot-save-path /dev/shm`; a real Qwen migration preserved cached prompt tokens. |
| Cold-autoload request reservations | Enabled; both concurrent cold-load regression cases pass. |
| Responses compatibility and configured-context model metadata | Enabled; Chat and Responses canaries pass and the catalog advertises 98,304. |
| CUDA graph capture/fallback fixes, bounded IQ1_M workspace, IQ1 MMQ kernels and selected-variant initialization | Compiled; numerical comparisons and varied-shape Qwen requests pass. Recoverable CUDA Graph allocation fallback remains observable. |
| [NVIDIA power governor](docs/phase-aware-nvidia-gpu-power-governor.md) and [memory-clock governor](docs/phase-aware-nvidia-gpu-memory-clock-governor.md) | Configured for 200 W prefill, 165 W decode and 11001 MHz decode memory clock. |
| Traditional-KV context fitting / calibration | Available in the binary; production uses explicit tested contexts with `fit=off`. |
| Paged KV, shared multi-slot pools, physical-budget admission, dynamic growth and CPU/GPU page migration | Compiled CUDA/CPU paths; disabled by the current preset (`kv_paged=false`, one slot). |
| `--kv-paged-prealloc-max` automatic pool calibration | Available; disabled in production. |
| SnapKV streaming/per-head/per-sequence scoring and selective retention | Experimental source and CUDA/CPU support present; disabled in production. |
| DFlash2 and stochastic draft verification | Available source path; no DFlash2 draft model is configured. Production speculation uses MTP only. |
| Vulkan paged attention and Android compatibility changes | Present in the fork; the `8090` build has `GGML_VULKAN=OFF`. HIP/ROCm is also not compiled into this release. |
| Paged examples and SnapKV benchmark tools | Operator tools in the source tree; not running as production services. |

Validation for this release: 7 CPU/parser/governor tests, 9 router tests,
130 CUDA numerical comparisons, and 56 sequential Qwen requests.
Both maximum requests were repeated through the router with `load-mode=none`.
Each profile passed an input of `context - 4096`, a requested output of 4096,
and another completion afterward. Production canaries checked both routing
boundaries, real MTP-to-no-MTP cache transfer, Chat, Responses, and loaded-library
hashes. An output of 4094-4096 at the context edge is accepted. These limits
apply to the tested model, GPU, cache format and request sequence.

The current operational record is in
[docs/mtp-router-split-plan.md](docs/mtp-router-split-plan.md).
Earlier capacity and throughput experiments remain in
[docs/kv-calibration-findings.md](docs/kv-calibration-findings.md) and
[docs/oom-reproduction-trace.md](docs/oom-reproduction-trace.md); their old
production profiles are superseded by the values above.

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
