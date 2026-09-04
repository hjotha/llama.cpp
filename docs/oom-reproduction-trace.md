# OOM reproduction trace

Status: root cause isolated to repeated CUDA graph capture/update under very
tight VRAM. A backend fix was validated with CUDA graphs enabled. This record
is for the RTX 4070 12 GB production model and configuration below, not a
general CUDA limit.

## Configuration

- Model: `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
- GPU: CUDA0, GeForce RTX 4070, 12,282 MiB reported by CUDA
- KV: traditional, K/V `q4_0`, `--kv-paged` absent
- MTP: `draft-mtp`, `n-max=2`, `p-min=0.80`
- Context: `--ctx-size 54272`, one slot
- Compute: `--batch-size 64`, `--ubatch-size 64`, Flash Attention
- `--fit off`
- Power governor: prefill 200 W, decode 165 W
- Prompt cache: tested with both `--cache-ram 4096` and `--cache-ram 0`

All prompt sizes in this document were measured by the running server's
`POST /tokenize` endpoint. Chat sizes were measured after rendering through
`POST /apply-template`.

## Original failure

The earlier OpenCode request reached 38,309 prompt tokens and aborted during
prefill:

```text
ggml_cuda_compute_forward: MUL_MAT failed
CUDA error: out of memory
current device: 0, in function launch_mul_mat_q
```

The process then produced a core dump and systemd restarted it. A later
controlled reproduction reached the same failure in a chat request at 50,090
processed tokens, with the stack including:

```text
ggml_cuda_graph_evaluate_and_capture
llama_context::graph_compute
llama_context::process_ubatch
llama_context::decode
server_context_impl::update_slots
```

The client observed `curl: (52) Empty reply from server`; systemd recorded a
core dump and incremented the restart counter.

## Sequence that reproduces it

After a clean server start, the following completion requests all succeeded,
in this order. Each request used `cache_prompt=true` and the output budget
shown below:

| Step | Endpoint | Prompt tokens | `max_tokens` | Result |
|---:|---|---:|---:|---|
| 1 | completion | 36,864 | 64 | pass |
| 2 | completion | 38,308 | 4,096 | pass |
| 3 | completion | 50,176 | 4,096 | pass; 4,094 at boundary |
| 4 | completion | 43,008 | 256 | pass |
| 5 | completion | 48,128 | 1,024 | pass |
| 6 | completion | 24,576 | 256 | pass |
| 7 | completion | 40,960 | 512 | pass |
| 8 | completion | 32,768 | 1,024 | pass |
| 9 | chat | 38,359 | 4,096 | pass; model stopped at 151 |
| 10 | chat | 50,175 | 4,096 | OOM, empty reply, server restart |

The same final chat request passed as the first request after a clean start,
and also passed after only `chat 38,359 -> chat 50,175`. The failure therefore
depends on accumulated allocations/graph shapes rather than the request size
alone.

## Cache A/B

The complete sequence above was run with `--cache-ram 4096` and again with
`--cache-ram 0`.

- With `4096`, the sequence failed at the final chat request. Prompt-cache
  entries of approximately 1,032--1,372 MiB were created and evicted, and
  process RSS reached several GiB.
- With `0`, the same sequence still failed at the final chat request with the
  same CUDA OOM/core-dump signature. RSS stayed around 1.34--1.70 GiB and
  device memory remained around 11,901 MiB before the failure.
- A separate `4096` test confirmed real cache reuse (`cache_n=50172`) without
  OOM.

Thus prompt cache can add host-memory pressure and state-copy work, but it is
not the necessary trigger for this failure.

## CUDA graph A/B

The same eight completions followed by the two chats were run with
`--cache-ram 4096` and the environment variable:

```text
GGML_CUDA_DISABLE_GRAPHS=1
```

All ten requests passed. The final chat processed 50,175 prompt tokens and
stopped after 140 generated tokens. The process remained alive with
`NRestarts=0`; device memory stayed at approximately 11,881--11,883 MiB.

This is the strongest current evidence that the failure is in accumulated
CUDA graph capture/replay or device-allocation fragmentation after variable
graph shapes. It is not proof of a single leaked allocation: the CUDA
allocator may retain graph/workspace blocks by design, or the fork's graph
inputs may prevent correct reuse/eviction.

Disabling context checkpoints was also tested separately. With CUDA graphs
enabled, `--cache-ram 4096`, and `--ctx-checkpoints 0`, the process still
crashed during the third request of the sequence. The log contained zero
context-checkpoint events. Checkpoints can add memory pressure, but they are
not the root cause for this model and reproduction.

## Clean upstream comparison

The intact upstream binaries passed a first-request chat at 50,175 tokens:

- `llama2` build 10807, commit `163a40796`: pass
- upstream build 10805, commit `86b351fd6`: pass

The complete ten-request sequence was then run on the compiled `llama2`
upstream binary. It reproduced the same failure in the final chat at 50,090
processed tokens, with `MUL_MAT failed`, CUDA OOM, and a core dump. This rules
out the power governor and the fork's paged-KV additions as the source. The
fork and upstream builds used the same CUDA architecture (`89`), Flash
Attention, CUDA graphs enabled, and MMQ settings.

## Root cause and backend fix

Instrumentation was added behind `GGML_CUDA_GRAPH_DEBUG=1`. A short request
showed two graph-cache keys and that the first executable graph instantiation
reduced free device memory from 21.62 MiB to 5.62 MiB. Together with the
time-based cache eviction and stable `nvidia-smi` total, this favors repeated
updates/fragmentation of hot entries over a simple large-block leak, although
the original failing run did not have this instrumentation enabled.

Traditional KV rounds its active size in blocks while a long prompt is split
into 64-token ubatches. The same graph shape therefore appears briefly,
becomes eligible after one matching warmup call, is captured, and then changes
again as KV grows. Repeating this cycle across varied long requests leaves the
CUDA driver with almost no contiguous margin for the next capture or MMQ
workspace.

The backend fix makes three changes:

1. Require four structurally identical calls before enabling a CUDA graph.
   The short-lived prefill shapes never reach that threshold, while stable
   decode starts using graphs after a few tokens.
2. Avoid calling `cudaGraphExecUpdate()` immediately after first
   instantiation of the same captured graph.
3. Synchronize the graph stream before destroying and reinstantiating an
   executable graph after an incompatible update.

Optional diagnostics report graph key, cache-entry count, capture/update
counters, stable-call count, and CUDA free memory:

```text
GGML_CUDA_GRAPH_DEBUG=1
```

The corrected binary passed all ten requests with CUDA graphs enabled,
`--cache-ram 4096`, and `--ctx-checkpoints 1`. The final 50,175-token chat
returned HTTP 200, the process remained alive, and no restart occurred.
Across the eight completion requests, the previous graph-enabled behavior
averaged 586.990 prompt tok/s and 49.113 decode tok/s; the fix averaged 586.261
and 49.107 tok/s respectively (approximately -0.12% and -0.01%). Production
was therefore restored with CUDA graphs enabled; `GGML_CUDA_DISABLE_GRAPHS`
is no longer set.

## External correlation

The upstream reports [#20315](https://github.com/ggml-org/llama.cpp/issues/20315),
[#23181](https://github.com/ggml-org/llama.cpp/issues/23181), and
[#25835](https://github.com/ggml-org/llama.cpp/issues/25835) describe the same
class of failure: varied requests eventually fail in CUDA graph capture or
instantiation, while `GGML_CUDA_DISABLE_GRAPHS=1` prevents the crash. The
current upstream cache uses the first graph-node pointer as a key and evicts
entries only by a time-to-live sweep. The
[NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
documents that graph executable resources and references can remain associated
with graph lifetime and that destroying a graph executable does not itself
synchronize prior work.

## Reproduction artifacts

The detailed client reports from the runs are currently stored at:

- `/tmp/llama-random-suite.OdRmG3/cases.txt`
- `/tmp/llama-cache0-sequence.9Cwb0b/report.txt`
- `/tmp/llama-graphs-disabled-sequence2.kYFBxb/report.txt`
- `/tmp/llama-checkpoints0-sequence.8G2nA5/report.txt`
- `/tmp/llama2-upstream-sequence.l8ZaNH/report.txt`
- `/tmp/llama-graph-fix-sequence.7LfhAi/report.txt`
- `/tmp/llama-graph-debug-short.log`
