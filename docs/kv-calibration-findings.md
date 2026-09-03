# KV calibration findings

Status: investigation in progress on GOKAYA (`192.168.1.57`). Updated after
the compute-aware probe, sentinel fix, and live `8091` tests.

All measurements below use CUDA0, `q4_0` K/V, `parallel=1`, unless stated
otherwise. A context value is the total per-slot budget: prompt plus output.

## Goal

Make `--ctx-size 0` report a usable context size for the four combinations:

1. normal/traditional KV without MTP;
2. normal/traditional KV with MTP;
3. paged KV without MTP;
4. paged KV with MTP.

The usable value must account for the actual `batch-size`/`ubatch-size`, model
compute buffers, and MTP draft resources, not only the KV byte estimate.

## Earlier matrix: Qwen3.8-27B-UD-IQ3_XXS, batch/ubatch 64, fit-target 643

| KV path | MTP | Reported candidate | Runtime result |
|---|---:|---:|---|
| traditional | no | 79,360 | model loaded/listening |
| traditional | yes | 53,504 | OOM allocating a 225.80 MiB compute buffer |
| paged | no | 77,824 | model loaded/listening |
| paged | yes | 43,008 | model loaded/listening |

The MTP traditional candidate was not usable despite the successful arithmetic.

## ISTA-DASLab GGQ-RCO-IQ3_XXS-mtp matrix, batch/ubatch 64, fit-target 643

Model:
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`

| KV path | MTP | Reported candidate | Runtime result |
|---|---:|---:|---|
| traditional | no | 105,472 | startup passed; maximum request later OOMed |
| traditional | yes | 78,080 | OOM during context creation |
| paged | no | 104,448 | pool froze, then final initialization OOMed |
| paged | yes | 67,584 | startup and fixed-pool calibration passed |

The high reported limits are therefore not automatically usable request
limits. The 67,584 MTP-paged case also became impractically slow in the
reference paged-attention path during a very large prefill request.

## Compute-aware matrix: GSQ-RCO MTP model, batch/ubatch 64, fit-target 643

The calibration now probes a small real target context and, when enabled, the
MTP draft context. It measures the fixed context/compute overhead before
extrapolating the per-token or per-block KV cost.

| KV path | MTP | Probe overhead | New candidate | Startup result |
|---|---:|---:|---:|---|
| traditional | no | 165.2 MiB | 96,000 | model loaded/listening |
| traditional | yes | 486.0 MiB | 51,968 | model loaded/listening |
| paged | no | 164.9 MiB | 94,208 | model loaded/listening |
| paged | yes | 484.2 MiB | 41,984 | model loaded/listening |

The probe logs include `batch=64` and `ubatch=64`. The `fit-target` is
subtracted separately (`643 MiB` here); MTP adds its explicit workspace
reserve. These are safer startup candidates than the prior KV-only values,
but a maximum request should still be used as the final gate.

## Final compute-aware matrix: GSQ-RCO MTP model, batch/ubatch 64, fit-target 643

The probe now treats the `--ctx-size 0` fit sentinel as a 4096-token probe
floor. All four cases reached `model loaded` and `listening` on the live
`8091` test port:

| KV path | MTP | Probe context | Probe overhead | Candidate context |
|---|---:|---:|---:|---:|
| traditional | no | 4,096 | 176.4 MiB | 95,488 |
| traditional | yes | 4,096 | 502.4 MiB | 50,944 |
| paged | no | 4,096 | 164.9 MiB | 94,208 |
| paged | yes | 4,096 | 480.9 MiB | 41,984 |

This is the current recommended matrix for this GGUF and these runtime
parameters. The service on `8090` was intentionally left stopped while the
tests ran; no production unit was changed.

## Direct request checks

For the previous UD model, traditional KV without MTP was tested with a
56,320-token context, 52,223 prompt tokens, and 4,096 output tokens. It
completed 4,096 output tokens without truncation.

For the GSQ-RCO model, a traditional no-MTP request at the reported 105,472
candidate failed with CUDA OOM. An MTP-traditional request at 78,080 could not
start because context creation failed first. These checks show that the
KV-only candidate is not sufficient as a production threshold.

The GSQ-RCO MTP-paged path was also tested at an explicit 44,032-token context
with 39,936 prompt tokens and a 4,096-token output limit. It completed 4,094
output tokens and reached:

```text
prefill: 496.71 tokens/s (80.401 s / 39,936 tokens)
decode:   36.97 tokens/s (110.706 s / 4,094 tokens)
total:   191.108 s / 44,030 tokens
```

The server reported `truncated=1` at the exact boundary, so 44,032 is a
working threshold but not yet a clean full-4,096-token boundary. The new
compute-aware candidate of 41,984 leaves additional headroom.

The same MTP-paged path was tested at the new 41,984-token candidate with a
37,888-token prompt and a 4,096-token output limit. It completed 4,094 output
tokens without OOM, at approximately 502.34 prompt tokens/s and 38.25 decode
tokens/s; total logged time was 182.435 s for 41,982 processed tokens. The
server again marked the exact boundary as `truncated=1`, so client-side output
reserve should leave a small additional margin when strict completion of all
4,096 tokens is required.

The MTP-traditional candidate was then tested at the exact `50,944`-token
context boundary with a `46,848`-token prompt (`50,944 - 4,096`) and
`max_tokens=4096`. The request returned HTTP 200 without OOM, with
`prompt_tokens=46848`, `completion_tokens=4094`, `total_tokens=50942`, and
`finish_reason=length`. The server metrics recorded 566.24 prompt tokens/s,
48.03 decode tokens/s, and 167.948 s total time. The two-token difference is
the server's boundary/protocol behavior, so the candidate is operationally
valid but does not yield a literal 4,096 generated tokens at the exact
context edge.

## Absolute MTP-traditional request boundary

To separate the startup limit from the usable request limit, the same server
configuration was tested with `prompt = ctx-size - 4,096` and
`max_tokens=4096`. The search narrowed the transition to one token:

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 53,120 | 49,024 | HTTP 200, 4,096 generated |
| 54,208 | 50,112 | HTTP 200, 4,096 generated |
| 54,240 | 50,144 | HTTP 200, 4,096 generated |
| 54,256 | 50,160 | HTTP 200, 4,096 generated |
| 54,264 | 50,168 | HTTP 200, 4,096 generated |
| 54,272 | 50,176 | HTTP 200, 4,094 generated; boundary truncation |
| 54,273 | 50,177 | CUDA OOM during request |

`54,272` is therefore the highest context that accepted this maximum-size
request in two independent runs; `54,273` was the first failing integer. If
the requirement is literally all 4,096 generated tokens, `54,264` is the
highest value validated. This is an operational absolute for the tested
model, GPU state, MTP settings, q4_0 KV, and batch/ubatch 64—not a hardware
invariant; changing any of those can move the boundary.

## Paged MTP memory versus throughput boundary

The paged search used dynamic growth with a 16-token block and enough
`--n-gpu-blocks`/`--kv-paged-admission-blocks` for the requested context. This
is distinct from the calibrated fixed pool of 41,984 tokens; omitting those
capacity values causes an artificial `Context size has been exceeded` error.

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 41,984 | 37,888 | HTTP 200, 4,094 generated |
| 44,032 | 39,936 | HTTP 200, 4,094 generated |
| 54,272 | 50,176 | HTTP 200, 4,094 generated |
| 59,392 | 55,296 | HTTP 200, 4,094 generated |
| 60,416 | 56,320 | CUDA OOM during prefill |
| 63,488 | 59,392 | CUDA OOM during prefill |
| 67,584 | 63,488 | CUDA OOM during prefill |

The maximum paged context that passed the full request test was therefore
59,392 tokens, but it is not a usable replacement target: its measured
throughput fell to 475.18 prompt tok/s and 30.45 decode tok/s, versus
496.71 prompt tok/s and 36.97 decode tok/s at 44,032. Since the reference
paged-attention implementation shows a clear throughput collapse as the
context grows, `44,032` is the current practical ceiling; `41,984` remains the
safer calibrated value. The memory ceiling and the performance ceiling must
not be reported as the same number.

With the explicit collapse rule of decode throughput below `10 tok/s`, the
last stable paged-MTP point was refined to `60,160` tokens. Its request used a
56,064-token prompt and completed with `30.11` decode tok/s. The next paged
allocation step, `60,176` tokens, required one additional 16-token block and
repeatedly failed to allocate approximately 63 MiB on CUDA before decode
could start. Thus `60,160` is the last validated non-collapsed value under
this exact configuration; the failure is an OOM boundary rather than a
measured sub-10 tok/s decode.

## Traditional versus paged MTP performance

At the same `ctx-size=54,272`, with the same 50,176-token prompt and
`max_tokens=4096`, the measured comparison was:

| Metric | Traditional KV | Paged KV | Paged delta |
|---|---:|---:|---:|
| Prefill | 549.13 tok/s | 485.01 tok/s | 11.7% slower |
| Decode | 46.66 tok/s | 32.26 tok/s | 30.9% slower |
| Total time | 179.10 s | 230.32 s | +51.2 s |

At each path's maximum validated context, traditional KV reached 54,272
tokens at 549.13 prompt tok/s and 46.66 decode tok/s, while paged KV reached
60,160 tokens at 471.00 prompt tok/s and 30.11 decode tok/s. Paged KV gained
5,888 context tokens, but lost approximately 78 prompt tok/s and 16.5 decode
tok/s. These values are for the current build, where the log reports that
paged attention is using the CPU reference implementation for correctness
validation; it is explicitly not optimized. The measured speed penalty is
therefore not an inherent limit of paged KV and may change with an optimized
backend.

No completed paged request reached the defined collapse threshold of decode
below 10 tok/s; the higher-context failures occurred first as CUDA OOM.

## ISTA GGUF without MTP: absolute request limits

The non-MTP ISTA file is
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`. The same maximum-request test was used in
both KV modes: `prompt = ctx-size - 4,096`, `max_tokens=4096`, batch/ubatch
64, q4_0 KV, CUDA0, and no `--spec-*` flags.

For traditional KV, the search ended at:

| `ctx-size` | Prompt | Result |
|---:|---:|---|
| 95,488 | 91,392 | HTTP 200, 4,096 generated |
| 96,736 | 92,640 | HTTP 200, 4,096 generated |
| 97,048 | 92,952 | HTTP 200, 4,096 generated |
| 97,204 | 93,108 | HTTP 200, 4,096 generated |
| 97,243 | 93,147 | HTTP 200, 4,096 generated |
| 97,272 | 93,176 | HTTP 200, 4,096 generated |
| 97,277 | 93,181 | HTTP 200, 4,096 generated |
| 97,280 | 93,184 | HTTP 200, 4,096 generated |
| 97,281 | 93,185 | CUDA OOM during request |

`97,280` is the highest traditional-KV context validated, with `456.42`
prompt tok/s, `21.19` decode tok/s, and `397.444 s` total time.

For paged KV, dynamic growth was enabled and the physical/admission pool was
set to the number of 16-token blocks needed by each candidate:

| `ctx-size` | Prompt | GPU blocks | Result |
|---:|---:|---:|---|
| 94,208 | 90,112 | 5,888 | HTTP 200, 4,096 generated |
| 99,328 | 95,232 | 6,208 | HTTP 200, 4,096 generated |
| 101,888 | 97,792 | 6,368 | HTTP 200, 4,096 generated |
| 104,448 | 100,352 | 6,528 | HTTP 200, 4,096 generated |
| 105,472 | 101,376 | 6,592 | HTTP 200, 4,096 generated |
| 105,536 | 101,440 | 6,596 | HTTP 200, 4,096 generated |
| 105,568 | 101,472 | 6,598 | HTTP 200, 4,096 generated |
| 105,584 | 101,488 | 6,599 | HTTP 200, 4,096 generated |
| 105,600 | 101,504 | 6,600 | CUDA OOM during prefill |

`105,584` is the highest paged-KV context validated, with `439.02` prompt
tok/s, `17.27` decode tok/s, and `468.267 s` total time. It adds 8,304
tokens over traditional KV, while decode is approximately 3.92 tok/s slower.
The paged-attention path still uses the unoptimized CPU reference
implementation noted above, so this speed difference is implementation-
dependent rather than an inherent KV-cache limit.

## Calibration implementation observations

- The normal KV estimator now counts only regular attention layers for hybrid
  target models and adds appended `nextn` layers when MTP is enabled.
- `fit-target` is a VRAM safety margin. With `--fit-target 643`, the normal
  MTP path uses an additional 64 MiB margin; paged MTP preallocation reserves
  an additional 224 MiB speculative workspace.
- `batch-size` and `ubatch-size` do not appear directly in the KV byte formula,
  but they change compute-graph memory. The new probe measures that fixed
  overhead using the configured values before calculating the usable
  threshold.
- There is no separate `block unlock` option in this source. Paged
  `--kv-paged-prealloc-max` freezes the calibrated physical pool and sets the
  final admission capacity to the measured pool size.

## Threshold validation status

The first conservative functional threshold for the GSQ-RCO MTP-paged model
was around 44k tokens, based on the previous model's successful 43,008-token
MTP-paged calibration. Requests at 44,032 and 41,984 tokens completed without
OOM, although the server stopped at 4,094 generated tokens at the exact
boundary. The MTP-traditional candidate at 50,944 shows the same two-token
boundary behavior. A small client-side reserve below the advertised context
limit is therefore still advisable when exactly 4,096 generated tokens are
required.
