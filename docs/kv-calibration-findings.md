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
