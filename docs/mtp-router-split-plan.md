# Plan: one public model name, two router children (MTP 54k / no-MTP 97k)

Status: design only, not implemented. Written after tracing
`tools/server/server-models.{h,cpp}`, `common/arg.cpp` (preset-only keys), `common/preset.h`,
`common/common.cpp` (`common_fit_normal_kv_context`) and re-deriving the VRAM arithmetic from
`docs/kv-calibration-findings.md` (GOKAYA, `192.168.1.57`).

**This plan supersedes `docs/mtp-adaptive-context-plan.md`**, whose goal is arithmetically
impossible on this model/GPU — see section 1.

## 0. The client contract (the requirement that drives everything)

Clients are **not** changing. Each one is configured with:

- model name: `qwen-3.8-27b` — one name, always
- context: 80,000
- output: 4,096

Whether a request is served with or without MTP must be invisible to them. That has three
hard consequences:

1. **The tier choice is server-side and automatic.** Letting the client name the tier
   (`qwen3-fast` / `qwen3-long`) is off the table as a deliverable — it survives only as a
   staging step to validate the two children before the routing code exists (section 5).
2. **The single-process plan is dead, not just suboptimal.** Its ceiling is ~57,900 tokens
   (section 1); the clients are configured for 80,000. It cannot satisfy the contract at all.
3. **Under-estimating a request's size becomes a client-visible bug**, not a performance
   detail. If a 60k request is routed to the 54,264 child, the client gets an error or a
   truncation on a request its own config says is legal. So the exact `/tokenize` estimator
   (section 6) is mandatory, not an optimization.

Ceiling check: 80,000 (client) + headroom fits inside the **validated** 97,280 of the no-MTP
GGUF, with a real maximum-size request behind that number
(`kv-calibration-findings.md:238-241`). The contract is servable.

## 1. Why one process cannot do it

Both calibration runs share one model: `available = free_vram − margin − overhead`, then
`candidate = available / bytes_per_token`. From the two rows of the final automatic matrix
(`kv-calibration-findings.md:73-78`, both `fit-target 643` MiB, batch/ubatch 64, q4_0 KV,
`parallel=1`):

- `...XXS.gguf` (no MTP head): `2499.6 − 643 − 144.4 = 1712.2` MiB / 97,280 tok
  → **18.0 KiB/token**
- `...-mtp.gguf` (MTP on): `2165.6 − 643 − 502.4 = 1020.2` MiB / 54,272 tok
  → **19.3 KiB/token**

Same architecture and quant, so the target-side KV cost is the same in both and the delta is
the nextn layer's own KV:

| Term | Value | Nature |
|---|---:|---|
| Target attention KV | 18.0 KiB/token | scales with `--ctx-size` |
| MTP nextn KV | **1.2 KiB/token** | scales with the draft budget |
| MTP context compute/workspace | **358 MiB** (502.4 − 144.4) | **fixed**, whenever the MTP context exists |
| Extra weights in the `-mtp` GGUF | **334 MiB** (2499.6 − 2165.6 free VRAM) | **fixed**, whenever that file is loaded |

The MTP draft KV is only **6.8%** of the per-token cost, so capping it buys almost nothing:

| `--mtp-ctx-size` | resulting `--ctx-size` ceiling | MTP covers |
|---:|---:|---|
| 54,264 (today) | 54,272 | everything |
| 32,768 | ~55,700 | prompts < 32k |
| 8,192 | ~57,400 | prompts < 8k |
| 0 (draft KV gone) | **~57,900** | nothing |

**~57,900 is the structural ceiling of the single-process approach** — below the clients'
80,000 — and it is only reached by shrinking the draft budget to zero, i.e. by disabling the
speedup the whole exercise exists to keep. The missing ~39,300 tokens are the two *fixed*
costs: 358 + 334 ≈ 692 MiB ≈ 39,300 tokens at 18.0 KiB/token. Cross-check:
`57,900 + 39,300 = 97,200` vs the measured 97,280 (the fit aligns to 256) — the model closes
to within 0.1%.

Those 692 MiB are recoverable only by not having the MTP head and its context in the process
at all. That is a process boundary. No per-request gating inside one process reaches it.

Confidence: two datapoints, one linear model, but it reproduces both measured candidates and
the implied 6.8% nextn share is sensible for one extra layer. Known wrinkle: the doc's earlier
compute-aware matrix (`:54-59`) reports 96,000 for "mtp GGUF, MTP off" while the same
arithmetic off the final matrix's free-VRAM figure gives ~77,000 for that row — a ~19k
disagreement, most likely different GPU occupancy at measurement time. It does not affect this
plan, which uses the no-MTP GGUF for the long tier precisely because 97,280 there is validated
with a real request.

## 2. Why the swap cost is acceptable

The requests that need the long tier are the ones where a model load is noise:

- 101,488-token prompt: 441.97 prompt tok/s → **230 s of prefill**, 465 s total (`:88-93`)
- 50,176-token prompt: 566.24 prompt tok/s → **89 s of prefill** (`:128-136`)
- so an 80,000-token prompt ≈ **180 s** before the first token

A swap is one model load per direction — sleeping is a full `destroy()`
(`server-context.cpp:964-1005`), so waking is a full load; there is no cheap warm swap. At the
assumed 8 s that is ~16 s against ~180 s, **≈ 9%**. And for these requests the honest baseline
is not "8 s slower" but **"completes" versus "CUDA OOM"**.

## 3. What the router already does (no code)

- **`--models-max 1`** (`common/arg.cpp:3895`) caps residency; the second child evicts the
  first via `unload_lru()` (`server-models.cpp:938-960`).
- **A busy child is never evicted**: candidates are filtered on `req_count != 0`
  (`server-models.cpp:92`, `:203`) and a request with no free slot is **queued**
  (`:113`, `server_lru_sched`). A long request arriving mid-generation waits, then swaps.
- **`ensure_model_ready(name)`** loads on demand and blocks until the child is listening.
- **Per-child args** via `--models-preset file.ini` (`tools/server/README.md:1770-1800`),
  with a `[*]` section for shared defaults.
- **Preset-only keys** already exist as a pattern: `set_preset_only()` + a
  `COMMON_ARG_PRESET_*` env name, read back in `load_models()` with `preset.get_option()`
  (`common/arg.cpp:4985-4999`, consumed at `server-models.cpp:558-589`). The two new routing
  keys in section 4 follow it exactly — no new mechanism.
- **`hidden`** already exists on `server_model_meta` ("hidden from GET /models, but still
  accept if requested"), today only set by the cache-dedup logic.

What is missing is exactly one thing: **one public name resolving to two children by request
size.** Aliases cannot express it — `get_meta()` resolves an alias to a single canonical model
(`server-models.cpp:1811-1812`), so two children cannot share one alias.

## 4. Configuration

`my-models.ini`:

```ini
version = 1

[*]
batch-size = 64
ubatch-size = 64
cache-type-k = q4_0
cache-type-v = q4_0
parallel = 1
flash-attn = on
sleep-idle-seconds = 300

[qwen3-fast]
model = /path/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
ctx-size = 54264
spec-draft-mtp = true          ; whatever this repo names the MTP enable flag
spec-draft-n-max = 2
spec-draft-p-min = 0.80
route-group = qwen-3.8-27b     ; new preset-only key
route-max-tokens = 50000       ; new preset-only key: serve only requests up to this budget

[qwen3-long]
model = /path/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
ctx-size = 97280
route-group = qwen-3.8-27b
; no route-max-tokens = the group's fallback tier
```

```sh
llama-server --models-preset ./my-models.ini --models-max 1
```

`route-max-tokens = 50000` against a 54,264 ceiling leaves 4,264 tokens of margin. That
matters because the boundary is sharp: `54,272` works and `54,273` OOMs
(`kv-calibration-findings.md:144-159`).

## 5. Staging step (config only, no code)

Before the routing code exists, run the same INI **without** the `route-*` keys and have a
test client name `qwen3-fast` / `qwen3-long` directly. This validates the two ceilings, the
real load time (replacing the assumed 8 s), the eviction behavior and the VRAM reclaim with
nothing to code-review. It is not the deliverable — the clients never see it — but it de-risks
everything below and produces the numbers section 7's thresholds depend on.

## 6. The routing code

### 6.1 Group resolution — deliberately outside `mapping`

Do **not** insert a synthetic group entry into `server_models::mapping`. Every existing
invariant (LRU counting, `load()`, `req_count`, status transitions) assumes a mapping entry is
a loadable child. Instead keep a separate, read-mostly map built in `load_models()`:

```
route_groups : public_name -> [ {member_name, max_tokens}, ... ]   // ordered, ascending cap
```

One helper resolves a public name to a concrete child, and every existing route stays
untouched:

```
std::string resolve_route_target(name, /* optional */ body)
```

- name is not a group → return it unchanged (all current behavior preserved)
- name is a group, `body` present → estimate the budget (6.2) and return the first member
  whose `max_tokens` covers it, else the uncapped member
- name is a group, no `body` (GET `/props`, `/tokenize`, …) → return the currently loaded
  member, else the group's uncapped member

Call it in `proxy_post` right after `json::parse(req.body)`, before
`router_validate_model` (`server-models.cpp:1914-1920`). Everything downstream —
validation, `conv_models.remember`, `ensure_model_ready` (load/queue/evict) and
`proxy_request` — already works with whatever `name` holds, so the whole feature is that one
substitution. Also call it in `proxy_get` for the non-body routes.

### 6.2 The estimator

The router parent has **no tokenizer** — it never loads a model, it spawns children and
proxies HTTP. Given consequence 3 of the contract, undershoot is a client-visible failure, so:

- **Exact path (required):** `POST /tokenize` to whichever member is *already loaded*.
  Tokenization needs no KV cache and no context space, so the 54k child tokenizes an 80k
  prompt fine. One localhost round-trip, a few ms.
- **Cold start (no member loaded):** body bytes ÷ 3.5, and **bias to the long tier** on ties
  or doubt. Loading a model just to count tokens would cost more than routing long and being
  wrong.
- **Always add the output reserve:** the budget is `prompt + n_predict`. When the request omits
  `max_tokens`/`n_predict`, assume the router's `-n` default (4,096 here, matching the client
  config) rather than 0 — otherwise a 52k prompt with an implicit 4k output routes to the 54,264
  child and dies at the boundary.

### 6.3 Promote-only within a conversation

`conv_model_tracker` (`server-models.h:145-197`) already maps `X-Conversation-Id` → serving
child, and `proxy_post` re-`remember()`s it on every POST *after* name resolution
(`server-models.cpp:1928`) — so a migrating conversation re-pins itself with no extra code.

Add one rule: **if a conversation is already pinned to the long tier, keep it there.** Never
demote. Conversations grow monotonically, so this makes each one migrate at most once instead
of flapping around the threshold, and it costs a single map lookup. Without it, a conversation
hovering near 50k would swap models on nearly every turn — and each swap also throws away the
prompt cache (see 7).

### 6.4 `/v1/models` (same phase, cheap)

Mark both members `hidden` (extend the existing flag with a preset-only key, or derive it from
`route-group` membership) and append one synthetic entry for the group in the listing loop
(`server-models.cpp:1995-2035`). Advertise the long tier's context so a client that inspects
the list sees the 97,280/80,000 story rather than a 54,264 ceiling. This is cosmetic for the
current clients — they already know the name — but any OpenAI-compatible library that
validates the model list would otherwise break on a name that answers requests yet is absent
from `/v1/models`.

## 7. What clients WILL notice (transparency is not free)

Worth agreeing on these before implementing, because they are the price of the contract:

- **TTFT spikes.** A small request that follows a big one pays the swap: ~8 s of first-token
  latency on a request that would otherwise answer in ~2 s. Same single model name, wildly
  variable TTFT. Unavoidable at `--models-max 1`; the mitigation is 6.3 plus, if measurement
  justifies it, a hysteresis timer keeping the long tier resident for N seconds.
- **Prompt cache loss on migration.** A swap discards the child's cached prefix. A 60k
  conversation crossing the threshold re-prefills from scratch, ~110 s. 6.3 limits this to
  once per conversation; interleaved conversations of mixed sizes are the bad case.
- **The two GGUFs are not bit-identical.** Same base model and same quant class, so quality is
  comparable and MTP itself is distribution-preserving (every draft token is verified against
  the target), but the two files are separate artifacts. A conversation that migrates tiers can
  show a subtle style shift mid-thread. If that is unacceptable, evaluate running both children
  off the same `-mtp` GGUF (one with MTP, one without) — it also warms the page cache for both
  and halves disk, at the cost of the 334 MiB of extra weights in the long child, which is
  exactly the row the two calibration matrices disagree about (96,000 vs ~77,000) and would need
  one measurement.
- **Throughput asymmetry stays visible.** ~47 decode tok/s under 50k, ~21 above it. Nothing to
  do about that — it is the point of the exercise.

## 8. Files touched

Staging step (section 5): none — one INI file outside the repo.

Routing:

| File | Change |
|---|---|
| `common/arg.cpp` | two `set_preset_only()` keys: `route-group`, `route-max-tokens` |
| `tools/server/server-models.h` | `route_groups` map + `resolve_route_target()` declaration |
| `tools/server/server-models.cpp` | build `route_groups` in `load_models()`; `resolve_route_target()` (estimator + promote-only rule); call it in `proxy_post` (`:1917`) and `proxy_get`; hide members and emit the group entry in the `/v1/models` loop |
| `tools/server/README.md` | document the routing group, the estimator fallback and the TTFT/prompt-cache caveats |

Nothing in `common/speculative.cpp`, nothing in `server-context.cpp`, no change to
`can_speculate()`, no new inference-side flag.

## 9. Validation

1. **Staging, both tiers named explicitly.** `qwen3-fast` with 50,168 prompt + 4,096 output →
   HTTP 200, ~47 decode tok/s. `qwen3-long` with 93,184 + 4,096 → HTTP 200, ~21 decode tok/s.
   Must match `kv-calibration-findings.md:414` and `:238` within noise.
2. **Swap cost and VRAM reclaim.** Time the second request's queue-to-first-token delta and
   watch `nvidia-smi` across the eviction: VRAM must return to the idle baseline between
   children. The measured load time replaces the assumed 8 s everywhere in this doc.
3. **A busy child is never evicted.** Start a long generation on the fast tier, then fire a
   long-tier request: the first completes normally, the second queues and only then swaps. The
   whole design leans on this (`server-models.cpp:92`, `:203`, `:113`).
4. **Transparency, the actual acceptance test.** With only `qwen-3.8-27b` configured, send
   8k / 40k / 52k / 60k / 78k-token prompts each with `max_tokens=4096`. All must return HTTP
   200; log which child served each; the 52k case must land on the long tier (52k + 4k > 50k
   cap) — that is the boundary the estimator exists to get right.
5. **Estimator accuracy**, Portuguese and English, `/tokenize` truth vs bytes ÷ 3.5, at the same
   sizes. Record the worst-case undershoot; it must stay inside the 4,264-token margin.
6. **Omitted `max_tokens`.** A 52k prompt with no `max_tokens` must still route long
   (implicit 4k reserve applied).
7. **Promote-only.** A conversation growing 40k → 52k → 55k must migrate once and never come
   back; count the model loads (exactly one).
8. **Thrash cost.** Alternate small/large 5× and record total wall clock, to decide whether a
   hysteresis timer is worth building.

## 10. Sequencing

1. Section 5 staging config — today, no code, produces the real load-time and ceiling numbers.
2. Sections 6.1–6.3 — the routing itself. This is the deliverable the clients need.
3. Section 6.4 — `/v1/models` cosmetics, same phase if cheap.
4. Hysteresis only if test 8 says so.

Keep `docs/mtp-adaptive-context-plan.md` for its code-path analysis — the always-on
`common_speculative_process()` shadow decode and the five static-capability uses of
`can_speculate()` are correct findings worth having on record — but do not implement it.
