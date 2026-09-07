# Plan: one public model name, a ladder of router children (MTP 54k / no-MTP 76k / overflow 97k)

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
   (`qwen3-fast` / `qwen3-mid` / `qwen3-long`) is off the table as a deliverable — it survives
   only as a staging step to validate the children before the routing code exists (section 5).
2. **The single-process plan is dead, not just suboptimal.** Its ceiling is ~57,900 tokens
   (section 1); the clients are configured for 80,000. It cannot satisfy the contract at all.
3. **Under-estimating a request's size becomes a client-visible bug**, not a performance
   detail. If a 60k request is routed to the 54,264 child, the client gets an error or a
   truncation on a request its own config says is legal. So the exact `/tokenize` estimator
   (section 6) is mandatory, not an optimization.

Ceiling check: 80,000 (client) + 4,096 output = 84,096 fits inside the **validated** 97,280 of
the no-MTP GGUF, with a real maximum-size request behind that number
(`kv-calibration-findings.md:238-241`). The contract is servable.

### 0.1 Three tiers, because two of them can share weights

The group is a ladder of members, ascending cap (the resolution in 6.1 is already an ordered
list, so a third member costs one INI section and zero code):

| Member | GGUF | MTP | `ctx-size` | Serves | Why it exists |
|---|---|---|---:|---|---|
| `qwen3-fast` | `-mtp.gguf` | on | 54,264 | ≤ 50k budget | the speedup, ~47 decode tok/s |
| `qwen3-mid` | `-mtp.gguf` | off | ~76,000 (measure) | 50k–72k | **same weights as fast** → no style shift, and KV state transfer becomes legal (section 11) |
| `qwen3-long` | `...XXS.gguf` | off | 97,280 | > 72k, uncapped fallback | keeps the 80,000 contract honest for the rare oversized request |

`qwen3-mid` is the workhorse of the long half: same file as `qwen3-fast`, so a conversation
crossing 50k keeps identical weights and can in principle carry its KV across (section 11).
`qwen3-long` exists only so that a prompt above ~72k still answers instead of erroring — without
it, the 80,000 in the client config would be a lie. It should be rare; when it happens it costs
a second GGUF load and, mid-thread, a possible subtle style shift (section 7).

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
the implied 6.8% nextn share is sensible for one extra layer.

### 1.1 The `-mtp` GGUF with MTP off tops out around 77,200

The earlier compute-aware matrix (`kv-calibration-findings.md:54-59`) is titled "GSQ-RCO MTP
model" and its `MTP = no` row reports 96,000, which reads like the `-mtp` file reaching 96k with
MTP disabled. It is not: that matrix predates per-GGUF accounting, as the next section says
outright — "the model size, free VRAM, KV geometry, and probe result are **not** reused between
the MTP and non-MTP files" (`:66-69`). The 96,000 reproduces exactly from the *non*-MTP file's
free VRAM: `2,499.6 − 643 − 165.2 = 1,691.4` MiB / 18.0 KiB/token = **96,200**.

Redo it with the `-mtp` file's own free VRAM and its MTP-off probe overhead:
`2,165.6 − 643 − 165.2 = 1,357.4` MiB / 18.0 KiB/token ≈ **77,200 tokens**. The 334 MiB of extra
weights stay resident whether the head is used or not, so the ceiling drops with them.

Consequence: the `-mtp` file with MTP off **cannot** carry the full contract — 84,096 is out of
reach by ~7k tokens (~121 MiB) — but it comfortably covers a **~72k prompt + 4k output**. Do not
try to close the ~7k with the 643 MiB `fit-params-target` margin: the ceiling is a cliff (54,272
works, 54,273 OOMs, `:144-159`) and that margin is exactly what keeps a maximum-size request from
dying at the boundary.

So this becomes `qwen3-mid` (section 0.1), capped at 72,000, with `qwen3-long` on the no-MTP file
behind it for the rare overflow. Two things follow: the same weights as `qwen3-fast` make section
11's state transfer legal again, and a conversation that grows past 50k sees no model change at
all until it passes 72k.

**77,200 is derived, not measured**, and the two plausible probe-overhead figures bracket it at
77,200–78,400. So `ctx-size` for `qwen3-mid` must come from a measurement, not from this
paragraph — staging (section 5) produces it with one extra INI section, and the `route-max-tokens`
cap is then set to (measured − 4,096), rounded down.

## 2. Why the swap cost is acceptable

The requests that need the long tier are the ones where a model load is noise:

- 101,488-token prompt: 441.97 prompt tok/s → **230 s of prefill**, 465 s total (`:88-93`)
- 50,176-token prompt: 566.24 prompt tok/s → **89 s of prefill** (`:128-136`)
- so an 80,000-token prompt ≈ **180 s** before the first token

A swap is one model load per direction — sleeping is a full `destroy()`
(`server-context.cpp:964-1005`), so waking is a full load; there is no cheap warm swap. At the
assumed 8 s that is ~16 s against ~180 s, **≈ 9%**. And for these requests the honest baseline
is not "8 s slower" but **"completes" versus "CUDA OOM"**. The fast↔mid swap is cheaper still:
same file, so the second load reads a warm page cache (measure it, section 5 item 2).

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

What is missing is exactly one thing: **one public name resolving to several children by request
size.** Aliases cannot express it — `get_meta()` resolves an alias to a single canonical model
(`server-models.cpp:1811-1812`), so several children cannot share one alias.

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

[qwen3-mid]
model = /path/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf   ; same file as qwen3-fast, MTP off
ctx-size = 76000               ; PLACEHOLDER - replace with the staged measurement (1.1)
route-group = qwen-3.8-27b
route-max-tokens = 72000       ; = ctx-size - 4096, rounded down

[qwen3-long]
model = /path/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
ctx-size = 97280
route-group = qwen-3.8-27b
; no route-max-tokens = the group's uncapped fallback, for prompts above ~72k
```

```sh
llama-server --models-preset ./my-models.ini --models-max 1
```

`route-max-tokens = 50000` against a 54,264 ceiling leaves 4,264 tokens of margin. That
matters because the boundary is sharp: `54,272` works and `54,273` OOMs
(`kv-calibration-findings.md:144-159`). Apply the same rule to `qwen3-mid`: cap = ceiling − 4,096
output − a few hundred of slack, never the ceiling itself.

`qwen3-fast` and `qwen3-mid` name the **same file**, so the second load hits a warm page cache and
the two never coexist anyway at `--models-max 1`.

## 5. Staging step (config only, no code)

Before the routing code exists, run the same INI **without** the `route-*` keys and have a
test client name `qwen3-fast` / `qwen3-mid` / `qwen3-long` directly. Nothing to code-review, and
it produces every number the rest of the plan assumes:

1. **The `qwen3-mid` ceiling** — the one figure the whole three-tier shape rests on. Run it with
   `ctx-size = 0` and read the candidate the fit probe picks; then pin that value explicitly and
   send a maximum-size request to confirm it is a *usable* limit, not just a startup one
   (`kv-calibration-findings.md:36-46` is the cautionary tale). Expect 77k–78k per 1.1. If it
   lands materially lower, re-derive the caps; if it somehow clears 84,096, drop `qwen3-long`
   entirely and the plan gets simpler.
2. **The real load time**, replacing the assumed 8 s — separately for a cold file and for the warm
   `-mtp.gguf` page cache that a fast↔mid swap sees.
3. **The two known ceilings** still hold: 50,168 + 4,096 on fast, 93,184 + 4,096 on long.
4. **Eviction and VRAM reclaim** across each pair, via `nvidia-smi`.

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
  member, else the group's **largest-capped** member (`qwen3-mid`), not the uncapped overflow —
  `/props` should not be able to boot the rarely-used third child

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
- **Cold start (no member loaded):** body bytes ÷ 3.5, and **bias up the ladder** on ties or
  doubt. Loading a model just to count tokens would cost more than routing one tier too high and
  being wrong.
- **Always add the output reserve:** the budget is `prompt + n_predict`. When the request omits
  `max_tokens`/`n_predict`, assume the router's `-n` default (4,096 here, matching the client
  config) rather than 0 — otherwise a 52k prompt with an implicit 4k output routes to the 54,264
  child and dies at the boundary.

### 6.3 No demotion within a conversation

`conv_model_tracker` (`server-models.h:145-197`) already maps `X-Conversation-Id` → serving
child, and `proxy_post` re-`remember()`s it on every POST *after* name resolution
(`server-models.cpp:1928`) — so a migrating conversation re-pins itself with no extra code.

Add one rule: **a conversation never moves down the ladder.** If it is pinned to `qwen3-mid`,
a turn that estimates under 50k still goes to `qwen3-mid`; same for `qwen3-long`. Conversations
grow monotonically, so this makes each one migrate at most twice instead of flapping around a
threshold, and it costs a single map lookup. Without it, a conversation hovering near 50k would
swap models on nearly every turn — and each swap also throws away the prompt cache (see 7).

### 6.4 `/v1/models` (same phase, cheap)

Mark all members `hidden` (extend the existing flag with a preset-only key, or derive it from
`route-group` membership) and append one synthetic entry for the group in the listing loop
(`server-models.cpp:1995-2035`). Advertise 80,000 (the contract) so a client that inspects the
list sees that rather than a 54,264 ceiling. This is cosmetic for the
current clients — they already know the name — but any OpenAI-compatible library that
validates the model list would otherwise break on a name that answers requests yet is absent
from `/v1/models`.

## 7. What clients WILL notice (transparency is not free)

Worth agreeing on these before implementing, because they are the price of the contract:

- **TTFT spikes.** A small request that follows a big one pays the swap: ~8 s of first-token
  latency on a request that would otherwise answer in ~2 s. Same single model name, wildly
  variable TTFT. Unavoidable at `--models-max 1`; the mitigation is 6.3 plus, if measurement
  justifies it, a hysteresis timer keeping the upper tier resident for N seconds. A fast↔mid swap
  is the cheap one — same file, warm page cache.
- **Prompt cache loss on migration.** A swap discards the child's cached prefix. A 60k
  conversation crossing the threshold re-prefills from scratch, ~110 s. 6.3 limits this to twice
  per conversation; interleaved conversations of mixed sizes are the bad case. The fast→mid hop is
  the one that can be optimised away later, because those two share weights (section 11).
- **A style shift, but only on the rare overflow.** `qwen3-fast` and `qwen3-mid` are the same
  file, so crossing 50k changes nothing about the weights (MTP is distribution-preserving anyway —
  every draft token is verified against the target). Only a jump to `qwen3-long`, above ~72k,
  switches artifacts: same base model and quant class, but a separate file, so a subtle mid-thread
  style shift is possible there. That is the price of honouring the 80,000 in the client config;
  if it ever shows up in practice, the fix is to requantize a matching pair.
- **Throughput asymmetry stays visible.** ~47 decode tok/s under 50k, ~21 above it. Nothing to
  do about that — it is the point of the exercise.

## 8. Files touched

Staging step (section 5): none — one INI file outside the repo.

Routing:

| File | Change |
|---|---|
| `common/arg.cpp` | two `set_preset_only()` keys: `route-group`, `route-max-tokens` |
| `tools/server/server-models.h` | `route_groups` map + `resolve_route_target()` declaration |
| `tools/server/server-models.cpp` | build `route_groups` in `load_models()`; `resolve_route_target()` (estimator + no-demotion rule); call it in `proxy_post` (`:1917`) and `proxy_get`; hide members and emit the group entry in the `/v1/models` loop |
| `tools/server/README.md` | document the routing group, the estimator fallback and the TTFT/prompt-cache caveats |

Adding the third tier changes none of these rows: `route_groups` holds an ordered list of members,
so the ladder is INI-only.

Nothing in `common/speculative.cpp`, nothing in `server-context.cpp`, no change to
`can_speculate()`, no new inference-side flag.

## 9. Validation

1. **Staging, every tier named explicitly.** `qwen3-fast` with 50,168 prompt + 4,096 output →
   HTTP 200, ~47 decode tok/s. `qwen3-long` with 93,184 + 4,096 → HTTP 200, ~21 decode tok/s.
   Must match `kv-calibration-findings.md:414` and `:238` within noise.
2. **The `qwen3-mid` ceiling** (section 5, item 1). Fit-probe candidate with `ctx-size = 0`, then a
   maximum-size request at the pinned value → HTTP 200, no OOM. This number sets `qwen3-mid`'s
   `ctx-size` and cap; everything about the three-tier shape depends on it.
3. **Swap cost and VRAM reclaim.** Time the second request's queue-to-first-token delta and
   watch `nvidia-smi` across the eviction: VRAM must return to the idle baseline between
   children. Measure fast↔mid (same file, warm cache) separately from mid↔long. The measured load
   time replaces the assumed 8 s everywhere in this doc.
4. **A busy child is never evicted.** Start a long generation on the fast tier, then fire a
   mid-tier request: the first completes normally, the second queues and only then swaps. The
   whole design leans on this (`server-models.cpp:92`, `:203`, `:113`).
5. **Transparency, the actual acceptance test.** With only `qwen-3.8-27b` configured, send
   8k / 40k / 52k / 60k / 78k-token prompts each with `max_tokens=4096`. All must return HTTP
   200; log which child served each. Two boundaries to get right: the 52k case must land on
   `qwen3-mid` (52k + 4k > 50k cap), and the 78k case must land on `qwen3-long` (78k + 4k > 72k
   cap) — that second one is the whole reason the third tier exists.
6. **Estimator accuracy**, Portuguese and English, `/tokenize` truth vs bytes ÷ 3.5, at the same
   sizes. Record the worst-case undershoot; it must stay inside the tightest margin, which is
   `qwen3-mid`'s (ceiling − cap − 4,096).
7. **Omitted `max_tokens`.** A 52k prompt with no `max_tokens` must still route to `qwen3-mid`
   (implicit 4k reserve applied).
8. **No demotion.** A conversation growing 40k → 52k → 55k must migrate once to `qwen3-mid` and
   never come back; count the model loads (exactly one). Then push the same conversation past 72k
   and confirm exactly one more load, to `qwen3-long`.
9. **Thrash cost.** Alternate small/large 5× and record total wall clock, to decide whether a
   hysteresis timer is worth building.

## 10. Sequencing

1. Section 5 staging config — today, no code. Produces the `qwen3-mid` ceiling (the number the
   three-tier shape rests on) and the real load times.
2. Sections 6.1–6.3 — the routing itself. This is the deliverable the clients need.
3. Section 6.4 — `/v1/models` cosmetics, same phase if cheap.
4. Hysteresis only if test 9 says so.
5. Section 11 KV state transfer between `qwen3-fast` and `qwen3-mid`, only if test 9 says the
   migration cost hurts. It is legal (same weights) but it is real code, not config.

## 11. Carrying the KV state across the swap

The question this answers: can the RAM cache make the context load faster when the router
switches tiers? Not the RAM cache — but a file-backed path already in the tree can, for the
fast↔mid hop, because those two children run the same GGUF.

### 11.1 `--cache-ram` cannot cross a swap

`server_prompt_cache` (`server-task.h:597-635`, implementation `server-task.cpp:1689-1793`) is a
`std::list` **in the child process's heap**, instantiated at `server-context.cpp:1419` and sized
by `-cram`/`--cache-ram` (`arg.cpp:1726`). Sleeping is a full `destroy()`
(`server-context.cpp:964-1005`) and eviction kills the subprocess, so the cache dies with the
child. It accelerates turns **within** a tier and does nothing for the crossing.

### 11.2 What can cross: slot state in a file

`--slot-save-path` (`arg.cpp:3853`) plus `SERVER_TASK_TYPE_SLOT_SAVE`/`_RESTORE`
(`server-context.cpp:2688`/`:2738`, HTTP `action=save|restore` at `:4974-4977`) over
`llama_state_seq_save_file`/`load_file` (`src/llama-context.cpp:4348`/`:4359`) moves a slot's KV
between processes through a file. Sizing at 18.0 KiB/token: ~954 MiB for 54,264 tokens, ~1.3 GiB
at 72k. On **tmpfs** that is a few seconds of write plus read against **~110 s of prefill** for a
60k prompt. That, not the model load, is where the latency is.

Flow: `save` on the outgoing child before eviction → swap → `restore` on the incoming child → the
next request finds its prefix in the slot and prefills only the delta.

What `save` captures, precisely (`server-context.cpp:2688-2737`): the **live KV of one slot**, via
`slot->prompt.tokens.serialize()` (`:2710`) plus
`llama_state_seq_save_file(ctx_tgt, …, slot->id, …)` (`:2717`). It does **not** dump
`server_prompt_cache`. So:

- **One slot per call.** At `parallel = 1` that is the single resident conversation. Prefixes
  parked in `-cram` (idle slots pushed there by `--cache-idle-slots`) are not captured — such an
  entry only becomes real again by passing through a slot.
- **Target only.** `ctx_tgt`; the `ctx_dft` (nextn) state never enters the file. Harmless: the
  receiving child runs with MTP off and has no `ctx_dft`.
- **Ordering is free.** A save on a processing slot is *deferred*, not failed (`:2696-2701`), so a
  "save before evict" hook waits for the generation to finish on its own, which matches the
  never-evict-a-busy-child rule.

### 11.3 Where it is legal, and where it is not

**Identical weights are mandatory.** Restore validation is purely structural — KV layer count
(`src/llama-kv-cache.cpp:2535`), KV type (`:2563`), row size (`:2572`) — with **no weight hash**.
A state file from a different GGUF either fails those checks or, worse, is accepted and serves KV
computed by other weights, silently. So:

- **`qwen3-fast` → `qwen3-mid`: legal.** Same file, so same hparams and the same target KV
  geometry; MTP on/off only decides whether a separate `ctx_dft` exists, which is not in the file.
  This is the hop the three-tier shape was chosen to make possible.
- **anything ↔ `qwen3-long`: forbidden.** Different artifact. Never write a restore path that can
  reach it.
- **Only upward.** 54k of cells fit in a 76k cache, never the reverse — and the no-demotion rule
  (6.3) already forbids the other direction.

### 11.4 Why it is still step 5, not step 2

Not config-only: `unload_lru()` (`server-models.cpp:938-960`) has no "save before you die" hook,
the restore has to be issued after the load completes, and a tmpfs directory needs a size budget
and a reaper. The guard against the forbidden hop above is also code that has to be written and
reviewed.

Meanwhile the cheap version is already in the plan: **no-demotion plus a generous `-cram` on the
upper children**. A conversation migrates once, then lives in `qwen3-mid`, where the in-RAM cache
works normally for every later turn. That leaves exactly one expensive turn per conversation.
Build 11.2 when test 9 shows those turns add up.

Unrelated config win, cheap and independent: **do not pass `--ctx-size 0` to the children in
production.** With 0 the fitting probe builds and frees full target + MTP contexts on every load,
inflating swap time. Use it once during staging to *find* the numbers, then pin them (section 4).

Keep `docs/mtp-adaptive-context-plan.md` for its code-path analysis — the always-on
`common_speculative_process()` shadow decode and the five static-capability uses of
`can_speculate()` are correct findings worth having on record — but do not implement it.
