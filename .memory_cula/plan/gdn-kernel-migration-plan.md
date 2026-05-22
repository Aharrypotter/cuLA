# GDN Kernel Migration Plan

Status: 2026-05-22

Current cuLA branch: `gdn-porting`

Local upstream checkout: `3rdparty/flashinfer`

Local upstream commit: `41e5aa29a0e218bde2e2045316ed88e3f4dc8ca2`

Implementation progress:

- Phase 1 started and landed:
  - `cula.gdn.chunk_gated_delta_rule` public API boundary.
  - SM100-first input validation and allocation path.
  - no-checkpoint restriction.
  - PyTorch reference skeleton under `tests/gdn`.
  - Phase2 placeholder at the kernel launch boundary.
- Phase 1 test status:
  - sandbox CPU/reference subset: passed.
  - B200 outside sandbox: `tests/gdn/test_prefill_delta_rule.py` passed.
- Phase 2 started:
  - copied FlashInfer Blackwell GDN CuTe DSL kernel into `cula/ops/gdn_chunked_prefill.py`.
  - copied persistent tile scheduler into `cula/ops/gdn_tile_scheduler.py`.
  - wired `cula.gdn.blackwell_fused_fwd` to `cute.compile` with persistent SM100 scheduling.
  - fixed one cuLA DSL compatibility point: `TmemAllocator(storage.tmem_holding_buf)` instead of upstream `.ptr`.
- Phase 2 code polish:
  - added public availability helper `is_sm100_gdn_prefill_available`.
  - tightened checkpoint argument validation before the deferred checkpoint path.
  - tightened basic tensor shape checks for empty token/head dimensions.
  - made CUDA 13+ guard error include the current torch CUDA version.
  - documented that the kernel and scheduler are mechanical FlashInfer ports and should avoid algorithmic changes before CUDA 13+ validation.
  - added `benchmarks/bench_gdn_prefill.py` as the CUDA 13+ benchmark entrypoint with separate compile-time and steady-state latency reporting.
  - this code-polish pass intentionally did not run tests or benchmarks.
- Phase 2 current blocker:
  - current environment is CUDA 12.8 (`torch 2.9.1+cu128`, `/usr/local/cuda-12.8`).
  - FlashInfer upstream guards Blackwell GDN prefill behind CUDA 13+.
  - without that guard, current cuLA environment fails during CuTe DSL lowering on a TMEM copy legalization error.
  - cuLA wrapper now preserves the upstream CUDA 13+ guard; B200 runtime correctness requires a CUDA 13+ environment.
- Phase 2 test status in current CUDA 12.8 environment:
  - `tests/gdn/test_prefill_delta_rule.py`: passed in sandbox and outside sandbox.
  - SM100 runtime smoke currently validates the explicit CUDA 13+ rejection path, not numerical correctness.

## Decision

Port GDN into cuLA with an SM100 / Blackwell first strategy.

The first implementation target is FlashInfer PR #3001, not the older PR #2276
SM90 path. The local validation hardware is 8 * B200, so the fastest reliable
route is to migrate the Blackwell CuTe DSL chunked prefill forward kernel first,
validate it on B200, and only then decide whether a separate SM90 backend is
worth carrying.

Scope ordering:

1. SM100 GDN chunked prefill forward.
2. B200 correctness and benchmark.
3. Optional feature expansion: checkpoints, broader dtype/state variants.
4. Decode forward as a separate task.
5. SM90/Hopper backend as an optional later backend.
6. Backward only if a training use case becomes explicit.

This means prefill and decoding are not migrated in one batch, and backward is
not part of the first port.

## Upstream References

Primary migration source:

- PR #3001: `https://github.com/flashinfer-ai/flashinfer/pull/3001`
- Commit: `7c562d50 [feat] Add blackwell GDN prefill kernel (#3001)`
- Local source:
  - `3rdparty/flashinfer/flashinfer/gdn_kernels/blackwell/gdn_prefill.py`
  - `3rdparty/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`
  - `3rdparty/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py`

Important follow-up commits already present in local `3rdparty/flashinfer`:

| Commit | Meaning for cuLA port |
|---|---|
| `3516d2bc [fix] fix blackwell gdn accuracy issue (#3156)` | Must port before trusting correctness. |
| `5e1318cb fix(gdn): use physical SM count for SM100 persistent prefill kernel (#3155)` | Use physical SM count for persistent scheduling/workspace sizing. |
| `f7acd259 fix(gdn): address remaining CodeRabbit feedback from #3001 (#3165)` | Include cleanup after initial merge. |
| `9f4d1617 [chore] Add guard to blackwell GDN prefill (#3267)` | Preserve explicit availability guards. |
| `f5e533ce Ameyn/gdn bf16 dispatcher and 4d pool (#3268)` | Audit for dtype/state-pool implications before enabling more modes. |
| `bc4dc972 fix deprecation warnings from cute-dsl (#3333)` | Useful if cuLA's Cutlass DSL version matches newer APIs. |

Secondary reference:

- PR #2276: `https://github.com/flashinfer-ai/flashinfer/pull/2276`
- Commit view used earlier:
  `https://github.com/flashinfer-ai/flashinfer/pull/2276/changes/16e5331322a6f45b3433b5b41fc5aaf2c4c87c5c`

PR #2276 introduced the GDN SM90 C++/CUTLASS path. Keep it as algorithm and
test context, but do not make it the first migration target for this B200
environment.

## cuLA Context

cuLA already has Python CuTe DSL Blackwell kernels. The difference between
FlashInfer SM100 GDN and cuLA SM100 KDA/Lightning is therefore not the choice of
CuTe DSL itself. The real integration differences are:

- FlashInfer imports and packaging versus cuLA imports and package boundaries.
- FlashInfer wrapper/cache conventions versus cuLA wrapper/cache conventions.
- FlashInfer public `flashinfer.gdn_prefill.chunk_gated_delta_rule` API versus a
  new cuLA `cula.gdn` API.
- FlashInfer's `get_num_sm`, availability guards, and compile options versus
  cuLA's `cula.utils` and existing Blackwell compile style.
- Different test/benchmark organization.

Existing cuLA files to use as style references:

| cuLA file | Use |
|---|---|
| `cula/kda/blackwell_fused_fwd.py` | Python CuTe DSL wrapper, cache, stream and workspace style. |
| `cula/ops/kda_fully_fused_wip.py` | Blackwell CuTe DSL kernel structure. |
| `cula/ops/lightning_attn.py` | Additional CuTe DSL integration reference. |
| `cula/utils.py` | Blackwell guards and local utility patterns. |

## First Deliverable

Expose a cuLA-native forward prefill API:

```python
cula.gdn.chunk_gated_delta_rule(...)
```

Minimum supported mode:

| Dimension | First target |
|---|---|
| GPU | SM100 / B200 |
| Kernel | chunked prefill forward |
| Implementation | Python CuTe DSL adapted from FlashInfer PR #3001 + follow-up fixes |
| dtype | bf16 first; fp16 after bf16 passes |
| head size | 128 |
| layout | packed varlen `[total_tokens, heads, 128]` |
| grouping | GQA and GVA |
| gate inputs | `g` and `beta` supported; wrapper may materialize ones for `None` |
| state | optional `initial_state`, optional `output_state` |
| checkpoints | disabled in first port |
| decode | disabled in first port |
| backward | disabled in first port |

Do not merge this into `cula.kda`. GDN and KDA share delta-rule structure, but
their public gate semantics and expected model usage differ enough that a
separate `cula.gdn` namespace is the safer API boundary.

## Non-Goals

- Do not start with the SM90 C++ path while B200 is the only validation target.
- Do not migrate FlashInfer's TVM FFI or custom-op registration stack.
- Do not implement decode in the same change as prefill.
- Do not implement backward without a concrete training requirement.
- Do not enable checkpointing until the no-checkpoint path is correct.
- Do not silently route unsupported GDN shapes through KDA.
- Do not optimize beyond obvious cache/workspace fixes before reference tests
  pass.

## Algorithm Contract

The operator implements the gated delta rule recurrence:

```text
S_t = g_t * S_{t-1} + beta_t * update(k_t, v_t, S_{t-1})
O_t = q_t @ S_t
```

The chunked kernel computes block-local `QK`, `KK`, transformed values,
intra-chunk output, inter-chunk output from carried state, and the state update.
The Blackwell implementation keeps the recurrent state in TMEM during a tile's
chunk loop and uses a persistent scheduler over `(sequence, output_head)` tiles.

Important behavior to preserve:

- `scale is None` or `scale == 0.0` maps to `1 / sqrt(head_size)` in the public
  wrapper.
- `g is None` behaves as all ones.
- `beta is None` behaves as all ones.
- SM100 kernel input `cu_seqlens` is int32 in the FlashInfer adapter; public API
  can accept int64 and convert before launch.
- Final state layout in the Blackwell path is `[N, H, V, K]`.
- The scheduler must use the physical SM count, matching FlashInfer fix #3155.

## Tensor Contract

Initial cuLA API should follow FlashInfer packed varlen semantics:

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `[total_tokens, num_q_heads, head_size]` | bf16 first | contiguous CUDA |
| `k` | `[total_tokens, num_k_heads, head_size]` | bf16 first | contiguous CUDA |
| `v` | `[total_tokens, num_v_heads, head_size]` | bf16 first | contiguous CUDA |
| `g` | `[total_tokens, num_o_heads]` | fp32 | optional, materialize ones if absent |
| `beta` | `[total_tokens, num_o_heads]` | fp32 | optional, materialize ones if absent |
| `cu_seqlens` | `[num_seqs + 1]` | int32 or int64 public; int32 kernel | required |
| `initial_state` | `[num_seqs, num_o_heads, head_size, head_size]` | fp32 first | optional, `[N,H,V,K]` layout |
| `output` | `[total_tokens, num_o_heads, head_size]` | same as `q` | optional preallocated |
| `output_state` | `[num_seqs, num_o_heads, head_size, head_size]` | fp32 first | optional, `[N,H,V,K]` layout |

Definitions:

```text
num_o_heads = max(num_q_heads, num_v_heads)
head_size = 128 in first port
```

Grouping guards:

```text
GQA: num_q_heads >= num_v_heads
     num_k_heads == num_v_heads
     num_q_heads % num_k_heads == 0

GVA: num_v_heads > num_q_heads
     num_k_heads == num_q_heads
     num_v_heads % num_q_heads == 0
```

Qwen3-Next smoke target:

```text
num_q_heads = 16
num_k_heads = 16
num_v_heads = 32
head_size = 128
```

## Proposed File Layout

Python/CuTe DSL first; no C++ extension wiring should be needed for the first
SM100 pass.

```text
cula/gdn/__init__.py
cula/gdn/blackwell_fused_fwd.py
cula/ops/gdn_chunked_prefill.py
tests/gdn/reference_delta_rule.py
tests/gdn/test_prefill_delta_rule.py
benchmarks/bench_gdn_prefill.py
```

Likely source mapping:

| FlashInfer source | cuLA destination |
|---|---|
| `flashinfer/gdn_prefill.py` | Public API behavior and validation copied into `cula/gdn/blackwell_fused_fwd.py`. |
| `gdn_kernels/blackwell/gdn_prefill.py` | Compile/cache/adapter logic copied into `cula/gdn/blackwell_fused_fwd.py`. |
| `gdn_kernels/blackwell/gated_delta_net_chunked.py` | Kernel body copied into `cula/ops/gdn_chunked_prefill.py`. |
| `gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py` | Either keep as `cula/ops/gdn_tile_scheduler.py` or inline near kernel if local style prefers. |
| `tests/gdn/reference_delta_rule.py` | `tests/gdn/reference_delta_rule.py`. |
| `tests/gdn/test_prefill_delta_rule.py` | `tests/gdn/test_prefill_delta_rule.py`, reduced to SM100-first matrix. |
| `benchmarks/bench_gdn_prefill.py` | `benchmarks/bench_gdn_prefill.py`, adapted to cuLA import path. |

## Migration Phases

### Phase 0: Environment and Source Lock

Goal: make sure the local implementation source is fixed before code is copied.

Tasks:

- Confirm local FlashInfer commit and clean status.
- Record PR #3001 commit plus follow-up fix commits in this document.
- Confirm current cuLA Cutlass DSL/CUDA environment can compile existing
  Blackwell KDA kernels.
- Check whether CUDA 13+ is available. FlashInfer guards Blackwell GDN prefill
  behind CUDA 13+.
- Confirm whether cuLA should preserve FlashInfer's `--enable-tvm-ffi
  --opt-level 2` compile option or use cuLA's current `cute.compile` option
  style.

Exit gate:

- Existing cuLA Blackwell smoke still imports/runs.
- A minimal API signature is accepted.

### Phase 1: API Wrapper and Test Skeleton

Goal: build the cuLA-facing surface and reference tests before tuning kernel
details.

Tasks:

- Add `cula/gdn/__init__.py`.
- Add `cula/gdn/blackwell_fused_fwd.py` with shape/dtype/device guards.
- Start from no-checkpoint path.
- Convert public `cu_seqlens` to int32 for the SM100 kernel.
- Allocate `output` and optionally `output_state`.
- Materialize default all-one `g` / `beta` only when inputs are `None`.
- Port the PyTorch reference and reduce the first test matrix to B200-friendly
  smoke cases.

Exit gate:

- API rejects unsupported configs clearly.
- Reference test imports and can generate expected output.

### Phase 2: Mechanical SM100 CuTe DSL Port

Goal: compile the FlashInfer Blackwell chunked prefill kernel inside cuLA with
minimal semantic changes.

Tasks:

- Copy/adapt `GatedDeltaNetChunkedKernel`.
- Copy/adapt `GDNTileSchedulerParams` and `GDNTileScheduler`.
- Replace FlashInfer imports:
  - `flashinfer.cute_dsl.utils.get_num_sm` -> cuLA local helper or
    `torch.cuda`/CUDA runtime based helper.
  - `flashinfer.gdn_kernels...` package paths -> `cula.ops...`.
- Align stream handling with cuLA Blackwell kernels.
- Align compile cache keys with static kernel properties:
  - IO dtype;
  - state dtype;
  - `HQ`, `HV`;
  - GQA/GVA flag;
  - initial state flag;
  - final state flag;
  - checkpoint flag, fixed false at first.
- Preserve dynamic token dimension marking.
- Preserve physical SM count usage.
- Cache workspace per device and grow when needed.

Exit gate:

- `python -m py_compile` passes for new files.
- First kernel compile completes on B200 for a tiny smoke case.

### Phase 3: Correctness on B200

Goal: prove semantic parity with the PyTorch reference before benchmarking.

Minimum correctness matrix:

| Dimension | Values |
|---|---|
| dtype | bf16 first |
| head size | 128 |
| grouping | `(1,1,1)`, `(4,1,1)`, `(1,1,2)`, Qwen3-Next `(16,16,32)` smoke |
| sequence lengths | `[64]`, `[128]`, `[65]`, `[127]`, `[64, 128]` |
| gates | both `g` and `beta` present; then `None` default cases |
| state | no initial state; then two-step continuation with final state |
| checkpoints | disabled |

Exit gate:

- B200 correctness passes for the matrix above.
- State layout comparison explicitly accounts for `[N,H,V,K]`.
- Existing KDA/Lightning imports are unaffected.

### Phase 4: Benchmark

Goal: measure the port under realistic B200 settings after correctness passes.

Benchmark inputs:

```bash
python benchmarks/bench_gdn_prefill.py \
  --preset qwen3-next \
  --batch-size 1 4 16 64 \
  --seq-len 128 256 512 1024 \
  --dtype bfloat16
```

Metrics:

- median latency;
- compile time separated from steady-state timing;
- tokens/s;
- approximate TFLOPS if the benchmark keeps the upstream formula;
- shape, dtype, device, SM count, and Cutlass DSL version.

Exit gate:

- Benchmark runs without counting first-call compilation in steady-state timing.
- Qwen3-Next preset is reported separately from custom sweeps.

### Phase 5: Feature Expansion

Only after the core SM100 prefill path is correct:

- Enable fp16 if needed.
- Enable bf16 state path if useful for memory pressure.
- Add checkpoint outputs.
- Add GDN decode forward by auditing FlashInfer decode kernels separately.
- Add SM90/Hopper C++ backend from PR #2276 if H100/H200 validation becomes
  necessary.
- Add backward only with a concrete training requirement and test oracle.

## Validation Commands

Syntax check:

```bash
python -m py_compile \
  cula/gdn/blackwell_fused_fwd.py \
  cula/ops/gdn_chunked_prefill.py \
  tests/gdn/test_prefill_delta_rule.py \
  benchmarks/bench_gdn_prefill.py
```

Import smoke:

```bash
python -c "import cula; import cula.gdn; print('ok')"
```

Minimal B200 correctness:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/gdn/test_prefill_delta_rule.py -q -k "smoke"
```

Full GDN prefill correctness:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/gdn/test_prefill_delta_rule.py -q
```

Benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_prefill.py --preset qwen3-next --dtype bfloat16
```

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA/Cutlass DSL version mismatch | Kernel import or compile fails | Start by compiling existing cuLA Blackwell kernels and include FlashInfer #3333 compatibility edits. |
| First-call compile is mistaken for runtime latency | Misleading benchmark | Separate compile warmup from measurement. |
| Physical SM count is not used | Persistent scheduler under/over-subscribes B200 | Port FlashInfer #3155 behavior explicitly. |
| Accuracy fix from #3156 is missed | Reference tests fail or pass only on easy shapes | Port from current local `main`, not only #3001 merge commit. |
| State layout is misunderstood | Continuation or decode integration breaks | Document `[N,H,V,K]` and encode it in tests. |
| `cu_seqlens` dtype differs between public API and kernel | Runtime errors or wrong indexing | Accept int64/int32 publicly, convert to int32 at SM100 adapter boundary. |
| Optional checkpoint support complicates first port | Larger compile/test surface | Hard-disable checkpointing until no-checkpoint path passes. |
| GQA/GVA head mapping is wrong | Correctness fails for Qwen3-Next | Include both grouped modes in the first correctness matrix. |
| Materializing default gates adds overhead | Benchmark looks worse for `None` gates | Accept for correctness first; optimize default-gate path later only if needed. |
| API gets mixed with KDA | Semantics become unclear | Keep `cula.gdn` as a separate namespace. |

## Acceptance Criteria

Minimum PR-ready state:

- `cula.gdn.chunk_gated_delta_rule` exists.
- SM100/B200 prefill forward compiles through cuLA.
- bf16 `head_size=128` correctness passes for:
  - one full chunk;
  - one tail chunk;
  - multi-sequence varlen;
  - GQA;
  - GVA;
  - optional final state continuation.
- Unsupported configs fail with explicit messages.
- Qwen3-Next benchmark command runs.
- Existing cuLA KDA/Lightning imports are not regressed.

Stretch goals:

- fp16 parity with upstream.
- checkpoint state outputs.
- bf16 state path.
- decode forward.
- SM90 backend.
