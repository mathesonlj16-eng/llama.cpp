# Qwen3.5 draft-simple rollback investigation

**Status:** research harness only; no rollback patch, benchmark rerun, or upstream submission.

## Objective

Understand why Qwen3.5-2B with `draft-simple` on a GTX 1650 regresses from 62.22 tokens/s (plain decode) to 28.19 tokens/s (speculative), and whether native recurrent-state rollback can safely replace expensive host-backed checkpoints. These numbers were recorded in a single-prompt experiment and must be reconfirmed over multiple prompts/sessions.

- Original upstream issue: https://github.com/ggml-org/llama.cpp/issues/32776
- Our independent profiling comment: https://github.com/ggml-org/llama.cpp/issues/32776#issuecomment-7084326461
- Prior on-device-checkpoint attempt: https://github.com/ggml-org/llama.cpp/pull/28118
- Prior native-rollback attempt: https://github.com/ggml-org/llama.cpp/pull/28120
- Research tracker and acceptance criteria: https://github.com/mathesonlj16-eng/gpu-performance-lab/issues/1

## Source-level findings to verify on each pinned commit

1. `common/common.h`: `common_params_speculative::need_n_rs_seq()` requests rollback capacity for MTP/EAGLE3/DFlash/DSpark/Lilicorr; `draft-simple` is not listed.
2. `src/llama-arch.cpp`: `llm_arch_supports_rs_rollback()` includes Qwen3.5.
3. `common/common.cpp`: `common_context_can_seq_rm()` reports RS only when `llama_n_rs_seq(ctx) > 0`, otherwise probes partial deletion, which can result in FULL.
4. `tools/server/server-context.cpp`: target and draft checkpoints are maintained separately, and rejection can trigger restore and target replay.
5. A prior experiment enabling target bounded rollback alone **aborted**. Determine the precise failing sequence-removal operation and both contexts' actual rollback capacities before modifying logic.

A single `n_rs_seq` change is **not** a validated fix. Do not make unchecked changes to the live sequence-state path.

## Benchmark procedure (localhost only)

Use the same model files, hashes, GPU layer split, context length, CPU threads, CUDA setup, and prompt list for each test. Models are **not** downloaded by the benchmark.

The script is a stdlib-only client for an **already-running** `llama-server` on 127.0.0.1. It does not start/stop servers or change application code. Run its offline smoke tests first:\n\n```bash\npython3 research/qwen35-speculative/test_bench_client.py\n```\n\nReplace `ACTUAL_LLAMA_CPP_COMMIT` with `git rev-parse HEAD` from **the source checkout that built the currently running server**. Do not label a newer executable as the historical b12602 / `418fe97b5` build. For the historical reproduction, build that commit in a separate checkout and invoke this branch's benchmark client by its absolute path.

Start the **vanilla** server in one WSL terminal (substitute existing GGUF paths and the *tested* target layer count):

```bash
./build/bin/llama-server \
  -m /absolute/path/Qwen3.5-2B-Q8_0.gguf \
  -ngl YOUR_TESTED_TARGET_LAYER_COUNT \
  --host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1
```

In a second WSL terminal, while the server is ready:

```bash
python3 research/qwen35-speculative/bench_client.py \
  --variant vanilla --build ACTUAL_LLAMA_CPP_COMMIT --session 01 \
  --notes "target Q8_0 sha256=...; -ngl=...; ctx=4096; np=1; CUDA=..." \
  --url http://127.0.0.1:8080 --rounds 5 --warmups 1 --output-dir /tmp/qwen35-study
```

Stop that server, then run the **speculative** variant with the *same target parameters*, adding the draft model and validated draft settings:

```bash
./build/bin/llama-server \
  -m /absolute/path/Qwen3.5-2B-Q8_0.gguf \
  -ngl YOUR_TESTED_TARGET_LAYER_COUNT \
  --host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1 \
  --spec-type draft-simple -md /absolute/path/Qwen3.5-0.8B-Q4_K_M.gguf \
  --spec-draft-n-max 3 -td 4 -ngld YOUR_TESTED_DRAFT_LAYER_COUNT
```

```bash
python3 research/qwen35-speculative/bench_client.py \
  --variant spec-default --build ACTUAL_LLAMA_CPP_COMMIT --session 01 \
  --notes "target/draft sha256=...; -ngl=...; -ngld=...; -td=4; CUDA=..." \
  --url http://127.0.0.1:8080 --rounds 5 --warmups 1 --output-dir /tmp/qwen35-study
```

Alternate sessions (vanilla→spec→spec→vanilla) rather than running a single long unpaired comparison. Repeat with current upstream only after preserving the pinned baseline. Keep raw `.jsonl` results outside the repo unless you intentionally review and commit a redacted summary.

The client records server-reported generation tokens/sec separately from end-to-end request latency. It records SHA-256 of generated text but does **not** infer model correctness from a single hash match or mismatch. To measure CUDA checkpoint and replay costs, collect separate Nsight Systems profiles.

**Stop rules:** stop for process abort, out-of-memory, unexpected nonlocal connection, invalid model state, or output corruption. Never interpret overlapping CUDA timings as additive time saved.

## Next engineering gate

Before testing an experimental implementation, map sequence IDs, `n_rs_seq` for both contexts, rollback-window limits, and the rejected-draft replay path. Test n-max 1/2/3, all-accepted and early/mid/late rejection, consecutive requests, long contexts, and multi-slot configurations if supported. An experimental rollback branch must preserve default behavior behind a flag and include a safe fallback or clean error rather than aborting.

**No upstream PR or correctness claims until the above gate passes.**


## Diagnostic patch: locate the sequence-removal abort

The research branch now has **log-only / error-message-only** instrumentation in three existing source files. It does **not** enable rollback, change checkpoint placement, alter execution policies, or suppress aborts.

| Source | Diagnostic |
|---|---|
| `tools/server/server-context.cpp` | On speculative initialization, reports target and draft `n_rs_seq` and resolved sequence-removal types; on post-draft removal failure identifies the draft path. |
| `common/common.cpp` | Identifies whether an abort came from the **target** or **draft** during combined sequence removal; prints requested range and that context's `n_rs_seq`. |
| `src/llama-memory-recurrent.cpp` | Explains why recurrent `seq_rm()` returned false: invalid sequence index, shared state, rollback beyond stored window, rollback already pending, or invalid negative-sequence range. |

**Run this branch's unchanged speculative configuration first**, without manually adding `n_rs_seq` or any checkpoint flag. Record the resolved configuration. The prior abort occurred in a manually altered setup, so it is **not yet reproduced** on this branch.

In the isolated WSL2 llama.cpp checkout:

```bash
python3 research/qwen35-speculative/test_bench_client.py

cmake -S . -B build-qwen35-research -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-qwen35-research --target llama-server --parallel 2
git rev-parse HEAD
```

Start `build-qwen35-research/bin/llama-server` using your **existing validated** target/draft files and settings (see above). Capture stderr/stdout to a local log with `2>&1 | tee /tmp/qwen35-research-server.log` and preserve the exact command and model hashes.

If and only if the unchanged configuration is correct and stable, reproduce the prior **target-only experimental change** on a separate disposable worktree with the smallest possible diff. Stop as soon as it fails, and keep the last log lines and exact commit. Do not run repeated crash attempts or change both contexts blindly.

### Interpreting failures

- `bounded rollback rejected`: inspect `requested`, `rs_max`, and `rs_idx`; this is an **invariant/capacity-of-snapshots** failure, not by itself a 4 GiB hardware-capacity failure.
- `cannot partially roll back shared state`: inspect sequence sharing and slot count.
- `failed to remove sequence in target context`: target memory rejected the combined removal.
- `failed to remove sequence in draft context` or `failed to remove draft sequence`: draft memory rejected a removal, potentially even if target rollback is enabled.
- `cannot save/load multiple ranges of cells to/from device memory`: separate limitation of on-device checkpointing; don't assume extra GPU RAM would fix it.
- CUDA allocation errors / OOM: **hardware memory capacity** may be relevant; capture VRAM usage and allocations before concluding.

The specific failure reason remains **unconfirmed until a real aborted run produces logs**. The research branch intentionally retains the failing assertions so incorrect sequence state is never silently accepted.

### Revision caveat

`main` / `master` evolves quickly. Compare the exact checked-out fork revision to current upstream before proposing a fix: some recurrent-state internals have changed. A result from a historical commit is not evidence that current upstream has the same failure.
