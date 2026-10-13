# Staging worker ordering and cancellation probes

Test-only follow-up to [KIOSHIROI/sglang#1](https://github.com/KIOSHIROI/sglang/pull/1).
Pinned head: `6cf15d15f2f9705923cf040adf8e518475286d65`.
Parent/base: `72a2d68687b755f5fd4c9430a90f95ca1462cca3`.

## Run

Only Python 3.10+ and NumPy are required. From the repository root:

```bash
python test/manual/disaggregation/test_staging_worker_ordering.py -v
```

Expected at the pinned head: **3 pass, 3 fail**, exit code 1. These are desired
safety-contract assertions, deliberately left red rather than marked xfail.
The file is a manual test, not registered for CI; this change fixes no production
behavior. The three passing controls can be run separately:

```bash
python test/manual/disaggregation/test_staging_worker_ordering.py -v \
  TestStagingWorkerOrdering.test_ready_chunks_finish_in_order \
  TestStagingWorkerOrdering.test_transport_failure_stops_later_chunk \
  TestStagingWorkerOrdering.test_abort_before_dequeue_skips_transfer
```

## Observations

| Case | Head | Parent |
|---|---|---|
| Ready chunks complete in order | Pass | Pass |
| First transport error stops later chunk | Pass | Pass |
| Abort before dequeue skips transfer | Pass | Pass |
| Deferred full-grid chunk completes before Success | Fail | Fail |
| Deferred partial chunk completes before final continuation's Success | Fail | Fail |
| Abort during readiness wait prevents new submission | Fail | Fail |

**Completion before the earlier transfer:** once the 0.5-second readiness spin
expires, `_do_staging_transfer` puts the early chunk at the queue tail. The
allocation becomes ready immediately afterwards. A queued final chunk runs
before the deferred chunk and publishes Success while both are still counted
outstanding. The recorded full-grid sequence is:

```text
defer chunk 0
transfer chunk 8
CHUNK_READY chunk 8
Success (outstanding = 2)
transfer chunk 0
CHUNK_READY chunk 0
```

The partial case uses pages `[0:7]` followed by the final `[7:8]` continuation
within an 8-page staging grid. On the head, the final continuation goes direct
and publishes Success before the earlier staged piece is transferred. The
parent also fails the worker-level ordering contract when fed these chunks;
its scheduler may not emit the same partial pieces. The full-grid case already
fails on the parent without requiring the new partial-send behavior.

**Cancellation during the wait:** the simulated control callback calls the real
`update_status(room, Failed)` while the worker is waiting. Readiness then becomes
true. The same worker resumes, submits a staged transfer, and emits CHUNK_READY:

```text
abort room 7
transfer chunk 0
CHUNK_READY chunk 0
```

The queued-before-abort control is skipped correctly, so the gap is specifically
between the worker's initial status check and submission after readiness wait.

These are inherited worker gaps, not evidence that this follow-up introduced a
regression. Candidate fixes should preserve same-room ordering across deferral
(or explicitly wait for all required chunks before publishing Success), and
coordinate cancellation with admission of new transfers after waiting. A lone
status check does not by itself prove race-free admission or safe buffer release.

## Scope and method

The harness compiles the complete production `transfer_worker`,
`_do_staging_transfer`, fragmentation/piece predicates when present, and common
status methods from this checkout. It preserves their control flow, substituting
only the allocator import with its `ALLOC_OVERSIZED` constant dependency. It
uses a deterministic FIFO, clock and condition callback; ready/transfer/control
I/O are fakes. TP=PP=1, staging grid 8 pages, fragmented destination indices, and
minimum fragmentation groups 2 form a small synthetic workload. The real
0.5-second spin budget advances on the fake clock, with no wall-clock sleeps.

There are no GPU allocations, byte transfers, RDMA, scatter, model execution or
performance measurements. The observations concern sender bookkeeping and call
ordering; they do not demonstrate data corruption, receiver behavior or actual
buffer reuse after cancellation.

The prior direct-transfer reproductions for #43995 cannot run unchanged here:
this branch has neither `_send_request_batch` nor its unit-test fixture. These
probes adapt the failure/completion questions to staging instead of treating
missing batching APIs as test failures.

## Parent comparison

```bash
git worktree add --detach ../sglang-staging-parent 72a2d68687b755f5fd4c9430a90f95ca1462cca3
SGLANG_STAGING_SOURCE_ROOT="$PWD/../sglang-staging-parent" \
  python test/manual/disaggregation/test_staging_worker_ordering.py -v
```

The same file executes the parent's method bodies. Expect the same 3 pass / 3
fail count; no parent source modifications are required.

In the review environment, normal imports of this older branch were blocked by
an installed Transformers configuration-name collision (`qwen3_asr`). The
author's existing 16 CPU cases passed in an isolated runner replacing only
`CustomTestCase` with `unittest.TestCase` and disabling CI registration. This is
not a claim that the full package test suite or GPU integration passed locally.
