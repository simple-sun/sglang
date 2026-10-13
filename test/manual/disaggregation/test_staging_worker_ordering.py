"""Adversarial CPU checks of the staging worker; no GPU/RDMA is initialized.

Run directly with Python. Known failures are intentionally not marked xfail.
SGLANG_STAGING_SOURCE_ROOT may point to another checkout for a parent comparison.
"""

import ast
import os
import threading
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

ROOT = Path(
    os.environ.get("SGLANG_STAGING_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)
SOURCE = ROOT / "python/sglang/srt/disaggregation"


def load_methods(relative, class_name, names, namespace):
    """Compile whole production methods, replacing only their import dependencies."""
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]

    class AllocatorImport(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "sglang.srt.disaggregation.common.staging_buffer":
                assert [alias.name for alias in node.names] == ["StagingAllocator"]
                return ast.copy_location(ast.Pass(), node)
            return node

    for node in nodes:
        AllocatorImport().visit(node)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=future + nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return {node.name: namespace[node.name] for node in nodes}


class StopWorker(BaseException):
    """Exit the real worker after draining our finite queue, without a thread."""


class WorkerScenario:
    def __init__(self, *, delay=False, abort=False, fail_transfer=False):
        self.events = []
        self.now = 0.0
        self.ready = not (delay or abort)
        self.abort = abort
        self.fail_transfer = fail_transfer
        self.poll = NS(Failed=0, WaitingForInput=2, Success=4)
        self.pending = deque()
        self.gets = 0
        scenario = self

        class Waiter:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def wait(self, timeout):
                scenario.now += timeout
                if scenario.abort:
                    scenario.manager.update_status(7, scenario.poll.Failed)
                    scenario.events.append(("abort", 7))
                    scenario.ready = True

        namespace = dict(
            np=np,
            time=NS(monotonic=lambda: self.now, perf_counter=lambda: self.now),
            logger=Mock(),
            KVPoll=self.poll,
            StagingAllocator=NS(ALLOC_OVERSIZED=-2),
            _STAGING_REQUEST_FAILED=object(),
            STAGING_READY_SPIN_S=0.5,
            STAGING_MAX_WAIT_S=60,
            NetworkAddress=lambda *args: NS(to_host_port_str=lambda: "test:1"),
            envs=NS(
                SGLANG_DISAGG_STAGING_MIN_GROUPS=NS(get=lambda: 2),
                SGLANG_DISAGG_STAGING_ALWAYS=NS(get=lambda: False),
            ),
        )
        methods = load_methods(
            "mooncake/conn.py",
            "MooncakeKVManager",
            {
                "transfer_worker",
                "_do_staging_transfer",
                "_is_first_staging_piece",
                "_fragmented_groups",
            },
            namespace,
        )
        status = load_methods(
            "common/conn.py",
            "CommonKVManager",
            {"check_status", "update_status"},
            namespace,
        )
        self.manager = NS(
            enable_trace=False,
            enable_staging=True,
            attn_tp_size=1,
            attn_cp_size=1,
            pp_size=1,
            attn_tp_rank=0,
            attn_cp_rank=0,
            pp_rank=0,
            is_mla_backend=False,
            is_hybrid_mla_backend=False,
            session_lock=threading.Lock(),
            failed_sessions=set(),
            session_failures=defaultdict(int),
            request_status={7: self.poll.WaitingForInput},
            _staging_outstanding=defaultdict(int),
            req_to_decode_prefix_len={},
            _staging_ctx=NS(
                watermark_cv=Waiter(), prefetch_requested=set(), prefetched_rooms=set()
            ),
            kv_args=NS(kv_data_ptrs=[1]),
            bootstrap_port=0,
            _get_dsa_cache_transfer_skip_flags=lambda peer: (False, False),
            _ensure_equal_tp_staging_req=lambda *args: True,
            record_failure=lambda *args: self.events.append(("failure", args[0])),
            send_aux=lambda *args: 0,
        )
        for name, method in {**methods, **status}.items():
            setattr(
                self.manager,
                name,
                method
                if isinstance(method, staticmethod)
                else MethodType(method, self.manager),
            )
        self.manager._send_chunk_ready = lambda req, idx, chunk, rank: (
            self.events.append(("chunk_ready", chunk.index_slice.start))
        )
        self.manager.sync_status_to_decode_endpoint = (
            lambda endpoint, port, room, status, rank: self.events.append(
                ("status", status, self.manager._staging_outstanding[room])
            )
        )
        self.manager.send_kvcache = lambda session, pages, *args, **kwargs: (
            self.transfer("direct", pages)
        )
        strategy = NS(
            full_chunk_pages=8,
            check_ready=lambda req, start, size: (
                self.ready,
                start // 8,
                0 if self.ready else -1,
                0,
                0,
            ),
            transfer=lambda session, pages, *args: self.transfer("staged", pages),
        )
        self.manager._try_create_staging_strategy = lambda buffer: strategy
        self.manager.decode_kv_args_table = {
            "s": NS(
                requires_dcp_relayout=False,
                dst_attn_tp_size=1,
                staging_base_ptr=1024,
                staging_total_size=4096,
                dst_kv_ptrs=[1],
                dst_kv_layer_ids=[],
                dst_kv_item_len=4,
                dst_aux_ptrs=[],
            )
        }
        self.manager.transfer_infos = {
            7: {
                "s": NS(
                    room=7,
                    mooncake_session_id="s",
                    is_dummy=False,
                    dst_device_kv_indices=None,
                    dst_kv_indices=np.arange(0, 64, 2),
                    endpoint="test",
                    dst_port=1,
                    required_dst_info_num=1,
                )
            }
        }

    def transfer(self, path, pages):
        self.events.append((path, int(pages[0])))
        return -1 if self.fail_transfer else 0

    def chunk(self, start, end, last=False):
        return NS(
            room=7,
            index_slice=slice(start, end),
            prefill_kv_indices=np.arange(start, end),
            is_last_chunk=last,
            prefill_aux_index=None,
            state_indices=None,
            staging_counted=False,
            staging_first_attempt=None,
        )

    def get(self):
        self.gets += 1
        if self.gets > 8:
            raise AssertionError("worker did not drain the finite queue")
        if not self.pending:
            raise StopWorker()
        return self.pending.popleft()

    def put(self, chunk):
        self.events.append(("defer", chunk.index_slice.start))
        self.pending.append(chunk)
        # Allocation becomes available just after the bounded spin expires.
        self.ready = True

    def run(self, chunks):
        self.pending.extend(chunks)
        try:
            self.manager.transfer_worker(self, None, staging_buffer=object())
        except StopWorker:
            pass
        return self.events


class TestStagingWorkerOrdering(unittest.TestCase):
    def test_ready_chunks_finish_in_order(self):
        scenario = WorkerScenario()
        events = scenario.run([scenario.chunk(0, 8), scenario.chunk(8, 16, last=True)])
        self.assertEqual(
            [e for e in events if e[0] == "staged"], [("staged", 0), ("staged", 8)]
        )
        self.assertEqual(
            [e[1] for e in events if e[0] == "status"], [scenario.poll.Success]
        )

    def test_transport_failure_stops_later_chunk(self):
        scenario = WorkerScenario(fail_transfer=True)
        events = scenario.run([scenario.chunk(0, 8), scenario.chunk(8, 16, last=True)])
        self.assertEqual([e for e in events if e[0] == "staged"], [("staged", 0)])
        self.assertEqual(scenario.manager.failed_sessions, {"s"})
        self.assertEqual(scenario.manager.request_status[7], scenario.poll.Failed)

    def test_deferred_chunk_completes_before_success(self):
        scenario = WorkerScenario(delay=True)
        events = scenario.run([scenario.chunk(0, 8), scenario.chunk(8, 16, last=True)])
        success = next(
            i
            for i, e in enumerate(events)
            if e[:2] == ("status", scenario.poll.Success)
        )
        self.assertIn(
            ("staged", 0),
            events[:success],
            msg=f"SUCCESS preceded the deferred transfer: {events}",
        )

    def test_deferred_partial_chunk_precedes_final_direct_continuation(self):
        scenario = WorkerScenario(delay=True)
        events = scenario.run([scenario.chunk(0, 7), scenario.chunk(7, 8, last=True)])
        success = next(
            i
            for i, e in enumerate(events)
            if e[:2] == ("status", scenario.poll.Success)
        )
        self.assertIn(
            ("staged", 0),
            events[:success],
            msg=f"Partial chunk still pending at SUCCESS: {events}",
        )

    def test_abort_before_dequeue_skips_transfer(self):
        scenario = WorkerScenario()
        scenario.manager.update_status(7, scenario.poll.Failed)
        events = scenario.run([scenario.chunk(0, 8)])
        self.assertEqual(events, [])

    def test_abort_during_wait_prevents_new_transfer(self):
        scenario = WorkerScenario(abort=True)
        events = scenario.run([scenario.chunk(0, 8)])
        self.assertEqual(
            [e for e in events if e[0] in ("staged", "direct", "chunk_ready")],
            [],
            msg=f"Transfer after cancellation: {events}",
        )


if __name__ == "__main__":
    unittest.main()
