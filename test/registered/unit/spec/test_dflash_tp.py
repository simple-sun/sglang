"""CPU regression tests for DFlash tensor-parallel decision synchronization."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed import bootstrap
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.dflash_tp import (
    DFlashTpSync,
    finalize_dflash_tp_decision,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RankZeroReplayGroup:
    world_size = 2

    def __init__(self, rank_zero_values):
        self.rank_zero_values = rank_zero_values
        self.broadcast_count = 0

    def broadcast_capture_safe(self, values, src):
        if src != 0:
            raise AssertionError(f"expected rank zero, got {src}")
        values.copy_(self.rank_zero_values[self.broadcast_count])
        self.broadcast_count += 1
        return values


class TestDFlashTpSync(CustomTestCase):
    def test_tp_one_is_identity_without_collective(self):
        group = SimpleNamespace(world_size=1, broadcast_capture_safe=MagicMock())
        values = torch.tensor([1, 2])

        self.assertIs(DFlashTpSync(group).sync(values), values)
        group.broadcast_capture_safe.assert_not_called()

    def test_tp_broadcast_uses_rank_zero_capture_safe_collective(self):
        values = torch.tensor([3, 4])
        group = SimpleNamespace(
            world_size=4, broadcast_capture_safe=MagicMock(return_value=values)
        )

        self.assertIs(DFlashTpSync(group).sync(values), values)
        group.broadcast_capture_safe.assert_called_once_with(values, src=0)

    def test_finalize_uses_two_authoritative_broadcasts_before_deriving_outputs(self):
        group = _RankZeroReplayGroup(
            [
                torch.tensor([3], dtype=torch.int32),
                torch.tensor([100], dtype=torch.int64),
            ]
        )
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        out_tokens = torch.full_like(candidates, -1)
        accept_len = torch.tensor([1], dtype=torch.int32)
        bonus = torch.tensor([200], dtype=torch.int64)

        commit_lens, new_seq_lens = finalize_dflash_tp_decision(
            tp_sync=DFlashTpSync(group),
            candidates=candidates,
            accept_len=accept_len,
            bonus=bonus,
            out_tokens=out_tokens,
            prefix_lens=torch.tensor([100], dtype=torch.int64),
        )

        self.assertEqual(group.broadcast_count, 2)
        torch.testing.assert_close(accept_len, torch.tensor([3], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([100], dtype=torch.int64))
        torch.testing.assert_close(
            out_tokens, torch.tensor([[11, 12, 13, 100]], dtype=torch.int64)
        )
        torch.testing.assert_close(commit_lens, torch.tensor([4], dtype=torch.int32))
        torch.testing.assert_close(new_seq_lens, torch.tensor([104], dtype=torch.int64))

    def test_finalize_preserves_fixed_simulated_acceptance_outputs(self):
        group = _RankZeroReplayGroup(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([100], dtype=torch.int64),
            ]
        )
        out_tokens = torch.full((1, 4), -1, dtype=torch.int64)

        commit_lens, new_seq_lens = finalize_dflash_tp_decision(
            tp_sync=DFlashTpSync(group),
            candidates=torch.tensor([[10, 11, 12, 13]], dtype=torch.int64),
            accept_len=torch.tensor([3], dtype=torch.int32),
            bonus=torch.tensor([999], dtype=torch.int64),
            out_tokens=out_tokens,
            prefix_lens=torch.tensor([7], dtype=torch.int64),
            fill_all_with_bonus=True,
        )

        self.assertEqual(group.broadcast_count, 2)
        torch.testing.assert_close(
            out_tokens, torch.tensor([[100, 100, 100, 100]], dtype=torch.int64)
        )
        torch.testing.assert_close(commit_lens, torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(new_seq_lens, torch.tensor([9], dtype=torch.int64))

    def test_finalize_zeros_uncommitted_real_draft_tokens(self):
        group = _RankZeroReplayGroup(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([99], dtype=torch.int64),
            ]
        )
        out_tokens = torch.full((1, 4), -1, dtype=torch.int64)

        finalize_dflash_tp_decision(
            tp_sync=DFlashTpSync(group),
            candidates=torch.tensor([[10, 11, 12, 13]], dtype=torch.int64),
            accept_len=torch.tensor([3], dtype=torch.int32),
            bonus=torch.tensor([999], dtype=torch.int64),
            out_tokens=out_tokens,
            prefix_lens=torch.tensor([7], dtype=torch.int64),
            zero_uncommitted=True,
        )

        torch.testing.assert_close(
            out_tokens, torch.tensor([[11, 99, 0, 0]], dtype=torch.int64)
        )


class TestDFlashTpIntegration(CustomTestCase):
    def test_dp_attention_selects_attention_tp_group(self):
        source = textwrap.dedent(inspect.getsource(DFlashWorkerV2.__init__))
        tree = ast.parse(source)
        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DFlashTpSync"
        ]
        self.assertEqual(len(constructors), 1)
        self.assertEqual(
            ast.unparse(constructors[0].args[0]),
            "parallel.attn_tp_group if server_args.enable_dp_attention else parallel.tp_group",
        )

    def test_dflash_dp_decode_graph_provisions_attention_pynccl(self):
        graph = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL),
            prefill=SimpleNamespace(backend=Backend.DISABLED),
        )
        args = SimpleNamespace(
            cuda_graph_config=graph,
            speculative_algorithm="dflash",
            enable_dp_attention=True,
        )
        with (
            patch.object(bootstrap.current_platform, "is_cuda", return_value=True),
            patch.object(bootstrap.current_platform, "is_rocm", return_value=False),
            patch.object(
                bootstrap.envs.SGLANG_DSA_TOPK_BROADCAST, "get", return_value=False
            ),
        ):
            self.assertTrue(bootstrap._needs_attn_tp_pynccl(args))
            args.enable_dp_attention = False
            self.assertFalse(bootstrap._needs_attn_tp_pynccl(args))


if __name__ == "__main__":
    unittest.main()
