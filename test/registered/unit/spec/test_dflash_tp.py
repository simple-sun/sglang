"""CPU regression tests for DFlash simulated TP decision synchronization."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.speculative.dflash_tp import (
    synchronize_dflash_simulated_acceptance,
)
from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TorchDistributedGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size

    def broadcast_capture_safe(self, values, src=0):
        dist.broadcast(values, src=src)
        return values


def _run_distributed_simulated_sync(
    rank: int,
    world_size: int,
    init_method: str,
    token_mode: str,
):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    try:
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)

        if rank == 0:
            forced_commit_len = 2
            target_predict = torch.tensor([[91, 92, 93, 94]], dtype=torch.int64)
        else:
            forced_commit_len = 4
            target_predict = torch.tensor([[191, 192, 193, 194]], dtype=torch.int64)

        accept_len = torch.tensor([0], dtype=torch.int32)
        commit_lens = torch.tensor([1], dtype=torch.int32)
        bonus = torch.tensor([0], dtype=torch.int64)
        out_tokens = torch.full_like(candidates, -1)
        prefix_lens = torch.tensor([7], dtype=torch.int64)

        with patch(
            "sglang.srt.speculative.dflash_utils.sample_simulated_acc_len",
            return_value=forced_commit_len,
        ):
            apply_dflash_simulated_acceptance(
                candidates=candidates,
                target_predict=target_predict,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=2.5,
                simulate_acc_method="match-expected",
                simulate_acc_token_mode=token_mode,
            )

        expected_local_accept_len = 1 if rank == 0 else 3
        expected_local_commit_len = 2 if rank == 0 else 4

        torch.testing.assert_close(
            accept_len,
            torch.tensor([expected_local_accept_len], dtype=torch.int32),
        )
        torch.testing.assert_close(
            commit_lens,
            torch.tensor([expected_local_commit_len], dtype=torch.int32),
        )

        if token_mode == "real-draft-token":
            expected_local_bonus = 92 if rank == 0 else 194
            torch.testing.assert_close(
                bonus,
                torch.tensor([expected_local_bonus], dtype=torch.int64),
            )

        synchronize_dflash_simulated_acceptance(
            tp_group=_TorchDistributedGroup(world_size),
            candidates=candidates,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            token_mode=token_mode,
        )

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)

        torch.testing.assert_close(
            accept_len,
            torch.tensor([1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            commit_lens,
            torch.tensor([2], dtype=torch.int32),
        )

        if token_mode == "fixed":
            torch.testing.assert_close(
                bonus,
                torch.tensor([100], dtype=torch.int64),
            )
            torch.testing.assert_close(
                out_tokens,
                torch.tensor([[100, 100, 100, 100]], dtype=torch.int64),
            )
        else:
            torch.testing.assert_close(
                bonus,
                torch.tensor([92], dtype=torch.int64),
            )
            torch.testing.assert_close(
                out_tokens,
                torch.tensor([[11, 92, 0, 0]], dtype=torch.int64),
            )

        torch.testing.assert_close(
            new_seq_lens,
            torch.tensor([9], dtype=torch.int64),
        )

        final_state = torch.cat(
            [
                accept_len.to(torch.int64),
                commit_lens.to(torch.int64),
                bonus.to(torch.int64),
                new_seq_lens.to(torch.int64),
                out_tokens.reshape(-1).to(torch.int64),
            ]
        )
        gathered_states = [torch.empty_like(final_state) for _ in range(world_size)]
        dist.all_gather(gathered_states, final_state)

        for gathered_state in gathered_states:
            torch.testing.assert_close(gathered_state, final_state)
    finally:
        dist.destroy_process_group()


def _spawn_distributed_simulated_sync(token_mode: str):
    world_size = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = Path(temp_dir) / "gloo_init"
        init_method = f"file://{init_file}"

        mp.spawn(
            _run_distributed_simulated_sync,
            args=(
                world_size,
                init_method,
                token_mode,
            ),
            nprocs=world_size,
            join=True,
        )


class _RankZeroReplayGroup:
    world_size = 2

    def __init__(self, accept_len: torch.Tensor, bonus: torch.Tensor):
        self._values = [accept_len, bonus]
        self.broadcast_count = 0

    def broadcast_capture_safe(self, values, src):
        if src != 0:
            raise AssertionError(f"expected rank zero, got {src}")
        values.copy_(self._values[self.broadcast_count])
        self.broadcast_count += 1
        return values


class TestDFlashSimulatedTpSync(CustomTestCase):
    def _apply_simulation(
        self,
        *,
        forced_commit_len: int,
        token_mode: str,
        target_predict: torch.Tensor,
    ):
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        accept_len = torch.tensor([0], dtype=torch.int32)
        commit_lens = torch.tensor([1], dtype=torch.int32)
        bonus = torch.tensor([20], dtype=torch.int64)
        out_tokens = torch.full_like(candidates, -1)

        with patch(
            "sglang.srt.speculative.dflash_utils.sample_simulated_acc_len",
            return_value=forced_commit_len,
        ):
            apply_dflash_simulated_acceptance(
                candidates=candidates,
                target_predict=target_predict,
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=2.5,
                simulate_acc_method="match-expected",
                simulate_acc_token_mode=token_mode,
            )

        return candidates, accept_len, commit_lens, bonus, out_tokens

    def test_tp_one_keeps_simulated_decision_without_collective(self):
        state = self._apply_simulation(
            forced_commit_len=3,
            token_mode="real-draft-token",
            target_predict=torch.tensor([[21, 22, 23, 24]], dtype=torch.int64),
        )
        candidates, accept_len, commit_lens, bonus, out_tokens = state
        group = SimpleNamespace(world_size=1, broadcast_capture_safe=MagicMock())

        synchronize_dflash_simulated_acceptance(
            tp_group=group,
            candidates=candidates,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            token_mode="real-draft-token",
        )

        group.broadcast_capture_safe.assert_not_called()
        torch.testing.assert_close(accept_len, torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(commit_lens, torch.tensor([3], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([23], dtype=torch.int64))
        torch.testing.assert_close(
            out_tokens, torch.tensor([[11, 12, 23, 0]], dtype=torch.int64)
        )

    def test_fixed_mode_reconciles_rank_local_forced_length(self):
        state = self._apply_simulation(
            forced_commit_len=4,
            token_mode="fixed",
            target_predict=torch.tensor([[21, 22, 23, 24]], dtype=torch.int64),
        )
        candidates, accept_len, commit_lens, bonus, out_tokens = state
        group = _RankZeroReplayGroup(
            accept_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.tensor([100], dtype=torch.int64),
        )

        synchronize_dflash_simulated_acceptance(
            tp_group=group,
            candidates=candidates,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            token_mode="fixed",
        )

        self.assertEqual(group.broadcast_count, 2)
        torch.testing.assert_close(accept_len, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(commit_lens, torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([100], dtype=torch.int64))
        torch.testing.assert_close(
            out_tokens, torch.tensor([[100, 100, 100, 100]], dtype=torch.int64)
        )
        torch.testing.assert_close(
            torch.tensor([7], dtype=torch.int64) + commit_lens,
            torch.tensor([9], dtype=torch.int64),
        )

    def test_real_draft_mode_reconciles_length_bonus_and_output(self):
        state = self._apply_simulation(
            forced_commit_len=4,
            token_mode="real-draft-token",
            target_predict=torch.tensor([[41, 42, 43, 44]], dtype=torch.int64),
        )
        candidates, accept_len, commit_lens, bonus, out_tokens = state
        group = _RankZeroReplayGroup(
            accept_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.tensor([99], dtype=torch.int64),
        )

        synchronize_dflash_simulated_acceptance(
            tp_group=group,
            candidates=candidates,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            token_mode="real-draft-token",
        )

        self.assertEqual(group.broadcast_count, 2)
        torch.testing.assert_close(accept_len, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(commit_lens, torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([99], dtype=torch.int64))
        torch.testing.assert_close(
            out_tokens, torch.tensor([[11, 99, 0, 0]], dtype=torch.int64)
        )
        torch.testing.assert_close(
            torch.tensor([7], dtype=torch.int64) + commit_lens,
            torch.tensor([9], dtype=torch.int64),
        )

    def test_fixed_mode_with_real_gloo_collective(self):
        _spawn_distributed_simulated_sync("fixed")

    def test_real_draft_mode_with_real_gloo_collective(self):
        _spawn_distributed_simulated_sync("real-draft-token")


if __name__ == "__main__":
    unittest.main()
