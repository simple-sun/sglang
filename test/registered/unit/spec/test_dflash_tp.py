"""CPU regression tests for DFlash simulated TP decision synchronization."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dflash_tp import (
    synchronize_dflash_simulated_acceptance,
)
from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
