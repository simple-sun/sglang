from __future__ import annotations

import torch


class DFlashTpSync:
    """Keep DFlash token decisions consistent across tensor-parallel ranks."""

    def __init__(self, group) -> None:
        self._group = group
        self._enabled = group.world_size > 1

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sync(self, values: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return values
        return self._group.broadcast_capture_safe(values, src=0)


def finalize_dflash_tp_decision(
    *,
    tp_sync: DFlashTpSync,
    candidates: torch.Tensor,
    accept_len: torch.Tensor,
    bonus: torch.Tensor,
    out_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    fill_all_with_bonus: bool = False,
    zero_uncommitted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synchronize rank-0's decision, then derive all dependent token state."""
    assert not (fill_all_with_bonus and zero_uncommitted)
    tp_sync.sync(accept_len)
    tp_sync.sync(bonus)

    block_size = int(candidates.shape[1])
    commit_lens = accept_len.to(torch.int32) + 1
    if fill_all_with_bonus:
        out_tokens.copy_(bonus[:, None].expand_as(out_tokens))
    else:
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].zero_()
        out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        if zero_uncommitted:
            positions = torch.arange(block_size, device=out_tokens.device)
            out_tokens.masked_fill_(positions[None, :] >= commit_lens[:, None], 0)

    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return commit_lens, new_seq_lens
