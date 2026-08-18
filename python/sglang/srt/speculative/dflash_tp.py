from __future__ import annotations

import torch


def synchronize_dflash_simulated_acceptance(
    *,
    tp_group,
    candidates: torch.Tensor,
    accept_len: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus: torch.Tensor,
    out_tokens: torch.Tensor,
    token_mode: str,
) -> None:
    """Synchronize a simulated decision and rebuild its dependent token state."""
    if tp_group.world_size > 1:
        tp_group.broadcast_capture_safe(accept_len, src=0)
        tp_group.broadcast_capture_safe(bonus, src=0)

    commit_lens.copy_(accept_len.to(torch.int32) + 1)
    if token_mode == "fixed":
        out_tokens.copy_(bonus[:, None].expand_as(out_tokens))
        return

    block_size = int(candidates.shape[1])
    out_tokens.zero_()
    if block_size > 1:
        out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
    out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
    positions = torch.arange(block_size, device=out_tokens.device)
    out_tokens.masked_fill_(positions[None, :] >= commit_lens[:, None], 0)
