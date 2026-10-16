"""Training FLOPs per token, the one compute count every run is budgeted and compared by."""

from __future__ import annotations

from mini_gpt.model import GPTConfig


def flops_per_token(cfg: GPTConfig) -> int:
    """6 per weight a token passes through, plus attention over the keys it attends.

    The weights are the non-embedding params plus the tied LM head; the input
    embedding lookup costs nothing. Attention costs 12 * n_layers * d_attn per
    key attended (QK^T and AV, forward and backward), and a causal query
    attends context / 2 keys on average.
    """
    lm_head = cfg.vocab_size * cfg.d_model
    attention = 12 * cfg.n_layers * cfg.d_model * cfg.context // 2
    return 6 * (cfg.param_counts()["non_embedding"] + lm_head) + attention
