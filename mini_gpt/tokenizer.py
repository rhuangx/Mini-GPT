"""Byte-level BPE with one digit per token and a single <|bos|> document marker."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Regex, decoders, models, pre_tokenizers, trainers
from tokenizers import Tokenizer as Backend

# The GPT-4 / Llama-3 split with \p{N}{1,3} narrowed to \p{N}: merges never
# cross a pre-token boundary, so every digit stays its own token.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class Tokenizer:
    """BPE over the 256 bytes; encode -> decode is exact.

    <|bos|> takes the one ID after the backend's vocab and is never in the
    backend, so text that spells "<|bos|>" encodes as plain bytes.
    """

    def __init__(self, backend: Backend):
        self._tok = backend

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> Tokenizer:
        backend = Backend(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size - 1,  # the last ID is <|bos|>
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(texts, trainer=trainer)
        tok = cls(backend)
        if tok.vocab_size != vocab_size:
            raise ValueError(f"trained {tok.vocab_size} tokens, wanted {vocab_size}: corpus too small")
        return tok

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        return cls(Backend.from_file(str(path)))

    @property
    def bos_id(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size() + 1

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch_fast(texts, add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([i for i in ids if i != self.bos_id])

    def fingerprint(self) -> str:
        """sha256 of the serialized tokenizer; packed data records it."""
        return hashlib.sha256(self._tok.to_str().encode("utf-8")).hexdigest()
