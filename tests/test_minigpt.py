"""Checks ordered like the package: tokenizer, data, model, FLOPs, val bpb, resume, scaling fits."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from mini_gpt import data, scaling
from mini_gpt.flops import flops_per_token
from mini_gpt.model import GPT, IGNORE_INDEX, GPTConfig
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.train import TrainConfig, load_config, train, val_bpb, val_windows

BASELINE = Path(__file__).resolve().parent.parent / "configs" / "baseline.json"


def tiny_config(steps: int = 30) -> tuple[GPTConfig, TrainConfig]:
    """configs/baseline.json shrunk to train `steps` steps on a CPU in seconds."""
    model_cfg, cfg = load_config(BASELINE, [
        "model.vocab_size=512", "model.context=32", "model.d_model=64", "model.n_layers=2",
        "model.head_dim=32", "train.batch_size=4", "train.micro_batch=2", "train.warmup_steps=5",
    ])
    flops = steps * cfg.batch_size * model_cfg.context * flops_per_token(model_cfg)
    return model_cfg, dataclasses.replace(cfg, flops=flops)


def random_doc(rng: random.Random) -> str:
    words = [*("".join(rng.choices("abcdefghij", k=rng.randint(2, 7))) for _ in range(40)),
             "2024", "3.14159", "naïve", "中文", "🙂", "<|bos|>", "\n\n", "\t"]
    return " ".join(rng.choice(words) for _ in range(rng.randint(20, 80)))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Two small parquet files shaped like FineWeb-Edu's, prepared by the real pipeline."""
    rng = random.Random(0)
    val_ids, train_ids = [], []
    i = 0
    while len(val_ids) < 4 or len(train_ids) < 300:
        doc_id = f"<urn:uuid:{i}>"
        if data.is_val(doc_id):
            val_ids.append(doc_id)
        elif len(train_ids) < 300:
            train_ids.append(doc_id)
        i += 1
    docs = {doc_id: random_doc(rng) for doc_id in val_ids + train_ids}
    order = list(docs)
    rng.shuffle(order)

    root = tmp_path_factory.mktemp("corpus")
    files = []
    for k, ids in enumerate((order[: len(order) // 2], order[len(order) // 2 :])):
        path = root / f"{k:03d}_00000.parquet"
        pq.write_table(pa.table({"id": ids, "text": [docs[d] for d in ids]}), path, row_group_size=50)
        files.append(path)

    data.write_val(files, root / "val.jsonl")
    Tokenizer.train(data.sample_texts(files, 10**9), vocab_size=512).save(root / "tokenizer.json")
    data.pack(files, root / "tokenizer.json", root)
    return {"root": root, "files": files, "docs": docs, "val_ids": val_ids, "order": order}


@pytest.fixture(scope="session")
def tok(corpus) -> Tokenizer:
    return Tokenizer.load(corpus["root"] / "tokenizer.json")


# Tokenizer

def test_tokenizer_round_trips_bytes(tok):
    for text in ["plain ascii", "naïve 中文 🙂", "tabs\tand\r\nnewlines\n\n", "  leading", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_splits_digits_one_per_token(tok):
    pieces = [tok.decode([i]) for i in tok.encode("in 2024 pi was 3.14159, not 22/7")]
    assert "".join(pieces) == "in 2024 pi was 3.14159, not 22/7"
    assert [p for p in pieces if any(c.isdigit() for c in p)] == list("2024314159227")


def test_special_token_text_encodes_as_plain_text(tok):
    ids = tok.encode("a <|bos|> b")
    assert tok.bos_id not in ids
    assert tok.decode(ids) == "a <|bos|> b"
    assert tok.vocab_size == tok.bos_id + 1 == 512


# Data

def test_pack_holds_out_val_and_keeps_every_train_doc_once(corpus, tok):
    root = corpus["root"]
    with (root / "val.jsonl").open() as f:
        val = [json.loads(line) for line in f]
    assert sorted(d["id"] for d in val) == sorted(corpus["val_ids"])
    assert all(d["text"] == corpus["docs"][d["id"]] for d in val)

    tokens = np.fromfile(root / "train.bin", dtype=np.uint16)
    meta = json.loads((root / "meta.json").read_text())
    assert meta["tokens"] == len(tokens) and meta["tokenizer_sha256"] == tok.fingerprint()
    starts = np.flatnonzero(tokens == tok.bos_id)
    assert starts[0] == 0 and len(starts) == meta["docs"]
    packed = [tok.decode(doc.tolist()) for doc in np.split(tokens, starts[1:])]

    train_ids = [d for d in corpus["order"] if d not in set(corpus["val_ids"])]
    in_file_order = [corpus["docs"][d] for d in train_ids]
    assert Counter(packed) == Counter(in_file_order)
    assert packed != in_file_order  # shuffled across files, not concatenated


# Model and FLOPs

def test_param_counts_match_the_module():
    model_cfg, _ = tiny_config()
    counts = model_cfg.param_counts()
    model = GPT(model_cfg)
    assert model.embed.weight.numel() == counts["embedding"]
    assert sum(p.numel() for p in model.parameters()) == counts["embedding"] + counts["non_embedding"]


def test_flops_per_token_matches_the_profiler():
    model_cfg, _ = tiny_config()
    model = GPT(model_cfg)
    x = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.context))
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(x, x).mean().backward()
    # The profiler skips the norm gains, which are not matmuls, and charges
    # causal attention for the full T x T score matrix, where the formula
    # charges the context / 2 keys a query attends on average.
    norm_gains = (2 * model_cfg.n_layers + 1) * model_cfg.d_model
    causal_half = 6 * model_cfg.n_layers * model_cfg.d_model * model_cfg.context
    expected = x.numel() * (flops_per_token(model_cfg) - 6 * norm_gains + causal_half)
    assert counter.get_total_flops() == expected


# Validation bpb

def test_uniform_logits_score_log2_vocab_bits_per_token(corpus, tok):
    model_cfg, _ = tiny_config()
    model = GPT(model_cfg)
    nn.init.zeros_(model.embed.weight)  # zero hidden states, so zero logits: uniform over the vocab
    docs = data.load_val(corpus["root"] / "val.jsonl")
    x, y, n_bytes = val_windows(docs, tok, model_cfg.context, rows=3)
    tokens = sum(len(tok.encode(d)) for d in docs)
    assert (y != IGNORE_INDEX).sum().item() == tokens  # each doc token once; no markers, no padding
    assert n_bytes == sum(len(d.encode("utf-8")) for d in docs)
    bpb = val_bpb(model, x, y, n_bytes, 3, torch.device("cpu"))
    assert bpb == pytest.approx(math.log2(model_cfg.vocab_size) * tokens / n_bytes, rel=1e-6)


# Training

def test_resumed_run_reproduces_the_uninterrupted_curve(corpus, tmp_path):
    model_cfg, cfg = tiny_config()
    paths = {"data_dir": corpus["root"], "val_path": corpus["root"] / "val.jsonl",
             "results_path": tmp_path / "runs.jsonl", "device": "cpu", "compile": False}
    whole = train(model_cfg, cfg, out_dir=tmp_path / "whole", **paths)
    assert train(model_cfg, cfg, out_dir=tmp_path / "split", stop_at=13, **paths) is None
    resumed = train(model_cfg, cfg, out_dir=tmp_path / "split", **paths)

    def losses(row):
        return [(p["step"], p["train_loss"], p.get("val_bpb")) for p in row["curve"]]

    assert len(whole["curve"]) == whole["steps"] == 30
    assert losses(resumed) == losses(whole)
    assert whole["curve"][-1]["train_loss"] < whole["curve"][0]["train_loss"]
    rows = [json.loads(line) for line in (tmp_path / "runs.jsonl").open()]
    assert [r["run"] for r in rows] == ["whole", "split"]  # the stopped session wrote no row


# Scaling fits

def fake_row(d_model: int, lr: float, seed: int, bpb: float) -> dict:
    """A results row for configs/baseline.json at this width, LR and seed, finishing at bpb."""
    model_cfg, cfg = load_config(BASELINE, [f"model.d_model={d_model}", f"train.lr={lr}", f"train.seed={seed}"])
    return {"run": f"d{d_model}-lr{lr:g}-s{seed}", "params": model_cfg.param_counts(), "val_bpb": bpb,
            "config": {"model": dataclasses.asdict(model_cfg), "train": dataclasses.asdict(cfg)},
            "curve": [{"flops": 1e14, "train_loss": 6.0, "val_bpb": bpb + 0.5},
                      {"flops": 1e15, "train_loss": 5.0, "val_bpb": bpb}]}


def test_lr_optimum_is_the_middle_of_the_near_best_band():
    lrs = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
    bpbs = [1.4752, 1.4427, 1.4420, 1.4393, 1.7575]  # the 3.1M sweep: flat bottom, cliff at 8e-3
    assert scaling.lr_optimum(lrs, bpbs, tol=0.005) == pytest.approx((2e-3, 1.4393))
    assert scaling.lr_optimum(lrs, bpbs, tol=0.001) == pytest.approx((4e-3, 1.4393))
    assert scaling.lr_optimum(lrs, [1.0, 2.0, 3.0, 4.0, 5.0], tol=0.005) is None  # best at the edge
    assert scaling.lr_optimum(lrs, [2.0, 1.0, float("nan"), 3.0, 3.0], tol=0.005) == pytest.approx((1e-3, 1.0))


def test_lr_sweep_recovers_the_power_law_and_seeds_give_the_spread(tmp_path):
    rows = []
    for d_model in (128, 256, 512):
        n = load_config(BASELINE, [f"model.d_model={d_model}"])[0].param_counts()["non_embedding"]
        lr_star = 0.1 * n**-0.5
        for f in (0.25, 0.5, 1, 2, 4):
            rows.append(fake_row(d_model, lr_star * f, 0, 1.0 + math.log(f) ** 2))
    rows += [fake_row(128, rows[2]["config"]["train"]["lr"], seed, rows[2]["val_bpb"] + delta)
             for seed, delta in ((1, 0.01), (2, -0.01))]  # mean unchanged, std 0.01

    result = scaling.lr_sweep(rows, tmp_path / "lr.png", tol=0.005)
    assert result["power_law"] == pytest.approx((0.1, -0.5))
    assert [f["bpb_star"] for f in result["variants"]] == pytest.approx([1.0] * 3)
    [spread] = scaling.seed_spread(rows)
    assert (spread["seeds"], spread["std"]) == (3, pytest.approx(0.01))
    scaling.plot_curves(rows[:3], tmp_path / "curves.png")
    assert (tmp_path / "lr.png").stat().st_size > 0 and (tmp_path / "curves.png").stat().st_size > 0
