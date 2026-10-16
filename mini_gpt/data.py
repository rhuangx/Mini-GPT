"""FineWeb-Edu sample-10BT -> held-out val docs, a tokenizer, one shuffled uint16 token stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from mini_gpt.tokenizer import Tokenizer

REPO = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"
# One doc in VAL_EVERY is held out, picked by a hash of its id: ~1.9K docs and
# ~9 MB of text, spread over every crawl dump whatever the file order.
VAL_EVERY = 5000
DTYPE = np.uint16  # every vocab here is <= 65,536


def download(raw_dir: str | Path) -> list[Path]:
    """Fetch the parquet files; files already present are skipped."""
    snapshot_download(REPO, repo_type="dataset", allow_patterns=f"{SUBSET}/*", local_dir=raw_dir)
    return sorted(Path(raw_dir, SUBSET).glob("*.parquet"))


def is_val(doc_id: str) -> bool:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % VAL_EVERY == 0


def write_val(files: list[Path], path: str | Path) -> int:
    """Write the held-out docs as raw text, so any tokenizer can be scored on them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for file in files:
            pf = pq.ParquetFile(file)
            for rg in range(pf.num_row_groups):
                ids = pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()
                hits = [i for i, doc_id in enumerate(ids) if is_val(doc_id)]
                if not hits:
                    continue  # most row groups: skip decoding their text
                texts = pf.read_row_group(rg, columns=["text"]).column("text").take(hits).to_pylist()
                for i, text in zip(hits, texts, strict=True):
                    f.write(json.dumps({"id": ids[i], "text": text}, ensure_ascii=False) + "\n")
                    n += 1
    tmp.replace(path)
    return n


def load_val(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def sample_texts(files: list[Path], max_chars: int, *, seed: int = 0) -> Iterator[str]:
    """Train docs from row groups in random order, so every crawl dump is represented.

    Each file is sorted by dump, so its first rows would be all 2013.
    """
    pfs = {f: pq.ParquetFile(f) for f in files}
    groups = [(f, rg) for f, pf in pfs.items() for rg in range(pf.num_row_groups)]
    chars = 0
    for g in np.random.default_rng(seed).permutation(len(groups)):
        f, rg = groups[g]
        t = pfs[f].read_row_group(rg, columns=["id", "text"])
        for doc_id, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True):
            if is_val(doc_id):
                continue
            yield text
            chars += len(text)
            if chars >= max_chars:
                return


def _tokenize_file(job: tuple[str, str, str]) -> None:
    """One file's train docs -> <part>.bin (<|bos|> before each doc) and <part>.len.npy."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one process per file is the parallelism
    file, tokenizer_path, part = job
    tok = Tokenizer.load(tokenizer_path)
    pf = pq.ParquetFile(file)
    lengths: list[int] = []
    with open(part + ".bin", "wb") as f:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["id", "text"])
            docs = [
                text
                for doc_id, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True)
                if not is_val(doc_id)
            ]
            buf: list[int] = []
            for ids in tok.encode_batch(docs):
                buf.append(tok.bos_id)
                buf.extend(ids)
                lengths.append(len(ids) + 1)
            np.array(buf, dtype=DTYPE).tofile(f)
    np.save(part + ".len.npy", np.array(lengths, dtype=np.int64))


def pack(files: list[Path], tokenizer_path: str | Path, out_dir: str | Path, *, seed: int = 0) -> dict:
    """Tokenize every train doc, shuffle the docs globally, write train.bin and meta.json.

    The shuffle is what keeps the data stationary over a run: the files are
    sorted by crawl dump, so reading them in order would drift from 2013 to 2024.
    """
    tok = Tokenizer.load(tokenizer_path)
    assert tok.vocab_size <= 2**16, "token IDs must fit in uint16"
    out = Path(out_dir)
    tmp = out / "parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = [str(tmp / f"{i:03d}") for i in range(len(files))]
    jobs = [(str(f), str(tokenizer_path), p) for f, p in zip(files, parts, strict=True)]
    with mp.get_context("spawn").Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        pool.map(_tokenize_file, jobs)

    tokens = [np.memmap(p + ".bin", dtype=DTYPE, mode="r") for p in parts]
    lengths = [np.load(p + ".len.npy") for p in parts]
    part_of = np.concatenate([np.full(len(n), i) for i, n in enumerate(lengths)]).tolist()
    start = np.concatenate([np.cumsum(n) - n for n in lengths]).tolist()
    length = np.concatenate(lengths).tolist()
    order = np.random.default_rng(seed).permutation(len(length))
    with open(out / "train.bin.tmp", "wb") as f:
        for chunk in np.array_split(order, max(1, len(order) // 65_536)):
            docs = [tokens[part_of[i]][start[i] : start[i] + length[i]] for i in chunk.tolist()]
            np.concatenate(docs).tofile(f)
    (out / "train.bin.tmp").replace(out / "train.bin")

    meta = {
        "source": f"{REPO} {SUBSET}",
        "tokenizer_sha256": tok.fingerprint(),
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "docs": len(length),
        "tokens": sum(length),
        "shuffle_seed": seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(tmp)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu: download, hold out val docs, train the BPE, pack tokens. "
        "Finished steps are skipped on a rerun."
    )
    ap.add_argument("--raw", default="data/fineweb-edu", help="parquet download dir")
    ap.add_argument("--val", default="data/val.jsonl", help="held-out docs, shared by every tokenizer")
    ap.add_argument("--vocab-size", type=int, default=16_384)
    ap.add_argument("--out", default=None, help="tokenizer and packed tokens (default data/v<vocab/1024>k)")
    ap.add_argument("--tokenizer-chars", type=float, default=1e9, help="train-text characters for the BPE")
    args = ap.parse_args(argv)
    out = Path(args.out or f"data/v{args.vocab_size // 1024}k")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    files = download(args.raw)
    print(f"download: {len(files)} parquet files in {args.raw}")
    if not Path(args.val).exists():
        n = write_val(files, args.val)
        print(f"val: {n} held-out docs -> {args.val} ({time.time() - t0:.0f}s)")

    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        tok = Tokenizer.train(sample_texts(files, int(args.tokenizer_chars)), args.vocab_size)
        tok.save(tok_path)
        print(f"tokenizer: {tok.vocab_size} tokens -> {tok_path} ({time.time() - t0:.0f}s)")

    if not (out / "train.bin").exists():
        meta = pack(files, tok_path, out)
        print(f"packed: {meta['tokens']:,} tokens, {meta['docs']:,} docs -> {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
