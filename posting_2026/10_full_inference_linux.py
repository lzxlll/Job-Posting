# -*- coding: utf-8 -*-
"""
10_full_inference_linux.py — Full-scale SOC inference on all 641M rows (Linux/Aliyun)
=======================================================================================
Designed for Alibaba Cloud ECS gn7i-4x (4× NVIDIA A10, 64 vCPU).
Run 4 parallel processes, one per GPU, each handling a disjoint chunk range.

Expected /data layout
---------------------
  /data/parquet/招聘/data.parquet          ← source (read-only)
  /data/enterprise.duckdb                  ← DuckDB (created if absent)
  /data/model/chinese-bert-wwm/            ← tokenizer
  /data/model/bert_soc/                    ← fine-tuned BertForSequenceClassification
  /data/model/est_sample.csv               ← label CSV
  /data/parquet/招聘_soc_full/             ← Phase B chunk output + per-GPU checkpoints
  /data/parquet/招聘_soc_title_lookup.parquet  ← Phase A lookup (upload from Windows)
  /data/parquet/招聘_soc_all_labels.parquet    ← final sidecar (written by combine)

Resuming from Windows (chunks 0-2 done, Phase A done)
------------------------------------------------------
Upload from Windows first (ossutil), then on the VM run 4 parallel Phase B processes:

  CUDA_VISIBLE_DEVICES=0 nohup python 10_full_inference_linux.py \
      --phase b --gpu_id 0 --start_chunk 3  --end_chunk 83  \
      --batch_full 1024 --tok_workers 4 > /data/logs/gpu0.log 2>&1 &

  CUDA_VISIBLE_DEVICES=1 nohup python 10_full_inference_linux.py \
      --phase b --gpu_id 1 --start_chunk 83 --end_chunk 163 \
      --batch_full 1024 --tok_workers 4 > /data/logs/gpu1.log 2>&1 &

  CUDA_VISIBLE_DEVICES=2 nohup python 10_full_inference_linux.py \
      --phase b --gpu_id 2 --start_chunk 163 --end_chunk 243 \
      --batch_full 1024 --tok_workers 4 > /data/logs/gpu2.log 2>&1 &

  CUDA_VISIBLE_DEVICES=3 nohup python 10_full_inference_linux.py \
      --phase b --gpu_id 3 --start_chunk 243 --end_chunk 321 \
      --batch_full 1024 --tok_workers 4 > /data/logs/gpu3.log 2>&1 &

After all 4 finish (or to combine partial results):
  python 10_full_inference_linux.py --phase combine

Spot instance / preemption: each completed chunk is written atomically and recorded
in checkpoint_gpu{N}.json before moving on — safe to kill and restart at any time.

Options
-------
    --phase {a,b,combine,all}
    --gpu_id      INT    (0-3, selects checkpoint_gpu{N}.json, default 0)
    --start_chunk INT    (first chunk index to process, inclusive, default 0)
    --end_chunk   INT    (last chunk index, exclusive, default = all remaining)
    --batch_full  INT    (default 1024 for A10 24GB)
    --batch_title INT    (default 1024)
    --tok_workers INT    (tokenizer threads, default 4 per GPU process)
    --chunk_rows  INT    (default 2000000)
    --reset
"""

import sys
import os
import json
import time
import math
import unicodedata
import argparse
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PARQUET_SRC   = Path("/data/parquet/招聘/data.parquet")
DB_PATH       = Path("/data/enterprise.duckdb")
TOKENIZER_DIR = Path("/data/model/chinese-bert-wwm")
MODEL_DIR     = Path("/data/model/bert_soc")
LABEL_SRC     = Path("/data/model/est_sample.csv")
ONNX_PATH     = MODEL_DIR / "bert_soc_406.onnx"
OUTPUT_BASE   = Path("/data/parquet")
CHUNK_DIR     = OUTPUT_BASE / "招聘_soc_full"
TITLE_LOOKUP  = OUTPUT_BASE / "招聘_soc_title_lookup.parquet"
FINAL_OUTPUT  = OUTPUT_BASE / "招聘_soc_all_labels.parquet"
CHECKPOINT    = CHUNK_DIR / "checkpoint.json"   # Phase A / combine shared state

def checkpoint_path(gpu_id: int) -> Path:
    """Per-GPU checkpoint so parallel processes don't overwrite each other."""
    return CHUNK_DIR / f"checkpoint_gpu{gpu_id}.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_BATCH_FULL  = 1024   # A10 24GB VRAM — safe headroom at max_len=512
DEFAULT_BATCH_TITLE = 1024
DEFAULT_CHUNK_ROWS  = 2_000_000
DEFAULT_MAX_LEN     = 512

PLACEHOLDERS = {"", "-", "--"}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def banner(msg: str):
    width = 66
    print(f"\n{'═'*width}")
    print(f"  {msg}")
    print(f"{'═'*width}")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def is_placeholder_str(s) -> bool:
    if s is None:
        return True
    return str(s).strip() in PLACEHOLDERS


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(state: dict, path: Path):
    state["last_updated"] = now_str()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def init_checkpoint(chunk_rows: int, total_chunks: int) -> dict:
    return {
        "chunk_rows":     chunk_rows,
        "total_chunks":   total_chunks,
        "phase_a_done":   False,
        "done":           [],
        "rows_full":      {},
        "rows_titleonly": {},
        "rows_skipped":   {},
        "combine_done":   False,
        "created_at":     now_str(),
    }


class Progress:
    def __init__(self, total: int, label: str, interval_sec: int = 30):
        self.total    = total
        self.label    = label
        self.done     = 0
        self.t0       = time.time()
        self.last     = self.t0
        self.interval = interval_sec

    def update(self, n: int, force: bool = False):
        self.done += n
        now = time.time()
        if force or (now - self.last) >= self.interval:
            elapsed = now - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            pct     = 100 * self.done / self.total if self.total > 0 else 0
            eta_hr  = (self.total - self.done) / rate / 3600 if rate > 0 else 0
            print(
                f"  [{self.label}] {pct:5.1f}%  "
                f"{self.done:>12,}/{self.total:,}  "
                f"{rate:,.0f} rows/sec  ETA {eta_hr:.1f} hr",
                flush=True,
            )
            self.last = now


# ══════════════════════════════════════════════════════════════════════════════
# Label map
# ══════════════════════════════════════════════════════════════════════════════

def build_label_map() -> dict:
    import pandas as pd
    print("  Building label map from est_sample.csv ...")
    df = pd.read_csv(LABEL_SRC, encoding="utf_8_sig", on_bad_lines="skip",
                     encoding_errors="ignore")
    df["soc_code"] = df["soc_code"].str.replace("-", "", regex=False)
    unique_codes   = sorted(df["soc_code"].unique())
    inv_map        = {i: code for i, code in enumerate(unique_codes)}
    print(f"  Label map: {len(inv_map)} classes  ({unique_codes[0]} … {unique_codes[-1]})")
    return inv_map


# ══════════════════════════════════════════════════════════════════════════════
# Model / tokenizer
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification

    print(f"  Loading tokenizer from {TOKENIZER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    print(f"  Loading model from {MODEL_DIR} ...")
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
        props  = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")
    else:
        device = "cpu"
        print("  WARNING: CUDA not available — running on CPU (much slower)")

    model = model.to(device)
    return model, tokenizer, device


# ══════════════════════════════════════════════════════════════════════════════
# Core GPU inference loop
# ══════════════════════════════════════════════════════════════════════════════

def infer_texts(
    texts: list,
    model,
    tokenizer,
    label_map: dict,
    batch_size: int,
    device: str,
    max_len: int = DEFAULT_MAX_LEN,
    tok_workers: int = 8,
    progress: Progress = None,
    report_every: int = 30,    # seconds between progress prints
) -> tuple:
    """
    Batched GPU inference with length-sorted batching + multi-worker double-buffering.

    Texts are sorted longest-first so each batch has uniform sequence length,
    minimising padding waste in BERT's O(seq_len²) attention.
    tok_workers background threads tokenize ahead of the GPU to keep it fed.
    Original order is restored before returning.
    Returns (preds: list[str], probs: list[float]).
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    import numpy as np

    n           = len(texts)
    t0          = time.time()
    last_report = t0

    # Sort by descending char length ≈ token length (Chinese: 1 char ≈ 1 token).
    order        = np.argsort([len(t) for t in texts])[::-1]
    inv_order    = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n)
    sorted_texts = [texts[i] for i in order]

    def _tok(start: int):
        return tokenizer(
            sorted_texts[start: start + batch_size],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

    preds_sorted, probs_sorted = [], []
    starts = list(range(0, n, batch_size))
    window = tok_workers * 2        # batches to keep in-flight ahead of GPU

    with ThreadPoolExecutor(max_workers=tok_workers) as ex:
        # Pre-fill the sliding window
        pending = deque(ex.submit(_tok, s) for s in starts[:window])

        for batch_idx, start in enumerate(starts):
            enc       = pending.popleft().result()   # wait for next tokenized batch
            batch_len = enc["input_ids"].shape[0]

            # Submit the next batch immediately (overlaps with GPU forward pass)
            ahead = batch_idx + window
            if ahead < len(starts):
                pending.append(ex.submit(_tok, starts[ahead]))

            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(**enc).logits
                else:
                    logits = model(**enc).logits

            p = torch.softmax(logits, dim=-1)
            preds_sorted.extend([label_map[idx] for idx in logits.argmax(dim=-1).cpu().tolist()])
            probs_sorted.extend(p.max(dim=-1).values.cpu().tolist())

            done = start + batch_len
            now  = time.time()
            if now - last_report >= report_every:
                last_report = now
                elapsed = now - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta_min = (n - done) / rate / 60 if rate > 0 else 0
                pct     = 100 * done / n
                print(
                    f"    {pct:5.1f}%  {done:>9,}/{n:,}  "
                    f"{rate:,.0f} seq/sec  ETA {eta_min:.0f} min",
                    flush=True,
                )
            if progress is not None:
                progress.update(batch_len)

    elapsed = time.time() - t0
    print(
        f"  Done: {n:,} seqs in {elapsed/60:.1f} min  ({n/elapsed:,.0f} seq/sec avg)",
        flush=True,
    )
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak:.2f} GB", flush=True)

    preds = [preds_sorted[inv_order[i]] for i in range(n)]
    probs = [probs_sorted[inv_order[i]] for i in range(n)]
    return preds, probs


# ══════════════════════════════════════════════════════════════════════════════
# Phase A — Title-only deduplication
# ══════════════════════════════════════════════════════════════════════════════

def phase_a(model, tokenizer, device, label_map, ckpt, args):
    import polars as pl
    import duckdb

    banner("PHASE A — Title-only deduplication & inference")

    if ckpt.get("phase_a_done") and TITLE_LOOKUP.exists():
        print("  Phase A already complete — skipping (delete checkpoint to redo).")
        return

    print("  Querying distinct 工作名称 for placeholder-description rows ...")
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df_titles = con.execute(f"""
        SELECT DISTINCT 工作名称
        FROM read_parquet('{PARQUET_SRC.as_posix()}')
        WHERE (职责描述 IS NULL OR TRIM(职责描述) IN ('', '-', '--'))
          AND 工作名称 IS NOT NULL
          AND TRIM(工作名称) NOT IN ('', '-', '--')
    """).pl()
    con.close()
    n_unique = len(df_titles)
    print(f"  Unique titles found: {n_unique:,}  ({time.time()-t0:.0f}s)")

    titles_raw = df_titles["工作名称"].to_list()
    titles_nfc = [nfc(t) for t in titles_raw]
    texts      = [f"{t} {t}" for t in titles_nfc]

    print(f"\n  Inference: {n_unique:,} unique titles | batch={args.batch_title}")
    prog = Progress(n_unique, "Phase A", interval_sec=30)
    preds, probs = infer_texts(
        texts, model, tokenizer, label_map,
        batch_size=args.batch_title,
        device=device,
        progress=prog,
        report_every=30,
    )

    lookup = pl.DataFrame({
        "工作名称":      pl.Series(titles_nfc, dtype=pl.Utf8),
        "soc_code_pred": pl.Series(preds,      dtype=pl.Utf8),
        "soc_prob":      pl.Series(probs,       dtype=pl.Float32),
    })
    TITLE_LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    lookup.write_parquet(TITLE_LOOKUP, compression="zstd")
    mb = TITLE_LOOKUP.stat().st_size / 1e6
    print(f"\n  Lookup saved → {TITLE_LOOKUP}  ({mb:.1f} MB, {len(lookup):,} rows)")

    ckpt["phase_a_done"] = True
    save_checkpoint(ckpt, CHECKPOINT)
    print("  Phase A complete ✓")


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — Chunked full-dataset inference
# ══════════════════════════════════════════════════════════════════════════════

def _write_atomic(df, path: Path):
    tmp = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def phase_b(model, tokenizer, device, label_map, ckpt, args):
    import pyarrow.parquet as pq
    import polars as pl

    start_chunk = args.start_chunk
    end_chunk   = args.end_chunk        # None → process to end
    gpu_ckpt    = checkpoint_path(args.gpu_id)

    banner(f"PHASE B — GPU {args.gpu_id}  chunks [{start_chunk}, "
           f"{'end' if end_chunk is None else end_chunk})")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    pf           = pq.ParquetFile(str(PARQUET_SRC))
    total_rows   = pf.metadata.num_rows
    total_chunks = math.ceil(total_rows / args.chunk_rows)
    my_end       = end_chunk if end_chunk is not None else total_chunks
    my_n_chunks  = my_end - start_chunk

    print(f"  Source      : {PARQUET_SRC}")
    print(f"  Total rows  : {total_rows:,}  ({total_chunks} chunks)")
    print(f"  This GPU    : chunks {start_chunk}–{my_end-1}  ({my_n_chunks} chunks)")
    print(f"  Batch sizes : full={args.batch_full}  title={args.batch_title}")
    print(f"  tok_workers : {args.tok_workers}")
    print(f"  Checkpoint  : {gpu_ckpt}")

    # Load this GPU's own checkpoint
    my_ckpt = load_checkpoint(gpu_ckpt)
    if not my_ckpt:
        my_ckpt = {"chunk_rows": args.chunk_rows, "total_chunks": total_chunks,
                   "done": [], "rows_full": {}, "rows_titleonly": {}, "rows_skipped": {}}
        save_checkpoint(my_ckpt, gpu_ckpt)

    if my_ckpt.get("chunk_rows", args.chunk_rows) != args.chunk_rows:
        print(f"\n  ERROR: checkpoint chunk_rows={my_ckpt['chunk_rows']} "
              f"!= --chunk_rows={args.chunk_rows}.  Use --reset to restart.")
        sys.exit(1)

    done_set = set(my_ckpt.get("done", []))
    print(f"\n  Chunks done so far (this GPU): {len(done_set)}/{my_n_chunks}")

    COLS    = ["表ID", "工作名称", "职责描述"]
    t_phase = time.time()

    for chunk_idx, batch in enumerate(pf.iter_batches(batch_size=args.chunk_rows,
                                                       columns=COLS)):
        # Skip chunks outside this GPU's range
        if chunk_idx < start_chunk:
            continue
        if chunk_idx >= my_end:
            break

        chunk_full_path  = CHUNK_DIR / f"chunk_{chunk_idx:05d}.parquet"
        chunk_title_path = CHUNK_DIR / f"titleonly_{chunk_idx:05d}.parquet"

        if chunk_idx in done_set \
                and chunk_full_path.exists()  and chunk_full_path.stat().st_size  > 0 \
                and chunk_title_path.exists() and chunk_title_path.stat().st_size > 0:
            continue

        t_chunk = time.time()
        df = pl.from_arrow(batch)

        desc_col   = pl.col("职责描述")
        title_col  = pl.col("工作名称")

        desc_missing  = desc_col.is_null() | desc_col.str.strip_chars().is_in(list(PLACEHOLDERS))
        title_missing = title_col.is_null() | title_col.str.strip_chars().is_in(list(PLACEHOLDERS))

        df_full = df.filter(~desc_missing & ~title_missing)
        df_to   = df.filter( desc_missing & ~title_missing)
        n_skip  = len(df) - len(df_full) - len(df_to)

        if len(df_full) > 0:
            df_full = df_full.with_columns(
                (pl.col("工作名称") + " " + pl.col("工作名称") + " " + pl.col("职责描述"))
                .alias("_text")
            )
            texts_full = df_full["_text"].to_list()
            preds_f, probs_f = infer_texts(
                texts_full, model, tokenizer, label_map,
                batch_size=args.batch_full,
                device=device,
                tok_workers=args.tok_workers,
                report_every=30,
            )
            out_full = df_full.select("表ID").with_columns([
                pl.Series("soc_code_pred", preds_f, dtype=pl.Utf8),
                pl.Series("soc_prob",      probs_f, dtype=pl.Float32),
            ])
        else:
            out_full = pl.DataFrame(
                {"表ID": [], "soc_code_pred": [], "soc_prob": []},
                schema={"表ID": pl.Int64, "soc_code_pred": pl.Utf8, "soc_prob": pl.Float32},
            )

        if len(df_to) > 0:
            titles_nfc = [nfc(t) if t is not None else t
                          for t in df_to["工作名称"].to_list()]
            out_to = df_to.select("表ID").with_columns(
                pl.Series("工作名称", titles_nfc, dtype=pl.Utf8)
            )
        else:
            out_to = pl.DataFrame(
                {"表ID": [], "工作名称": []},
                schema={"表ID": pl.Int64, "工作名称": pl.Utf8},
            )

        # Atomic writes — safe if spot instance is preempted mid-write
        _write_atomic(out_full, chunk_full_path)
        _write_atomic(out_to,   chunk_title_path)

        # Save checkpoint immediately after both files are safely written
        done_set.add(chunk_idx)
        my_ckpt["done"] = sorted(done_set)
        my_ckpt.setdefault("rows_full",     {})[str(chunk_idx)] = len(out_full)
        my_ckpt.setdefault("rows_titleonly",{})[str(chunk_idx)] = len(out_to)
        my_ckpt.setdefault("rows_skipped",  {})[str(chunk_idx)] = n_skip
        save_checkpoint(my_ckpt, gpu_ckpt)

        chunk_sec  = time.time() - t_chunk
        phase_sec  = time.time() - t_phase
        done_total = len(done_set)
        rate_chunk = len(df) / chunk_sec if chunk_sec > 0 else 0
        eta_hr     = (my_n_chunks - done_total) * (phase_sec / done_total) / 3600 \
                     if done_total > 0 else 0
        print(
            f"\n  [GPU{args.gpu_id}] Chunk {chunk_idx:05d}  "
            f"full={len(out_full):>8,}  title_only={len(out_to):>8,}  "
            f"skip={n_skip:>6,}  "
            f"{chunk_sec:.0f}s  {rate_chunk:,.0f} rows/sec  "
            f"ETA {eta_hr:.1f} hr",
            flush=True,
        )

    print(f"\n  [GPU{args.gpu_id}] Phase B complete ✓  "
          f"({len(done_set)}/{my_n_chunks} chunks, {time.time()-t_phase:.0f}s total)")


# ══════════════════════════════════════════════════════════════════════════════
# Combine
# ══════════════════════════════════════════════════════════════════════════════

def combine_and_finalize(ckpt):
    import duckdb

    banner("COMBINE — DuckDB assembly of final sidecar parquet")

    if ckpt.get("combine_done") and FINAL_OUTPUT.exists():
        print("  Combine already complete — skipping.")
        return

    if not TITLE_LOOKUP.exists():
        print("  ERROR: Phase A lookup not found. Run --phase a first.")
        sys.exit(1)

    total_chunks = ckpt.get("total_chunks", 0)
    done_count   = len(ckpt.get("done", []))
    if done_count < total_chunks:
        print(f"  ERROR: Phase B incomplete ({done_count}/{total_chunks} chunks done).")
        sys.exit(1)

    chunk_glob     = (CHUNK_DIR / "chunk_*.parquet").as_posix()
    titleonly_glob = (CHUNK_DIR / "titleonly_*.parquet").as_posix()
    lookup_path    = TITLE_LOOKUP.as_posix()
    final_path     = FINAL_OUTPUT.as_posix()

    print(f"  Chunks     : {chunk_glob}")
    print(f"  Title-only : {titleonly_glob}")
    print(f"  Lookup     : {lookup_path}")
    print(f"  Output     : {final_path}")

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA memory_limit='80GB'")
    con.execute("PRAGMA threads=16")

    t0 = time.time()
    print("\n  Running DuckDB COPY TO (UNION ALL) ...")
    con.execute(f"""
        COPY (
            SELECT 表ID, soc_code_pred, soc_prob
            FROM read_parquet('{chunk_glob}')
            UNION ALL
            SELECT r.表ID, l.soc_code_pred, l.soc_prob
            FROM read_parquet('{titleonly_glob}') r
            JOIN read_parquet('{lookup_path}')    l
              ON r.工作名称 = l.工作名称
        )
        TO '{final_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    elapsed = time.time() - t0
    mb = FINAL_OUTPUT.stat().st_size / 1e6
    print(f"  Written in {elapsed:.0f}s  ({mb/1024:.1f} GB)")

    n_final = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{final_path}')"
    ).fetchone()[0]
    print(f"  Final row count: {n_final:,}")

    con.execute(f"""
        CREATE OR REPLACE VIEW 招聘_soc_all_labeled AS
        SELECT * FROM read_parquet('{FINAL_OUTPUT.as_posix()}')
    """)
    print("  View '招聘_soc_all_labeled' registered in enterprise.duckdb")

    print("\n  Top-10 SOC codes:")
    con.sql("""
        SELECT soc_code_pred, COUNT(*) AS n, ROUND(AVG(soc_prob)*100,2) AS avg_conf_pct
        FROM 招聘_soc_all_labeled
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """).show()

    con.close()

    ckpt["combine_done"] = True
    save_checkpoint(ckpt, CHECKPOINT)
    print(f"\n  Combine complete ✓  Output: {FINAL_OUTPUT}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI & main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-scale SOC code inference on 招聘 (641M rows) — Linux/Aliyun"
    )
    p.add_argument("--phase",       choices=["a","b","combine","all"], default="all")
    p.add_argument("--gpu_id",      type=int, default=0,
                   help="GPU index (0-3); determines checkpoint_gpu{N}.json (default 0)")
    p.add_argument("--start_chunk", type=int, default=0,
                   help="First chunk index to process, inclusive (default 0)")
    p.add_argument("--end_chunk",   type=int, default=None,
                   help="Last chunk index, exclusive (default: all remaining)")
    p.add_argument("--batch_full",  type=int, default=DEFAULT_BATCH_FULL)
    p.add_argument("--batch_title", type=int, default=DEFAULT_BATCH_TITLE)
    p.add_argument("--tok_workers", type=int, default=8,
                   help="Tokenizer threads per GPU process (default 8; A10 has 32 vCPU/GPU)")
    p.add_argument("--chunk_rows",  type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--max_len",     type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--reset",       action="store_true",
                   help="Delete this GPU's checkpoint and restart its chunk range")
    return p.parse_args()


def main():
    args = parse_args()

    banner("10_full_inference_linux.py — Full-scale SOC labeling (Aliyun)")
    print(f"  Phase       : {args.phase}")
    print(f"  gpu_id      : {args.gpu_id}")
    print(f"  start_chunk : {args.start_chunk}")
    print(f"  end_chunk   : {args.end_chunk if args.end_chunk is not None else 'all'}")
    print(f"  batch_full  : {args.batch_full}")
    print(f"  batch_title : {args.batch_title}")
    print(f"  tok_workers : {args.tok_workers}")
    print(f"  chunk_rows  : {args.chunk_rows:,}")
    print(f"  max_len     : {args.max_len}")
    print(f"  Started     : {now_str()}")

    if args.reset:
        gpu_ckpt = checkpoint_path(args.gpu_id)
        for f in [gpu_ckpt, gpu_ckpt.with_suffix(".json.tmp")]:
            if f.exists():
                f.unlink()
                print(f"  Removed: {f}")
        print(f"  Checkpoint reset for GPU {args.gpu_id}. Chunk files are kept.")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Shared checkpoint (Phase A / combine state only)
    ckpt = load_checkpoint(CHECKPOINT)
    if not ckpt:
        ckpt = {}
        save_checkpoint(ckpt, CHECKPOINT)

    if args.phase in ("a", "b", "all"):
        label_map = build_label_map()
        model, tokenizer, device = load_model_and_tokenizer()
    else:
        label_map = model = tokenizer = device = None

    if args.phase in ("a", "all"):
        phase_a(model, tokenizer, device, label_map, ckpt, args)
        ckpt = load_checkpoint(CHECKPOINT)

    if args.phase in ("b", "all"):
        phase_b(model, tokenizer, device, label_map, ckpt, args)

    if args.phase in ("combine", "all"):
        ckpt = load_checkpoint(CHECKPOINT)
        combine_and_finalize(ckpt)

    banner("DONE")
    print(f"  Finished: {now_str()}")
    print(f"  Output  : {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
