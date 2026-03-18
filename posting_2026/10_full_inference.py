# -*- coding: utf-8 -*-
"""
10_full_inference.py — Full-scale SOC code inference on all 641M 招聘 rows
=========================================================================
Two-phase pipeline:

  Phase A  (title-only deduplication)
    DuckDB DISTINCT query on 工作名称 for rows with missing/placeholder
    职责描述 → infer once per unique title → save lookup parquet.

  Phase B  (single sequential pass through all 641M rows)
    PyArrow iter_batches(2M rows) — pure sequential I/O, no OFFSET penalty.
    Per chunk:
      • full rows  (title + desc)    → GPU inference → chunk_NNNNN.parquet
      • title-only (no/placeholder desc) → capture [表ID, 工作名称] → titleonly_NNNNN.parquet
      • neither (both missing)       → silently skipped

  Combine
    DuckDB COPY TO: UNION ALL chunk_*.parquet with (titleonly_*.parquet ⋈ lookup)
    → 招聘_soc_all_labels.parquet   [表ID, soc_code_pred, soc_prob]
    → DuckDB view 招聘_soc_all_labeled

Output is a SIDECAR — source parquet is never modified.
Join back at query time:
    SELECT r.*, l.soc_code_pred, l.soc_prob
    FROM 招聘_clean r JOIN 招聘_soc_all_labeled l USING (表ID)

Usage
-----
    python 10_full_inference.py                         # full run A→B→combine
    python 10_full_inference.py --phase a               # Phase A only
    python 10_full_inference.py --phase b               # Phase B only
    python 10_full_inference.py --phase combine         # combine only
    python 10_full_inference.py --reset                 # wipe checkpoint, restart
    python 10_full_inference.py --batch_full 128 --batch_title 256
    python 10_full_inference.py --chunk_rows 2000000
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
PARQUET_SRC   = Path(r"I:\posting_2026\parquet\招聘\data.parquet")
DB_PATH       = Path(r"I:\posting_2026\enterprise.duckdb")
TOKENIZER_DIR = Path(r"D:\Dropbox\Dropbox\vs_cloud\HPC\chinese-bert-wwm")
MODEL_DIR     = Path(r"G:\Data\job_posting\processed\model")
LABEL_SRC     = Path(r"G:\Data\job_posting\processed\finetune\est_sample.csv")
ONNX_PATH     = MODEL_DIR / "bert_soc_406.onnx"
OUTPUT_BASE   = Path(r"I:\posting_2026\parquet")
CHUNK_DIR     = OUTPUT_BASE / "招聘_soc_full"
TITLE_LOOKUP  = OUTPUT_BASE / "招聘_soc_title_lookup.parquet"
FINAL_OUTPUT  = OUTPUT_BASE / "招聘_soc_all_labels.parquet"
CHECKPOINT    = CHUNK_DIR / "checkpoint.json"

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_BATCH_FULL  = 256   # increased: full-desc rows, 512-token max, 16GB VRAM fp16
DEFAULT_BATCH_TITLE = 1024  # increased: title-only rows are short (~40 tok), CPU was bottleneck
DEFAULT_CHUNK_ROWS  = 2_000_000
DEFAULT_MAX_LEN     = 512

# Placeholder values for 职责描述 (after stripping whitespace)
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


# ── Atomic checkpoint I/O ─────────────────────────────────────────────────────

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
    os.replace(tmp, path)  # atomic on NTFS same-volume


def init_checkpoint(chunk_rows: int, total_chunks: int) -> dict:
    return {
        "chunk_rows":    chunk_rows,
        "total_chunks":  total_chunks,
        "phase_a_done":  False,
        "done":          [],
        "rows_full":     {},
        "rows_titleonly":{},
        "rows_skipped":  {},
        "combine_done":  False,
        "created_at":    now_str(),
    }


# ── Progress tracker ──────────────────────────────────────────────────────────

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

    print(f"  Loading tokenizer from {TOKENIZER_DIR.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    print(f"  Loading model from {MODEL_DIR.name} ...")
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
# Core GPU inference loop (shared by Phase A and B)
# ══════════════════════════════════════════════════════════════════════════════

def infer_texts(
    texts: list,
    model,
    tokenizer,
    label_map: dict,
    batch_size: int,
    device: str,
    max_len: int = DEFAULT_MAX_LEN,
    progress: Progress = None,
    report_every: int = 30,    # seconds between progress prints
    tok_workers: int = 3,      # parallel CPU tokenizer threads
) -> tuple[list, list]:
    """
    Batched GPU inference with multi-threaded tokenization pipeline.

    tok_workers Rust tokenizer threads run in parallel (GIL is released by the
    fast tokenizer), keeping a sliding window of pre-tokenised batches ready so
    the GPU never waits.  Window size = tok_workers * 2 batches ahead.
    Returns (preds: list[str], probs: list[float]).
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    n           = len(texts)
    t0          = time.time()
    last_report = t0
    window      = tok_workers * 2          # batches pre-tokenised ahead of GPU

    def _tok(start: int):
        return tokenizer(
            texts[start: start + batch_size],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

    preds, probs = [], []
    starts = list(range(0, n, batch_size))

    with ThreadPoolExecutor(max_workers=tok_workers) as ex:
        # Pre-fill the sliding window before entering the main loop
        pending = deque(ex.submit(_tok, starts[i])
                        for i in range(min(window, len(starts))))

        for batch_idx, start in enumerate(starts):
            enc = pending.popleft().result()               # oldest future → ready
            batch_len = enc["input_ids"].shape[0]

            # Keep the window full: submit the batch that is `window` steps ahead
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
            preds.extend([label_map[idx] for idx in logits.argmax(dim=-1).cpu().tolist()])
            probs.extend(p.max(dim=-1).values.cpu().tolist())

            done = start + batch_len
            now  = time.time()
            if now - last_report >= report_every:   # report_every = seconds
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

    # ── Step A1: Extract unique titles from title-only rows ───────────────────
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

    # ── Step A2: NFC normalisation ────────────────────────────────────────────
    titles_raw = df_titles["工作名称"].to_list()
    titles_nfc = [nfc(t) for t in titles_raw]
    texts      = [f"{t} {t}" for t in titles_nfc]   # title × 2, no description

    # ── Step A3: GPU inference ────────────────────────────────────────────────
    print(f"\n  Inference: {n_unique:,} unique titles | batch={args.batch_title}")
    prog = Progress(n_unique, "Phase A", interval_sec=30)
    preds, probs = infer_texts(
        texts, model, tokenizer, label_map,
        batch_size=args.batch_title,
        device=device,
        progress=prog,
        report_every=30,
        tok_workers=args.tok_workers,
    )

    # ── Step A4: Save lookup ──────────────────────────────────────────────────
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
    """Write parquet to .tmp then rename — avoids corrupt partial files."""
    tmp = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def phase_b(model, tokenizer, device, label_map, ckpt, args):
    import pyarrow.parquet as pq
    import polars as pl

    banner("PHASE B — Chunked sequential inference (all 641M rows)")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # ── Validate / initialise chunk parameters ────────────────────────────────
    pf         = pq.ParquetFile(str(PARQUET_SRC))
    total_rows = pf.metadata.num_rows
    rg0_rows   = pf.metadata.row_group(0).num_rows
    total_chunks = math.ceil(total_rows / args.chunk_rows)
    print(f"  Source: {PARQUET_SRC}")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Row group 0: {rg0_rows:,} rows")
    print(f"  Chunk size : {args.chunk_rows:,} rows  →  {total_chunks} chunks")
    print(f"  Batch sizes: full={args.batch_full}  title={args.batch_title}")

    # Guard against chunk_rows mismatch on resume
    if "chunk_rows" in ckpt and ckpt["chunk_rows"] != args.chunk_rows:
        print(
            f"\n  ERROR: checkpoint chunk_rows={ckpt['chunk_rows']} "
            f"!= current --chunk_rows={args.chunk_rows}.\n"
            f"  Use --reset to discard the checkpoint and restart."
        )
        sys.exit(1)

    if "total_chunks" not in ckpt:
        ckpt["chunk_rows"]   = args.chunk_rows
        ckpt["total_chunks"] = total_chunks
        save_checkpoint(ckpt, CHECKPOINT)

    done_set = set(ckpt.get("done", []))
    rows_done_total = sum(ckpt.get("rows_full", {}).values()) \
                    + sum(ckpt.get("rows_titleonly", {}).values())

    # Estimate remaining GPU time
    full_share = 0.58   # ~58% of rows have descriptions
    est_gpu_hr = (total_rows * full_share - sum(ckpt.get("rows_full", {}).values())) \
                 / (args.batch_full * 10) / 3600   # rough: 10 batches/sec
    print(f"\n  Chunks done so far: {len(done_set)}/{total_chunks}")
    print(f"  Rough GPU time remaining: {est_gpu_hr:.0f} hr  "
          f"(varies with actual description density)")

    COLS = ["表ID", "工作名称", "职责描述"]
    global_prog = Progress(total_rows, "Phase B total", interval_sec=30)
    global_prog.done = rows_done_total   # account for already-done chunks

    t_phase = time.time()

    for chunk_idx, batch in enumerate(pf.iter_batches(batch_size=args.chunk_rows,
                                                       columns=COLS)):
        # ── Resume: skip completed chunks ─────────────────────────────────────
        chunk_full_path  = CHUNK_DIR / f"chunk_{chunk_idx:05d}.parquet"
        chunk_title_path = CHUNK_DIR / f"titleonly_{chunk_idx:05d}.parquet"

        if chunk_idx in done_set \
                and chunk_full_path.exists()  and chunk_full_path.stat().st_size  > 0 \
                and chunk_title_path.exists() and chunk_title_path.stat().st_size > 0:
            skip_n = len(batch)
            global_prog.update(skip_n)
            continue

        t_chunk = time.time()
        df = pl.from_arrow(batch)

        # ── Split into full / title-only / skip ───────────────────────────────
        desc_col   = pl.col("职责描述")
        title_col  = pl.col("工作名称")

        desc_missing  = desc_col.is_null() | desc_col.str.strip_chars().is_in(list(PLACEHOLDERS))
        title_missing = title_col.is_null() | title_col.str.strip_chars().is_in(list(PLACEHOLDERS))

        df_full = df.filter(~desc_missing & ~title_missing)
        df_to   = df.filter( desc_missing & ~title_missing)
        n_skip  = len(df) - len(df_full) - len(df_to)

        # ── Full rows: GPU inference ───────────────────────────────────────────
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
                report_every=30,
                tok_workers=args.tok_workers,
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

        # ── Title-only rows: save metadata for combine step ───────────────────
        if len(df_to) > 0:
            # NFC-normalize the join key
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

        # ── Write chunk files (atomic) ─────────────────────────────────────────
        _write_atomic(out_full, chunk_full_path)
        _write_atomic(out_to,   chunk_title_path)

        # ── Update checkpoint ─────────────────────────────────────────────────
        done_set.add(chunk_idx)
        ckpt["done"] = sorted(done_set)
        ckpt.setdefault("rows_full",     {})[str(chunk_idx)] = len(out_full)
        ckpt.setdefault("rows_titleonly",{})[str(chunk_idx)] = len(out_to)
        ckpt.setdefault("rows_skipped",  {})[str(chunk_idx)] = n_skip
        save_checkpoint(ckpt, CHECKPOINT)

        # ── Chunk summary line (always printed) ───────────────────────────────
        chunk_sec  = time.time() - t_chunk
        phase_sec  = time.time() - t_phase
        done_total = len(done_set)
        rate_chunk = len(df) / chunk_sec if chunk_sec > 0 else 0
        eta_hr     = (total_chunks - done_total) * (phase_sec / done_total) / 3600 \
                     if done_total > 0 else 0
        print(
            f"\n  Chunk {chunk_idx:05d}/{total_chunks-1}  "
            f"full={len(out_full):>8,}  title_only={len(out_to):>8,}  "
            f"skip={n_skip:>6,}  "
            f"{chunk_sec:.0f}s  {rate_chunk:,.0f} rows/sec  "
            f"phase ETA {eta_hr:.1f} hr",
            flush=True,
        )
        global_prog.update(len(df), force=False)

    print(f"\n  Phase B complete ✓  "
          f"({len(done_set)} chunks, {time.time()-t_phase:.0f}s total)")


# ══════════════════════════════════════════════════════════════════════════════
# Combine — final assembly via DuckDB
# ══════════════════════════════════════════════════════════════════════════════

def combine_and_finalize(ckpt):
    import duckdb

    banner("COMBINE — DuckDB assembly of final sidecar parquet")

    if ckpt.get("combine_done") and FINAL_OUTPUT.exists():
        print("  Combine already complete — skipping.")
        return

    # ── Verify prerequisites ──────────────────────────────────────────────────
    if not TITLE_LOOKUP.exists():
        print("  ERROR: Phase A lookup not found. Run --phase a first.")
        sys.exit(1)

    total_chunks = ckpt.get("total_chunks", 0)
    done_count   = len(ckpt.get("done", []))
    if done_count < total_chunks:
        print(f"  ERROR: Phase B incomplete ({done_count}/{total_chunks} chunks done).")
        sys.exit(1)

    chunk_glob    = (CHUNK_DIR / "chunk_*.parquet").as_posix()
    titleonly_glob= (CHUNK_DIR / "titleonly_*.parquet").as_posix()
    lookup_path   = TITLE_LOOKUP.as_posix()
    final_path    = FINAL_OUTPUT.as_posix()

    print(f"  Chunks       : {chunk_glob}")
    print(f"  Title-only   : {titleonly_glob}")
    print(f"  Lookup       : {lookup_path}")
    print(f"  Output       : {final_path}")

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

    # ── Row count sanity check ────────────────────────────────────────────────
    n_final = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{final_path}')"
    ).fetchone()[0]
    print(f"  Final row count: {n_final:,}")

    # ── Register DuckDB view ──────────────────────────────────────────────────
    con.execute(f"""
        CREATE OR REPLACE VIEW 招聘_soc_all_labeled AS
        SELECT * FROM read_parquet('{FINAL_OUTPUT.as_posix()}')
    """)
    print("  View '招聘_soc_all_labeled' registered in enterprise.duckdb")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n  Top-10 SOC codes in final output:")
    con.sql("""
        SELECT soc_code_pred,
               COUNT(*)                      AS n,
               ROUND(AVG(soc_prob)*100, 2)   AS avg_conf_pct
        FROM 招聘_soc_all_labeled
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """).show()

    print("\n  Confidence distribution:")
    con.sql("""
        SELECT
            CASE
                WHEN soc_prob < 0.50 THEN '< 50%'
                WHEN soc_prob < 0.70 THEN '50-70%'
                WHEN soc_prob < 0.80 THEN '70-80%'
                WHEN soc_prob < 0.90 THEN '80-90%'
                ELSE '≥ 90%'
            END                              AS band,
            COUNT(*)                         AS n,
            ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
        FROM 招聘_soc_all_labeled
        GROUP BY 1 ORDER BY MIN(soc_prob)
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
        description="Full-scale SOC code inference on 招聘 (641M rows)"
    )
    p.add_argument("--phase",       choices=["a","b","combine","all"], default="all")
    p.add_argument("--batch_full",  type=int, default=DEFAULT_BATCH_FULL,
                   help=f"GPU batch size for full-desc rows (default {DEFAULT_BATCH_FULL})")
    p.add_argument("--batch_title", type=int, default=DEFAULT_BATCH_TITLE,
                   help=f"GPU batch size for title-only rows (default {DEFAULT_BATCH_TITLE})")
    p.add_argument("--tok_workers", type=int, default=3,
                   help="Parallel CPU tokenizer threads (default 3; try 4-6 on A100)")
    p.add_argument("--chunk_rows",  type=int, default=DEFAULT_CHUNK_ROWS,
                   help=f"Rows per Phase B iteration chunk (default {DEFAULT_CHUNK_ROWS:,})")
    p.add_argument("--max_len",     type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--reset",       action="store_true",
                   help="Delete checkpoint and restart from scratch")
    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    banner("10_full_inference.py — Full-scale SOC labeling")
    print(f"  Phase      : {args.phase}")
    print(f"  batch_full : {args.batch_full}")
    print(f"  batch_title: {args.batch_title}")
    print(f"  chunk_rows : {args.chunk_rows:,}")
    print(f"  max_len    : {args.max_len}")
    print(f"  Started    : {now_str()}")

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        for f in [CHECKPOINT, CHECKPOINT.with_suffix(".json.tmp"),
                  TITLE_LOOKUP, FINAL_OUTPUT]:
            if f.exists():
                f.unlink()
                print(f"  Removed: {f}")
        if CHUNK_DIR.exists():
            import shutil
            shutil.rmtree(CHUNK_DIR)
            print(f"  Removed: {CHUNK_DIR}")
        print("  Checkpoint reset complete.")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(CHECKPOINT)
    if not ckpt:
        # Will be properly initialised in phase_b when total_chunks is known
        ckpt = {}
        save_checkpoint(ckpt, CHECKPOINT)

    # ── Load model (only needed for A and B) ──────────────────────────────────
    if args.phase in ("a", "b", "all"):
        label_map = build_label_map()
        model, tokenizer, device = load_model_and_tokenizer()
    else:
        label_map = model = tokenizer = device = None

    # ── Run phases ────────────────────────────────────────────────────────────
    if args.phase in ("a", "all"):
        phase_a(model, tokenizer, device, label_map, ckpt, args)
        ckpt = load_checkpoint(CHECKPOINT)   # reload after save

    if args.phase in ("b", "all"):
        phase_b(model, tokenizer, device, label_map, ckpt, args)
        ckpt = load_checkpoint(CHECKPOINT)

    if args.phase in ("combine", "all"):
        # Check both phases done before combining
        if args.phase == "combine":
            ckpt = load_checkpoint(CHECKPOINT)
        if not ckpt.get("phase_a_done") and args.phase != "combine":
            print("  Skipping combine: Phase A not done.")
        elif args.phase == "combine" or (ckpt.get("phase_a_done")
                                          and len(ckpt.get("done",[])) == ckpt.get("total_chunks",0)):
            combine_and_finalize(ckpt)
        else:
            remaining = ckpt.get("total_chunks",0) - len(ckpt.get("done",[]))
            print(f"  Combine not run: Phase B still has {remaining} chunks remaining.")

    banner("DONE")
    print(f"  Finished: {now_str()}")
    print(f"  Output  : {FINAL_OUTPUT}")
    print(f"  DuckDB view: 招聘_soc_all_labeled")
    print("""
  Query example:
    SELECT r.工作名称, r.公司ID, r.发布日期, l.soc_code_pred, l.soc_prob
    FROM 招聘_clean r
    JOIN 招聘_soc_all_labeled l USING (表ID)
    LIMIT 10;
""")


if __name__ == "__main__":
    main()
