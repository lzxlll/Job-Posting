# -*- coding: utf-8 -*-
"""
08_bert_inference.py  —  SOC code inference on 招聘_clean
==========================================================
Uses the already-fine-tuned BertForSequenceClassification model (406 SOC-code
classes) to label a sample of job postings from 招聘_clean.

Input text format  (matches firm_data_programming.ipynb training):
    工作名称 + ' ' + 工作名称 + ' ' + 职责描述   (title × 2 + description)

Output Parquet columns:
    表ID, 公司ID, 发布日期, 工作名称, soc_code_pred, soc_prob

Output also registered as view 招聘_labeled in enterprise.duckdb.

Usage
-----
    python 08_bert_inference.py              # default: 1M rows, local test
    python 08_bert_inference.py --n 5000000  # 5M rows
    python 08_bert_inference.py --batch 128  # tune batch size for your GPU
"""

import sys
import json
import time
import argparse
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
TOKENIZER_DIR = Path(r"D:\Dropbox\Dropbox\vs_cloud\HPC\chinese-bert-wwm")
MODEL_DIR     = Path(r"G:\Data\job_posting\processed\model")
LABEL_SRC     = Path(r"G:\Data\job_posting\processed\finetune\est_sample.csv")
OUTPUT_DIR    = Path(r"I:\posting_2026\parquet")
DB_PATH       = Path(r"I:\posting_2026\enterprise.duckdb")

# ── Defaults (overridable via CLI) ─────────────────────────────────────────────
DEFAULT_N_SAMPLE   = 1_000_000   # local test; use 5M+ for production
DEFAULT_BATCH_SIZE = 64          # RTX 5000 16 GB safe default
#   Cloud options:  ecs.gn7i-c16g1.4xlarge (A10 24GB) → batch=512
#                   GCP a2-highgpu-1g       (A100 40GB) → batch=1200
DEFAULT_MAX_LEN    = 512
DEFAULT_SEED       = 42

# ── Data profile notes (from 招聘, 2026-03-16) ─────────────────────────────────
# Total rows: 641,831,221
# 兼职 share: 0.003%  → negligible
# Duplicate rate (same 公司ID + 工作名称 + month): ~2.38% → no dedup needed
# Top-10 数据来源 combined: 65.05%
# Note: "boss" / "boss直聘" / "BOSS直聘" are the same platform (3 spellings)


def parse_args():
    p = argparse.ArgumentParser(description="SOC code inference on 招聘_clean")
    p.add_argument("--n",     type=int, default=DEFAULT_N_SAMPLE,   help="rows to sample")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="GPU batch size")
    p.add_argument("--seed",  type=int, default=DEFAULT_SEED)
    return p.parse_args()


# ── Section 1: Reconstruct label map ──────────────────────────────────────────
def build_label_map() -> dict:
    """
    Reconstruct {int → soc_code} mapping from est_sample.csv.
    Exact logic from firm_data_programming.ipynb.
    """
    print("  Building label map from est_sample.csv ...")
    import pandas as pd
    df_map = pd.read_csv(
        LABEL_SRC,
        encoding="utf_8_sig",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    df_map["soc_code"] = df_map["soc_code"].str.replace("-", "", regex=False)
    unique_soc_codes      = sorted(df_map["soc_code"].unique())
    soc_code_dict         = {code: i for i, code in enumerate(unique_soc_codes)}
    inverse_soc_code_dict = {v: k for k, v in soc_code_dict.items()}
    print(f"  Label map: {len(inverse_soc_code_dict)} classes  "
          f"(range: {unique_soc_codes[0]} … {unique_soc_codes[-1]})")
    return inverse_soc_code_dict


# ── Section 2: Sample rows from 招聘_clean ─────────────────────────────────────
def load_sample(n: int, seed: int):
    """Draw n rows from 招聘 via data_utils.sample()."""
    sys.path.insert(0, str(Path(__file__).parent))
    from data_utils import sample
    import polars as pl

    print(f"\n  Sampling {n:,} rows from 招聘 (seed={seed}) ...")
    df = sample(
        "招聘",
        n=n,
        cols=["表ID", "公司ID", "工作名称", "职责描述", "发布日期"],
        seed=seed,
    )

    # Filter nulls / placeholder descriptions
    before = len(df)
    df = df.filter(
        pl.col("职责描述").is_not_null()
        & (pl.col("职责描述") != "--")
        & pl.col("工作名称").is_not_null()
    )
    print(f"  After null filter: {len(df):,} rows  (dropped {before - len(df):,})")

    # Build input text: title × 2 + description  (matches training format)
    df = df.with_columns(
        (pl.col("工作名称") + " " + pl.col("工作名称") + " " + pl.col("职责描述"))
        .alias("input_text")
    )
    return df


# ── Section 3: Batch inference ────────────────────────────────────────────────
def run_inference(df, label_map: dict, batch_size: int, max_len: int):
    """GPU fp16 inference; returns (preds, probs) lists."""
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification

    print(f"\n  Loading tokenizer from {TOKENIZER_DIR.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    print(f"  Loading model from {MODEL_DIR.name} ...")
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available — running on CPU (much slower)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    model = model.to(device)

    texts  = df["input_text"].to_list()
    preds  = []
    probs  = []
    n      = len(texts)
    t0     = time.time()

    print(f"  Inference: {n:,} rows | batch={batch_size} | max_len={max_len} | {device}")

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            if device == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

            p = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            probs.extend(p.max(dim=-1).values.cpu().tolist())

            # Progress every 200 batches
            done = i + len(batch)
            if (i // batch_size) % 200 == 0:
                elapsed  = time.time() - t0
                rate     = done / elapsed if elapsed > 0 else 0
                eta_min  = (n - done) / rate / 60 if rate > 0 else 0
                pct      = 100 * done / n
                print(f"  {pct:5.1f}%  {done:>9,}/{n:,}  "
                      f"{rate:,.0f} rows/sec  ETA {eta_min:.0f} min")

    elapsed = time.time() - t0
    print(f"  Done: {n:,} rows in {elapsed/60:.1f} min  "
          f"({n/elapsed:,.0f} rows/sec avg)")

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak_gb:.2f} GB")

    return preds, probs


# ── Section 4: Save output ────────────────────────────────────────────────────
def save_output(df, preds: list, probs: list, label_map: dict):
    """Write Parquet and register DuckDB view."""
    import polars as pl
    import duckdb

    # Decode labels
    decoded = [label_map[p] for p in preds]

    df = df.with_columns([
        pl.Series("soc_code_pred", decoded),
        pl.Series("soc_prob", probs).cast(pl.Float32),
    ]).drop("input_text")

    # Keep output columns matching prior notebook style
    df = df.select(["表ID", "公司ID", "发布日期", "工作名称", "soc_code_pred", "soc_prob"])

    # Write Parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "招聘_soc_labels.parquet"
    df.write_parquet(out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Parquet → {out_path}  ({size_mb:.1f} MB, {len(df):,} rows)")

    # DuckDB view
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"""
        CREATE OR REPLACE VIEW 招聘_labeled AS
        SELECT * FROM read_parquet('{out_path.as_posix()}')
    """)
    print(f"  View '招聘_labeled' registered in enterprise.duckdb")

    # Summary
    print("\n  Top-10 predicted SOC codes:")
    con.sql("""
        SELECT soc_code_pred,
               COUNT(*)              AS n,
               ROUND(AVG(soc_prob), 3) AS avg_conf
        FROM 招聘_labeled
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 10
    """).show()

    con.close()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    print("=" * 60)
    print("08_bert_inference.py — SOC code labeling")
    print(f"  N_SAMPLE   = {args.n:,}")
    print(f"  BATCH_SIZE = {args.batch}")
    print(f"  MAX_LEN    = {DEFAULT_MAX_LEN}")
    print(f"  SEED       = {args.seed}")
    print("=" * 60)

    label_map = build_label_map()
    df        = load_sample(n=args.n, seed=args.seed)
    preds, probs = run_inference(
        df, label_map,
        batch_size=args.batch,
        max_len=DEFAULT_MAX_LEN,
    )
    out_df = save_output(df, preds, probs, label_map)

    print("\n✅  Inference complete.")
    print(f"    Rows labeled : {len(out_df):,}")
    print(f"    Output       : {OUTPUT_DIR / '招聘_soc_labels.parquet'}")


if __name__ == "__main__":
    main()
