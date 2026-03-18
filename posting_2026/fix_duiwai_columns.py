# -*- coding: utf-8 -*-
"""
fix_duiwai_columns.py
----------------------
Fixes confirmed-swapped columns in 对外投资.parquet:

  认缴投资金额  ←→  认缴投资时间
  实缴投资金额  ←→  实缴投资时间

Evidence from 04_validation_report.md:
  认缴投资金额 contained: '--', '2030-12-10', '2030-12-21'  ← dates, not money
  认缴投资时间 contained: '1.0万元人民币', '0.51万元人民币'  ← money, not dates

Steps:
  1. Backup original to  对外投资.parquet.bak
  2. Load with Polars, rename the four columns
  3. Spot-check 5 rows to confirm correctness
  4. Overwrite the Parquet file (zstd compression)
  5. Write a short log to  fix_duiwai_log.txt
"""

import sys, shutil, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import polars as pl

PARQUET  = Path(r"I:\posting_2026\parquet\对外投资.parquet")
BACKUP   = PARQUET.with_suffix(".parquet.bak")
LOG_PATH = Path(r"D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\fix_duiwai_log.txt")

log_lines = [f"fix_duiwai_columns.py — {datetime.now()}", ""]

def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("Fixing swapped columns in 对外投资.parquet")
log("=" * 60)

# ── Step 1: Backup ────────────────────────────────────────────────────────────
log(f"\n[1/4] Backing up to {BACKUP.name} ...")
t0 = time.time()
shutil.copy2(PARQUET, BACKUP)
log(f"      Done in {time.time()-t0:.1f}s  ({BACKUP.stat().st_size/1e9:.3f} GB)")

# ── Step 2: Load ──────────────────────────────────────────────────────────────
log(f"\n[2/4] Loading Parquet ...")
t0 = time.time()
df = pl.read_parquet(PARQUET)
log(f"      {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")
log(f"      Memory: ~{df.estimated_size('mb'):.0f} MB")

log("\n      Columns in file:")
for c in df.columns:
    log(f"        {c}")

# ── Step 3: Rename ────────────────────────────────────────────────────────────
log(f"\n[3/4] Swapping columns ...")

# Swap 认缴 pair
if "认缴投资金额" in df.columns and "认缴投资时间" in df.columns:
    df = df.rename({"认缴投资金额": "__tmp_renji__", "认缴投资时间": "认缴投资金额"})
    df = df.rename({"__tmp_renji__": "认缴投资时间"})
    log("      ✅ Swapped: 认缴投资金额 ↔ 认缴投资时间")
else:
    log("      ⚠️  认缴 columns not found — skipped")

# Swap 实缴 pair
if "实缴投资金额" in df.columns and "实缴投资时间" in df.columns:
    df = df.rename({"实缴投资金额": "__tmp_shiji__", "实缴投资时间": "实缴投资金额"})
    df = df.rename({"__tmp_shiji__": "实缴投资时间"})
    log("      ✅ Swapped: 实缴投资金额 ↔ 实缴投资时间")
else:
    log("      ⚠️  实缴 columns not found — skipped")

# ── Step 4: Spot-check ────────────────────────────────────────────────────────
log(f"\n      Post-fix sample (认缴 — 金额 should be money, 时间 should be date):")
sample = (
    df.filter(pl.col("认缴投资金额").is_not_null() & (pl.col("认缴投资金额") != "--"))
      .select(["认缴投资金额", "认缴投资时间"])
      .head(5)
)
for row in sample.iter_rows():
    log(f"        金额={str(row[0]):<25}  时间={row[1]}")

if "实缴投资金额" in df.columns and "实缴投资时间" in df.columns:
    log(f"\n      Post-fix sample (实缴 — 金额 should be money, 时间 should be date):")
    sample2 = (
        df.filter(pl.col("实缴投资金额").is_not_null() & (pl.col("实缴投资金额") != "--"))
          .select(["实缴投资金额", "实缴投资时间"])
          .head(5)
    )
    for row in sample2.iter_rows():
        log(f"        金额={str(row[0]):<25}  时间={row[1]}")
else:
    log(f"\n      实缴 columns not present in this table — skipping spot-check")

# ── Step 5: Write ─────────────────────────────────────────────────────────────
log(f"\n[4/4] Writing fixed Parquet ...")
t0 = time.time()
df.write_parquet(PARQUET, compression="zstd")
elapsed = time.time() - t0
new_sz  = PARQUET.stat().st_size / 1e9
log(f"      Written in {elapsed:.1f}s  ({new_sz:.3f} GB)")

log(f"\n✅ Fix complete!")
log(f"   Fixed file : {PARQUET}")
log(f"   Backup kept: {BACKUP}  ← delete when satisfied")

# Write log file
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\nLog written to: {LOG_PATH}")
