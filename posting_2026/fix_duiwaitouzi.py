# -*- coding: utf-8 -*-
"""
fix_duiwaitouzi.py
------------------
Fixes swapped columns in 对外投资.parquet:
  认缴投资金额  ↔  认缴投资时间   (confirmed: dates in 金额, amounts in 时间)

Also checks whether 实缴投资金额 / 实缴投资时间 are similarly swapped.
Writes corrected file back to the same location (overwrites in-place after backup).
"""

import sys, shutil, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import polars as pl

SRC  = Path(r"I:\posting_2026\parquet\对外投资.parquet")
BAK  = Path(r"I:\posting_2026\parquet\对外投资.parquet.bak")

print("=" * 60)
print("Fix: 对外投资 — swapped column repair")
print("=" * 60)

# ── Step 1: Load ──────────────────────────────────────────────────────────────
print(f"\n[1] Loading {SRC} ...")
t0 = time.time()
df = pl.read_parquet(SRC)
print(f"    Shape: {df.shape}  ({time.time()-t0:.1f}s)")
print(f"    Columns: {df.columns}")

# ── Step 2: Verify the swap before touching anything ─────────────────────────
print("\n[2] Verifying column content ...")

sample = df.filter(pl.col("认缴投资金额").is_not_null()).head(5)
print("    认缴投资金额 sample:", sample["认缴投资金额"].to_list())
print("    认缴投资时间 sample:", sample["认缴投资时间"].to_list())

# Check 实缴 columns if they exist
has_shijiao = "实缴投资金额" in df.columns and "实缴投资时间" in df.columns
if has_shijiao:
    sample2 = df.filter(pl.col("实缴投资金额").is_not_null()).head(5)
    print("    实缴投资金额 sample:", sample2["实缴投资金额"].to_list())
    print("    实缴投资时间 sample:", sample2["实缴投资时间"].to_list())

# ── Step 3: Backup ───────────────────────────────────────────────────────────
print(f"\n[3] Backing up to {BAK} ...")
shutil.copy2(SRC, BAK)
print(f"    Backup done ({BAK.stat().st_size/1e6:.1f} MB)")

# ── Step 4: Rename to fix the swap ───────────────────────────────────────────
print("\n[4] Swapping 认缴投资金额 ↔ 认缴投资时间 ...")
df = df.rename({
    "认缴投资金额": "认缴投资时间",
    "认缴投资时间": "认缴投资金额",
})

# Swap 实缴 if also swapped (check if amounts appear in 金额 after rename)
if has_shijiao:
    sample3 = df.filter(pl.col("实缴投资金额").is_not_null()).head(3)
    sj_vals = sample3["实缴投资金额"].to_list()
    # If first non-null value looks like a date (contains '-'), swap it too
    date_like = any(
        isinstance(v, str) and len(v) == 10 and v[4] == '-'
        for v in sj_vals if v and v != '--'
    )
    if date_like:
        print("    实缴 columns also swapped — fixing ...")
        df = df.rename({
            "实缴投资金额": "实缴投资时间",
            "实缴投资时间": "实缴投资金额",
        })
    else:
        print("    实缴 columns look correct — no change needed.")

# ── Step 5: Verify after fix ─────────────────────────────────────────────────
print("\n[5] Verifying after fix ...")
sample_fixed = df.filter(pl.col("认缴投资金额").is_not_null()).head(5)
print("    认缴投资金额 (should be amounts now):", sample_fixed["认缴投资金额"].to_list())
print("    认缴投资时间 (should be dates now):  ", sample_fixed["认缴投资时间"].to_list())

# ── Step 6: Write back ───────────────────────────────────────────────────────
print(f"\n[6] Writing fixed file back to {SRC} ...")
t0 = time.time()
df.write_parquet(SRC, compression="zstd")
print(f"    Done ({time.time()-t0:.1f}s)  size={SRC.stat().st_size/1e6:.1f} MB")

print("\n✅ Fix complete!")
print(f"   Backup kept at: {BAK}")
print("   Delete backup once you are satisfied: BAK.unlink()")
