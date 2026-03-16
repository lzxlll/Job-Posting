# -*- coding: utf-8 -*-
"""
01_pilot_csv_to_parquet.py
--------------------------
Pilot script: Convert the smallest CSV in posting_2026 raw data to Parquet.
Validates:
  - File encoding detection
  - Schema inference
  - Polars streaming conversion
  - DuckDB query on resulting Parquet

Usage:
    python 01_pilot_csv_to_parquet.py
"""

import subprocess, sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Install dependencies if missing
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

for pkg in ["polars", "duckdb", "pyarrow", "chardet"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import polars as pl
import duckdb
import chardet
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Use unicode escapes to safely embed Chinese folder name
RAW_DIR     = Path("I:\\posting_2026") / "\u5b9a\u5236\u6570\u636e"   # 定制数据
PARQUET_DIR = Path(r"I:\posting_2026\parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# ── Discover all CSV files sorted by size (smallest first for pilot) ──────────
csv_files = sorted(RAW_DIR.glob("*.csv"), key=lambda f: f.stat().st_size)

print("=" * 60)
print("Dataset inventory (all files):")
print("=" * 60)
for i, f in enumerate(csv_files):
    sz = f.stat().st_size / 1e9
    print(f"  [{i:02d}] {f.name:<50} {sz:>8.3f} GB")

total_gb = sum(f.stat().st_size for f in csv_files) / 1e9
print(f"\n  Total: {len(csv_files)} files, {total_gb:.1f} GB")

# Pilot = smallest file
PILOT_FILE = csv_files[0]
OUT_FILE   = PARQUET_DIR / (PILOT_FILE.stem + ".parquet")

print(f"\nPilot target: {PILOT_FILE.name} ({PILOT_FILE.stat().st_size/1e6:.1f} MB)")
print("=" * 60)

# ── Step 0: Detect encoding ───────────────────────────────────────────────────
print("\n[0/4] Detecting file encoding from first 100 KB...")
with open(PILOT_FILE, "rb") as f:
    raw = f.read(100_000)

detection = chardet.detect(raw)
detected_enc = detection["encoding"]
confidence  = detection["confidence"]
print(f"      Detected: {detected_enc}  (confidence: {confidence:.0%})")

# Check for BOM
has_utf8_bom = raw[:3] == b"\xef\xbb\xbf"
if has_utf8_bom:
    detected_enc = "utf-8-sig"
    print("      UTF-8 BOM detected, using utf-8-sig")

# Map detected to Polars-compatible encoding name
ENC_MAP = {
    "GB2312": "gb18030",
    "GBK":    "gb18030",
    "GB18030":"gb18030",
    "UTF-8":  "utf8",
    "UTF-8-SIG": "utf8",
    "utf-8-sig": "utf8",
    "ascii":  "utf8",
}
polars_enc = ENC_MAP.get(detected_enc, "utf8")
print(f"      Using Polars encoding: {polars_enc}")

# ── Step 1: Sample to infer schema ───────────────────────────────────────────
print(f"\n[1/4] Sampling first 10,000 rows to infer schema (enc={polars_enc})...")
t0 = time.time()

# Try detected encoding, fall back through alternatives
encodings_to_try = [polars_enc, "utf8", "gb18030", "utf8-lossy"]
sample = None
used_enc = None

for enc in encodings_to_try:
    try:
        sample = pl.read_csv(
            PILOT_FILE,
            encoding=enc,
            n_rows=10_000,
            infer_schema_length=10_000,
            ignore_errors=True,
        )
        used_enc = enc
        print(f"      Successfully read with encoding: {enc}")
        break
    except Exception as e:
        print(f"      Encoding {enc} failed: {e}")

if sample is None:
    raise RuntimeError("Could not read CSV with any attempted encoding!")

print(f"      Elapsed: {time.time()-t0:.1f}s")
print(f"      Shape (sample): {sample.shape[0]} rows x {sample.shape[1]} columns")
print(f"\n      Columns & dtypes:")
for name, dtype in zip(sample.columns, sample.dtypes):
    n_null = sample[name].null_count()
    print(f"        {str(name):<40} {str(dtype):<20} nulls: {n_null}")

# ── Step 2: Streaming CSV → Parquet ──────────────────────────────────────────
print(f"\n[2/4] Converting full file to Parquet (streaming, enc={used_enc})...")
t0 = time.time()

(
    pl.scan_csv(
        PILOT_FILE,
        encoding=used_enc,
        infer_schema_length=10_000,
        ignore_errors=True,
    )
    .sink_parquet(
        OUT_FILE,
        row_group_size=500_000,
    )
)

elapsed = time.time() - t0
csv_size_mb  = PILOT_FILE.stat().st_size / 1e6
pq_size_mb   = OUT_FILE.stat().st_size   / 1e6
compression  = csv_size_mb / pq_size_mb if pq_size_mb > 0 else 0

print(f"      Elapsed:      {elapsed:.1f}s")
print(f"      CSV size:     {csv_size_mb:.1f} MB")
print(f"      Parquet size: {pq_size_mb:.1f} MB")
print(f"      Compression:  {compression:.1f}x")

# ── Step 3: Validate with DuckDB ─────────────────────────────────────────────
print(f"\n[3/4] Validating with DuckDB...")
t0 = time.time()

con = duckdb.connect()
pq_posix = OUT_FILE.as_posix()

row_count = con.sql(f"SELECT COUNT(*) FROM read_parquet('{pq_posix}')").fetchone()[0]
print(f"      Total rows: {row_count:,}")
print(f"      Elapsed:    {time.time()-t0:.2f}s")

print(f"\n      Schema from DuckDB:")
schema_df = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pq_posix}')").df()
print(schema_df[["column_name", "column_type"]].to_string(index=False))

print(f"\n      First 3 rows preview:")
preview = con.sql(f"SELECT * FROM read_parquet('{pq_posix}') LIMIT 3").df()
print(preview.to_string())

# ── Step 4: Summary ───────────────────────────────────────────────────────────
print(f"\n[4/4] Pilot Summary")
print(f"  File:          {PILOT_FILE.name}")
print(f"  Encoding used: {used_enc}")
print(f"  Columns:       {sample.shape[1]}")
print(f"  Total rows:    {row_count:,}")
print(f"  Output:        {OUT_FILE}")
print(f"  Size:          {csv_size_mb:.1f} MB -> {pq_size_mb:.1f} MB ({compression:.1f}x smaller)")
print("\nPilot complete. Next: run 02_convert_all.py to process all files.")
