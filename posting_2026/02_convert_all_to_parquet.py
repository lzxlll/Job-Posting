# -*- coding: utf-8 -*-
"""
02_convert_all_to_parquet.py
-----------------------------
Batch convert ALL CSV files in posting_2026/定制数据 to Parquet format.

Features:
  - Auto-detects encoding per file (falls back through utf8 / gb18030 / utf8-lossy)
  - Skips files already converted (resume-safe)
  - Giant files (>= 10 GB): partitioned into chunks of N rows per parquet file
  - Logs progress and timing to console + log file
  - Final summary table showing compression ratios

Usage:
    python 02_convert_all_to_parquet.py

Output:
    I:\\posting_2026\\parquet\\<stem>.parquet          (small/medium files)
    I:\\posting_2026\\parquet\\<stem>\\part_0000.parquet  (giant files, partitioned)
"""

import subprocess, sys, json, time
from datetime import datetime
from pathlib import Path

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

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR      = Path("I:\\posting_2026") / "\u5b9a\u5236\u6570\u636e"    # 定制数据
PARQUET_DIR  = Path(r"I:\posting_2026\parquet")
LOG_DIR      = Path(r"D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026")
LOG_FILE     = LOG_DIR / "02_convert_log.json"

PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Files >= this size get partitioned into multiple parquet files
GIANT_THRESHOLD_GB = 10.0
ROWS_PER_PARTITION = 5_000_000   # 5M rows per partition file

ENC_MAP = {
    "GB2312":  "gb18030",
    "GBK":     "gb18030",
    "GB18030": "gb18030",
    "UTF-8":   "utf8",
    "ASCII":   "utf8",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_encoding(path: Path, sample_bytes: int = 100_000) -> str:
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf8"
    result = chardet.detect(raw)
    detected = result.get("encoding") or "utf8"
    return ENC_MAP.get(detected.upper(), "utf8")

def try_encodings(path: Path, base_enc: str) -> tuple[str, pl.LazyFrame]:
    """Try multiple encodings, return the first that works."""
    candidates = list(dict.fromkeys([base_enc, "utf8", "gb18030", "utf8-lossy"]))
    for enc in candidates:
        try:
            lf = pl.scan_csv(
                path,
                encoding=enc,
                infer_schema_length=10_000,
                ignore_errors=True,
            )
            # Quick schema check
            _ = lf.schema
            return enc, lf
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {path.name} with any encoding.")

def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes/1e9:.3f} GB"

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"

# ── Load existing log (resume support) ────────────────────────────────────────
log: dict = {}
if LOG_FILE.exists():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log = json.load(f)
    print(f"Loaded conversion log: {len(log)} files previously recorded.")

def save_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

# ── Main conversion loop ──────────────────────────────────────────────────────
csv_files = sorted(RAW_DIR.glob("*.csv"), key=lambda f: f.stat().st_size)
total_csv_bytes = sum(f.stat().st_size for f in csv_files)

print("=" * 70)
print(f"Batch CSV → Parquet Conversion")
print(f"Files:     {len(csv_files)}")
print(f"Total CSV: {fmt_gb(total_csv_bytes)}")
print(f"Output:    {PARQUET_DIR}")
print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

results = []
session_start = time.time()

for idx, csv_path in enumerate(csv_files):
    name       = csv_path.name
    stem       = csv_path.stem
    csv_bytes  = csv_path.stat().st_size
    size_gb    = csv_bytes / 1e9
    is_giant   = size_gb >= GIANT_THRESHOLD_GB

    print(f"\n[{idx+1:02d}/{len(csv_files)}] {name}")
    print(f"         Size: {fmt_gb(csv_bytes)} | Giant: {is_giant}")

    # ── Skip if already done ──────────────────────────────────────────────────
    if stem in log and log[stem].get("status") == "done":
        pq_bytes = log[stem].get("parquet_bytes", 0)
        print(f"         SKIPPED (already converted, {fmt_gb(pq_bytes)})")
        results.append({**log[stem], "name": name})
        continue

    file_start = time.time()

    # ── Detect encoding ───────────────────────────────────────────────────────
    base_enc = detect_encoding(csv_path)
    print(f"         Encoding detected: {base_enc}")

    try:
        used_enc, lazy_frame = try_encodings(csv_path, base_enc)
        if used_enc != base_enc:
            print(f"         Fallback encoding used: {used_enc}")
    except RuntimeError as e:
        print(f"         ERROR: {e}")
        log[stem] = {"status": "error", "error": str(e), "name": name}
        save_log()
        continue

    # ── Convert ───────────────────────────────────────────────────────────────
    if not is_giant:
        # Single parquet file
        out_path = PARQUET_DIR / f"{stem}.parquet"
        print(f"         Converting → {out_path.name} ...")
        lazy_frame.sink_parquet(out_path, row_group_size=500_000)
        pq_bytes = out_path.stat().st_size
    else:
        # Partitioned: write in streaming chunks
        part_dir = PARQUET_DIR / stem
        part_dir.mkdir(exist_ok=True)
        print(f"         Giant file: partitioning into {part_dir.name}/ ...")

        # Stream directly — skip row-count pre-scan (too slow on 100GB+ files)
        print(f"         Giant file: streaming directly to {part_dir.name}/data.parquet ...")
        print(f"         (No row-count pre-scan — sink_parquet streams without full load)")

        lazy_frame.sink_parquet(
            part_dir / "data.parquet",
            row_group_size=500_000,
        )
        pq_bytes = sum(f.stat().st_size for f in part_dir.glob("*.parquet"))

    elapsed   = time.time() - file_start
    ratio     = csv_bytes / pq_bytes if pq_bytes > 0 else 0
    speed_mbs = (csv_bytes / 1e6) / elapsed if elapsed > 0 else 0

    print(f"         Done in {fmt_time(elapsed)}")
    print(f"         {fmt_gb(csv_bytes)} → {fmt_gb(pq_bytes)}  ({ratio:.1f}x compression)")
    print(f"         Speed: {speed_mbs:.1f} MB/s")

    entry = {
        "name":          name,
        "status":        "done",
        "encoding":      used_enc,
        "csv_bytes":     csv_bytes,
        "parquet_bytes": pq_bytes,
        "compression_x": round(ratio, 2),
        "elapsed_s":     round(elapsed, 1),
        "speed_mbs":     round(speed_mbs, 1),
        "converted_at":  datetime.now().isoformat(),
    }
    log[stem] = entry
    results.append({**entry, "name": name})
    save_log()

# ── Final Summary ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - session_start
done_results  = [r for r in results if r.get("status") == "done"]
total_pq_gb   = sum(r.get("parquet_bytes", 0) for r in done_results) / 1e9

print("\n" + "=" * 70)
print("CONVERSION SUMMARY")
print("=" * 70)
print(f"{'File':<50} {'CSV':>8} {'Parquet':>8} {'Ratio':>6} {'Time':>8}")
print("-" * 70)
for r in results:
    csv_g = r.get("csv_bytes", 0) / 1e9
    pq_g  = r.get("parquet_bytes", 0) / 1e9
    ratio = r.get("compression_x", 0)
    t     = fmt_time(r.get("elapsed_s", 0))
    status= "" if r.get("status") == "done" else " [SKIP]" if r.get("status") == "skipped" else " [ERR]"
    print(f"{r['name']:<50} {csv_g:>7.2f}G {pq_g:>7.2f}G {ratio:>5.1f}x {t:>8}{status}")

print("-" * 70)
print(f"{'TOTAL':<50} {total_csv_bytes/1e9:>7.2f}G {total_pq_gb:>7.2f}G")
print(f"\nTotal wall time: {fmt_time(total_elapsed)}")
print(f"Log saved to:    {LOG_FILE}")
print(f"\nNext step: run 03_schema_discovery.py")
