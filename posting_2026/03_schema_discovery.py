# -*- coding: utf-8 -*-
"""
03_schema_discovery.py
-----------------------
Scan all converted Parquet files and produce a data dictionary:
  - Column names, types, null rates, cardinality estimates
  - Sample values per column
  - Cross-table key candidate detection (columns that appear in multiple tables)
  - Output: schema_dictionary.xlsx + schema_dictionary.md

Run AFTER 02_convert_all_to_parquet.py has completed.

Usage:
    python 03_schema_discovery.py
"""

import subprocess, sys
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

for pkg in ["polars", "duckdb", "pyarrow", "openpyxl"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import polars as pl
import duckdb

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_DIR = Path(r"I:\posting_2026\parquet")
OUT_DIR     = Path(r"D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026")
SAMPLE_ROWS = 50_000   # rows to sample per file for cardinality estimates

print("=" * 70)
print("Schema Discovery & Data Dictionary")
print(f"Parquet directory: {PARQUET_DIR}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

con = duckdb.connect()

# ── Discover parquet files/dirs ───────────────────────────────────────────────
def get_parquet_glob(p: Path) -> str:
    """Return DuckDB-compatible glob for a parquet file or partitioned dir."""
    if p.is_dir():
        return (p / "*.parquet").as_posix()
    return p.as_posix()

parquet_paths = []
for item in sorted(PARQUET_DIR.iterdir()):
    if item.suffix == ".parquet":
        parquet_paths.append((item.stem, item))
    elif item.is_dir() and any(item.glob("*.parquet")):
        parquet_paths.append((item.name, item))

print(f"\nFound {len(parquet_paths)} Parquet table(s):\n")

# ── Per-table schema ──────────────────────────────────────────────────────────
all_rows = []      # for Excel/markdown output
col_index = {}     # column_name -> list of tables containing it

for table_name, pq_path in parquet_paths:
    glob_path = get_parquet_glob(pq_path)
    print(f"  Analysing: {table_name}")

    try:
        # Row count
        row_count = con.sql(
            f"SELECT COUNT(*) FROM read_parquet('{glob_path}')"
        ).fetchone()[0]

        # Schema
        schema_df = con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{glob_path}')"
        ).df()
        columns = schema_df["column_name"].tolist()
        dtypes  = schema_df["column_type"].tolist()

        # Sample for null rate + cardinality (on SAMPLE_ROWS rows)
        sample_sql = f"""
            SELECT * FROM read_parquet('{glob_path}')
            USING SAMPLE {SAMPLE_ROWS} ROWS
        """
        sample_df = con.sql(sample_sql).df()
        actual_sample = len(sample_df)

        for col, dtype in zip(columns, dtypes):
            # Null rate from sample
            null_count = int(sample_df[col].isna().sum())
            null_rate  = null_count / actual_sample if actual_sample > 0 else 0

            # Approx cardinality from sample
            n_unique = int(sample_df[col].nunique())

            # Sample values (up to 5 non-null)
            non_null_vals = sample_df[col].dropna().astype(str).unique()[:5].tolist()
            sample_vals   = " | ".join(non_null_vals)

            all_rows.append({
                "table":        table_name,
                "column":       col,
                "dtype":        dtype,
                "row_count":    row_count,
                "null_rate_%":  round(null_rate * 100, 1),
                "approx_unique":n_unique,
                "sample_values":sample_vals[:120],   # truncate
            })

            # Track columns appearing across multiple tables (join key candidates)
            if col not in col_index:
                col_index[col] = []
            col_index[col].append(table_name)

        print(f"    {row_count:>12,} rows | {len(columns)} columns")

    except Exception as e:
        print(f"    ERROR: {e}")
        all_rows.append({
            "table": table_name, "column": "ERROR",
            "dtype": str(e), "row_count": 0,
            "null_rate_%": 0, "approx_unique": 0, "sample_values": "",
        })

# ── Identify join key candidates ──────────────────────────────────────────────
join_candidates = {
    col: tables
    for col, tables in col_index.items()
    if len(tables) >= 2
}

print(f"\n{'='*70}")
print(f"Join key candidates (columns appearing in 2+ tables):")
print(f"{'='*70}")
for col, tables in sorted(join_candidates.items(), key=lambda x: -len(x[1])):
    print(f"  {col:<40} in {len(tables)} tables: {', '.join(tables)}")

# ── Write Markdown data dictionary ───────────────────────────────────────────
md_lines = [
    "# Data Dictionary — posting_2026 定制数据",
    f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    "## Join Key Candidates\n",
    "Columns present in 2 or more tables (potential join keys):\n",
    "| Column | # Tables | Tables |",
    "|--------|----------|--------|",
]
for col, tables in sorted(join_candidates.items(), key=lambda x: -len(x[1])):
    md_lines.append(f"| `{col}` | {len(tables)} | {', '.join(tables)} |")

md_lines += ["\n---\n", "## Per-Table Schemas\n"]

current_table = None
for row in all_rows:
    if row["table"] != current_table:
        current_table = row["table"]
        rows_in_table = row.get("row_count", 0)
        md_lines += [
            f"\n### {current_table}",
            f"**Rows**: {rows_in_table:,}\n",
            "| Column | Type | Null % | ~Unique | Sample Values |",
            "|--------|------|--------|---------|---------------|",
        ]
    md_lines.append(
        f"| `{row['column']}` | {row['dtype']} | {row['null_rate_%']}% "
        f"| {row['approx_unique']:,} | {row['sample_values']} |"
    )

md_path = OUT_DIR / "schema_dictionary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"\nMarkdown saved: {md_path}")

# ── Write Excel data dictionary ───────────────────────────────────────────────
try:
    import openpyxl
    xl_path = OUT_DIR / "schema_dictionary.xlsx"

    schema_pl = pl.DataFrame(all_rows)
    schema_pl.write_excel(
        xl_path,
        worksheet="Schema",
        autofit=True,
        freeze_row=1,
        column_widths={
            "table": 25, "column": 30, "dtype": 15, "row_count": 15,
            "null_rate_%": 12, "approx_unique": 15, "sample_values": 60,
        },
        header_format={"bold": True, "bg_color": "#4472C4", "font_color": "#FFFFFF"},
    )

    # Second sheet: join key candidates
    with openpyxl.load_workbook(xl_path) as wb:
        ws2 = wb.create_sheet("JoinKeys")
        ws2.append(["Column", "# Tables", "Tables"])
        for col, tables in sorted(join_candidates.items(), key=lambda x: -len(x[1])):
            ws2.append([col, len(tables), ", ".join(tables)])
        wb.save(xl_path)

    print(f"Excel saved:    {xl_path}")
except Exception as e:
    print(f"Excel export skipped: {e}")

print("\nSchema discovery complete.")
print("Review schema_dictionary.md to identify join keys before merging tables.")
