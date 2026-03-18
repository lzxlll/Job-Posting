# -*- coding: utf-8 -*-
"""
06a_profile_dirty_cols.py
--------------------------
Profiles every column that needs numeric parsing BEFORE we write
cleaning rules.  Outputs a report so we can see all real-world patterns.

Columns to profile:
  招聘          工作薪酬
  工商信息       注册资本
  年报社保财报信息 参保人数, 资产总额, 净利润, 纳税总额
  股东信息       持股比例, 认缴, 实缴
  股权出质       出质股权数额

Strategy: pull top-200 most-frequent values per column (fast — DuckDB
reads only that one column from Parquet, no full scan).
"""

import sys, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import duckdb
from config import DB_PATH, TMP_DIR

DB   = str(DB_PATH)
OUT  = Path(__file__).parent / "06a_dirty_col_profile.md"
TOP  = 200   # top N values per column

con  = duckdb.connect(DB)
con.execute("SET memory_limit='96GB'")
con.execute("SET threads=8")

lines = [
    "# Dirty Column Profile — posting_2026",
    f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    "Showing top 200 most-frequent non-null values per column.\n",
    "> `--` is the dataset's null placeholder.\n",
]

TARGETS = [
    # (view_name,  column,          label)
    ("招聘",       "工作薪酬",       "招聘.工作薪酬"),
    ("工商信息",    "注册资本",       "工商信息.注册资本"),
    ("年报社保财报信息", "参保人数",   "年报.参保人数"),
    ("年报社保财报信息", "资产总额",   "年报.资产总额"),
    ("年报社保财报信息", "净利润",     "年报.净利润"),
    ("年报社保财报信息", "纳税总额",   "年报.纳税总额"),
    ("股东信息",    "持股比例",       "股东信息.持股比例"),
    ("股东信息",    "认缴",          "股东信息.认缴"),
    ("股东信息",    "实缴",          "股东信息.实缴"),
    ("股权出质",    "出质股权数额",   "股权出质.出质股权数额"),
]

print("=" * 65)
print("Dirty Column Profiler")
print(f"Querying {len(TARGETS)} columns  ...  (may take ~10 min)")
print("=" * 65)

for view, col, label in TARGETS:
    t0 = time.time()
    print(f"\n  [{label}] ... ", end="", flush=True)

    # Total rows & null / '--' rates
    try:
        stats = con.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN "{col}" IS NULL OR "{col}" = '--' THEN 1 ELSE 0 END) AS null_like,
                COUNT(DISTINCT "{col}") AS n_distinct
            FROM "{view}"
        """).fetchone()
        total, null_like, n_distinct = stats
        null_pct = null_like / total * 100 if total else 0
    except Exception as e:
        print(f"ERROR: {e}")
        continue

    # Top values by frequency
    rows = con.execute(f"""
        SELECT "{col}" AS val, COUNT(*) AS n
        FROM "{view}"
        WHERE "{col}" IS NOT NULL
          AND "{col}" != '--'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT {TOP}
    """).fetchall()

    elapsed = time.time() - t0
    print(f"{total:,} rows  |  null/-- {null_pct:.1f}%  |  {n_distinct:,} distinct  ({elapsed:.1f}s)")

    lines += [
        f"\n---\n## {label}",
        f"\n- **Total rows**: {total:,}",
        f"- **Null or '--'**: {null_like:,} ({null_pct:.1f}%)",
        f"- **Distinct values**: {n_distinct:,}",
        f"\n| Rank | Value | Count |",
        "|------|-------|-------|",
    ]
    for i, (val, cnt) in enumerate(rows, 1):
        display = str(val)[:80].replace("|", "｜")
        lines.append(f"| {i} | `{display}` | {cnt:,} |")
        if i <= 30:
            print(f"    {cnt:>12,}  {display}")

OUT.write_text("\n".join(lines), encoding="utf-8")
con.close()

print(f"\n{'='*65}")
print(f"✅  Profile written → {OUT}")
print(f"{'='*65}")
