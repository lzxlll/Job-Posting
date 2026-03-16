# -*- coding: utf-8 -*-
"""
04_duckdb_validation.py
------------------------
Sanity-check all 15 Parquet tables:
  1. File readability + row count vs conversion log
  2. Join key coverage  (公司ID cross-table match rates)
  3. Known data-quality issues (swapped cols, 100%-null cols)
  4. Sample join  (招聘 ⋈ 工商信息)
  5. Summary report written to  04_validation_report.md

Run after 03_schema_discovery.py.
"""

import sys, json, time
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import duckdb

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_DIR = Path(r"I:\posting_2026\parquet")
LOG_PATH    = Path(r"D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\02_convert_log.json")
OUT_DIR     = Path(r"D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026")
OUT_MD      = OUT_DIR / "04_validation_report.md"

con = duckdb.connect()

# ── Helpers ───────────────────────────────────────────────────────────────────
def glob(name: str) -> str:
    """Return DuckDB glob string for a table name."""
    d = PARQUET_DIR / name
    f = PARQUET_DIR / f"{name}.parquet"
    if d.is_dir():
        return (d / "*.parquet").as_posix()
    return f.as_posix()

def sql(query: str, label: str = ""):
    t0 = time.time()
    try:
        result = con.sql(query).fetchall()
        elapsed = time.time() - t0
        return result, elapsed, None
    except Exception as e:
        return None, time.time() - t0, str(e)

def hr(n: int) -> str:
    """Human-readable row count."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

# ── Load conversion log ───────────────────────────────────────────────────────
with open(LOG_PATH, encoding="utf-8") as f:
    log = json.load(f)

# ── Discover tables ───────────────────────────────────────────────────────────
tables = {}
for item in sorted(PARQUET_DIR.iterdir()):
    if item.suffix == ".parquet":
        tables[item.stem] = glob(item.stem)
    elif item.is_dir() and any(item.glob("*.parquet")):
        tables[item.name] = glob(item.name)

print("=" * 70)
print("DuckDB Validation")
print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tables  : {len(tables)}")
print("=" * 70)

md = [
    "# DuckDB Validation Report — posting_2026",
    f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
]

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Row counts vs conversion log
# ══════════════════════════════════════════════════════════════════════════════
print("\n── CHECK 1: Row counts ──────────────────────────────────────────────")
md += ["\n## Check 1 — Row Counts vs Conversion Log\n",
       "| Table | Log (expected) | DuckDB (actual) | Match | Time |",
       "|-------|---------------|-----------------|-------|------|"]

all_match = True
for name, g in tables.items():
    log_rows = log.get(name, {}).get("row_count")   # may be absent in older logs
    rows, elapsed, err = sql(f"SELECT COUNT(*) FROM read_parquet('{g}')", name)
    actual = rows[0][0] if rows else None
    if err:
        status = f"❌ ERROR: {err[:60]}"
        all_match = False
    elif log_rows is not None and actual != log_rows:
        status = "⚠️ MISMATCH"
        all_match = False
    else:
        status = "✅"
    log_str = hr(log_rows) if log_rows else "n/a"
    act_str = hr(actual)   if actual  else "ERROR"
    print(f"  {name:<35} log={log_str:>8}  actual={act_str:>8}  {status}  ({elapsed:.1f}s)")
    md.append(f"| {name} | {log_str} | {act_str} | {status} | {elapsed:.1f}s |")

print(f"\n  Overall: {'ALL MATCH ✅' if all_match else 'SOME ISSUES ⚠️'}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Join key coverage  (公司ID)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── CHECK 2: 公司ID join coverage ─────────────────────────────────────")
md += ["\n## Check 2 — 公司ID Join Key Coverage\n",
       "Measures what % of 公司ID values in each table match 工商信息 (master).\n",
       "| Table | Distinct 公司ID | Match to 工商信息 | Match % | Time |",
       "|-------|---------------|-----------------|---------|------|"]

# Tables that have 公司ID
gongsi_tables = [n for n in tables if n != "工商信息"]
master_glob   = glob("工商信息")

for name in gongsi_tables:
    g = tables[name]
    # Check if table has 公司ID column
    cols_res, _, _ = sql(f"DESCRIBE SELECT * FROM read_parquet('{g}')")
    if cols_res is None:
        continue
    col_names = [r[0] for r in cols_res]
    if "公司ID" not in col_names:
        continue

    q = f"""
        WITH src AS (
            SELECT DISTINCT 公司ID FROM read_parquet('{g}')
        ),
        matched AS (
            SELECT COUNT(*) AS n
            FROM src s
            INNER JOIN (SELECT DISTINCT 公司ID FROM read_parquet('{master_glob}')) m
              ON s.公司ID = m.公司ID
        )
        SELECT (SELECT COUNT(*) FROM src) AS total,
               (SELECT n FROM matched)   AS matched
    """
    rows, elapsed, err = sql(q, name)
    if err or rows is None:
        print(f"  {name:<35} ERROR: {str(err)[:60]}")
        md.append(f"| {name} | - | - | ❌ ERROR | {elapsed:.1f}s |")
        continue
    total, matched = rows[0]
    pct = matched / total * 100 if total > 0 else 0
    flag = "✅" if pct >= 80 else ("⚠️" if pct >= 50 else "❌")
    print(f"  {name:<35} total={hr(total):>8}  matched={hr(matched):>8}  {pct:.1f}%  {flag}  ({elapsed:.1f}s)")
    md.append(f"| {name} | {hr(total)} | {hr(matched)} | {pct:.1f}% {flag} | {elapsed:.1f}s |")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Known data quality issues
# ══════════════════════════════════════════════════════════════════════════════
print("\n── CHECK 3: Known data-quality issues ───────────────────────────────")
md += ["\n## Check 3 — Known Data Quality Issues\n"]

checks = [
    ("工商变更",  "变更类型",    "SELECT COUNT(*) FROM read_parquet('{g}') WHERE 变更类型 IS NOT NULL",
     "变更类型 should be 100% NULL"),
    ("产品许可",  "截止日期",    "SELECT COUNT(*) FROM read_parquet('{g}') WHERE 截止日期 IS NOT NULL",
     "截止日期 should be 100% NULL"),
    ("对外投资",  "认缴投资金额/时间 swap",
     "SELECT 认缴投资金额, 认缴投资时间 FROM read_parquet('{g}') WHERE 认缴投资金额 LIKE '%-%-%' LIMIT 3",
     "Dates appearing in 认缴投资金额 → columns are swapped"),
]

for tbl, col, query_tmpl, note in checks:
    g = tables.get(tbl, "")
    if not g:
        continue
    query = query_tmpl.replace("{g}", g)
    rows, elapsed, err = sql(query)
    if err:
        result_str = f"ERROR: {err[:80]}"
    else:
        result_str = str(rows)
    print(f"  [{tbl}] {note}")
    print(f"    → {result_str}  ({elapsed:.1f}s)")
    md += [f"### {tbl} — {note}", f"```\n{result_str}\n```\n"]

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Sample join: 招聘 ⋈ 工商信息
# ══════════════════════════════════════════════════════════════════════════════
print("\n── CHECK 4: Sample join  招聘 ⋈ 工商信息 ─────────────────────────────")
md += ["\n## Check 4 — Sample Join: 招聘 ⋈ 工商信息\n"]

join_q = f"""
    SELECT
        z.公司ID,
        z.工作名称,
        z.工作薪酬,
        z.发布日期,
        g.省份,
        g.一级行业,
        g.经营状态
    FROM read_parquet('{glob("招聘")}')   z
    JOIN read_parquet('{glob("工商信息")}') g
      ON z.公司ID = g.公司ID
    WHERE z.发布日期 >= '2023-01-01'
      AND g.经营状态 LIKE '%在营%'
    LIMIT 10
"""
rows, elapsed, err = sql(join_q)
if err:
    print(f"  ERROR: {err}")
    md.append(f"❌ Join failed: {err}\n")
else:
    print(f"  Join succeeded — {len(rows)} sample rows returned  ({elapsed:.1f}s)")
    md += ["Join returned sample rows successfully ✅\n",
           "| 公司ID | 工作名称 | 薪酬 | 发布日期 | 省份 | 行业 | 经营状态 |",
           "|--------|---------|------|---------|------|------|---------|"]
    for r in rows:
        md.append("| " + " | ".join(str(x)[:30] for x in r) + " |")
    for r in rows[:5]:
        print(f"    {r}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — 招聘 date range & top job titles
# ══════════════════════════════════════════════════════════════════════════════
print("\n── CHECK 5: 招聘 date range & top job titles ─────────────────────────")
md += ["\n## Check 5 — 招聘 Quick Profiling\n"]

date_q = f"""
    SELECT MIN(发布日期) AS min_date,
           MAX(发布日期) AS max_date,
           COUNT(DISTINCT 数据来源) AS n_sources
    FROM read_parquet('{glob("招聘")}')
"""
rows, elapsed, err = sql(date_q)
if rows:
    min_d, max_d, n_src = rows[0]
    print(f"  Date range: {min_d} → {max_d}   Sources: {n_src}  ({elapsed:.1f}s)")
    md.append(f"- Date range: **{min_d}** → **{max_d}**\n- Distinct data sources: **{n_src}**\n")

top_q = f"""
    SELECT 工作名称, COUNT(*) AS n
    FROM read_parquet('{glob("招聘")}')
    GROUP BY 工作名称
    ORDER BY n DESC
    LIMIT 10
"""
rows, elapsed, err = sql(top_q)
if rows:
    print(f"  Top 10 job titles  ({elapsed:.1f}s):")
    md += ["### Top 10 Job Titles\n",
           "| 工作名称 | Count |", "|---------|-------|"]
    for title, cnt in rows:
        print(f"    {cnt:>10,}  {title}")
        md.append(f"| {title} | {cnt:,} |")

# ══════════════════════════════════════════════════════════════════════════════
# Finalise
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Validation complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Report written to:   {OUT_MD}")
print("=" * 70)

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
