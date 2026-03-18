# -*- coding: utf-8 -*-
"""
05_setup_duckdb.py
------------------
Creates a persistent DuckDB database at I:\posting_2026\enterprise.duckdb

What this does:
  1. Creates/opens the .duckdb file
  2. Registers all 15 Parquet tables as VIEWs (no data copied — views point
     directly at the Parquet files on disk)
  3. Creates useful helper MACROs for common patterns
  4. Verifies every view is readable and prints row counts
  5. Prints a quick-start SQL cheatsheet

After running this script, open enterprise.duckdb with:
  - DBeaver (free GUI, recommended)
  - Python: duckdb.connect(r'I:\\posting_2026\\enterprise.duckdb')
  - CLI:    duckdb I:\\posting_2026\\enterprise.duckdb
"""

import sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import duckdb
from config import DB_PATH, PARQUET_DIR, TMP_DIR

TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Table registry ─────────────────────────────────────────────────────────────
# name → glob pattern relative to PARQUET_DIR
TABLES = {
    "招聘":            "招聘/*.parquet",
    "工商信息":         "工商信息/*.parquet",
    "工商变更":         "工商变更/*.parquet",
    "年报社保财报信息":  "年报社保财报信息/*.parquet",
    "主要人员":         "主要人员/*.parquet",
    "股东信息":         "股东信息/*.parquet",
    "对外投资":         "对外投资.parquet",
    "分支机构":         "分支机构.parquet",
    "产品许可":         "产品许可.parquet",
    "动产抵押":         "动产抵押.parquet",
    "动产抵押_抵押物":  "动产抵押-抵押物.parquet",
    "动产抵押_抵押人":  "动产抵押-抵押人.parquet",
    "动产抵押_抵押变更": "动产抵押-抵押变更.parquet",
    "股权出质":         "股权出质.parquet",
    "融资历史":         "融资历史.parquet",
}
# Note: DuckDB view names use underscore instead of hyphen (SQL-safe)
# e.g. 动产抵押_抵押物  (not 动产抵押-抵押物)

print("=" * 65)
print("enterprise.duckdb — Setup")
print(f"DB path : {DB_PATH}")
print("=" * 65)

# ── Connect (creates file if new, opens if existing) ──────────────────────────
con = duckdb.connect(str(DB_PATH))
con.execute(f"SET memory_limit='96GB'")
con.execute(f"SET threads=8")
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
print(f"\n[1] Connected to {DB_PATH}")

# ── Drop & recreate all views ─────────────────────────────────────────────────
print("\n[2] Creating views ...")
for view_name, rel_glob in TABLES.items():
    abs_glob = (PARQUET_DIR / rel_glob).as_posix()
    con.execute(f"DROP VIEW IF EXISTS \"{view_name}\"")
    con.execute(f"CREATE VIEW \"{view_name}\" AS SELECT * FROM read_parquet('{abs_glob}')")
    print(f"    ✓  {view_name}")

# ── Useful MACROs ──────────────────────────────────────────────────────────────
print("\n[3] Creating macros ...")

macros = {
    # Quick company lookup by name (partial match)
    "find_company": """
        CREATE OR REPLACE MACRO find_company(kw) AS TABLE
        SELECT 公司ID, 公司名称, 省份, 城市, 一级行业, 经营状态, 注册资本
        FROM   工商信息
        WHERE  公司名称 LIKE '%' || kw || '%'
        LIMIT  50
    """,

    # Recent postings for a company
    "company_jobs": """
        CREATE OR REPLACE MACRO company_jobs(cid) AS TABLE
        SELECT 工作名称, 工作薪酬, 发布日期, 公司所在区域, 工作学历, 数据来源
        FROM   招聘
        WHERE  公司ID = cid
        ORDER  BY 发布日期 DESC
        LIMIT  200
    """,

    # Shareholder list for a company
    "company_shareholders": """
        CREATE OR REPLACE MACRO company_shareholders(cid) AS TABLE
        SELECT 股东名称, 股东类型, 持股比例, 认缴, 实缴
        FROM   股东信息
        WHERE  公司ID = cid
    """,

    # Key personnel for a company
    "company_personnel": """
        CREATE OR REPLACE MACRO company_personnel(cid) AS TABLE
        SELECT 主要人员, 职位
        FROM   主要人员
        WHERE  公司ID = cid
    """,

    # Annual report financials for a company
    "company_financials": """
        CREATE OR REPLACE MACRO company_financials(cid) AS TABLE
        SELECT 年份, 参保人数, 资产总额, 所有者权益合计,
               "销售总额(营业总收入)" AS 营业收入, 净利润, 纳税总额
        FROM   年报社保财报信息
        WHERE  公司ID = cid
        ORDER  BY 年份 DESC
    """,
}

for name, ddl in macros.items():
    try:
        con.execute(ddl)
        print(f"    ✓  {name}()")
    except Exception as e:
        # Column name may differ — skip gracefully
        print(f"    ⚠  {name}() skipped: {e}")

# ── Verify all views ──────────────────────────────────────────────────────────
print("\n[4] Verifying views (row counts) ...")
print(f"  {'View':<30} {'Rows':>15}  {'Time':>6}")
print("  " + "-" * 55)
total_rows = 0
for view_name in TABLES:
    t0  = time.time()
    try:
        n   = con.execute(f'SELECT COUNT(*) FROM "{view_name}"').fetchone()[0]
        ms  = (time.time() - t0) * 1000
        total_rows += n
        print(f"  {view_name:<30} {n:>15,}  {ms:>5.0f}ms")
    except Exception as e:
        print(f"  {view_name:<30} ERROR: {e}")

print(f"  {'TOTAL':<30} {total_rows:>15,}")

# ── List all objects in the DB ────────────────────────────────────────────────
print("\n[5] Objects in enterprise.duckdb:")
objects = con.execute("SHOW ALL TABLES").fetchall()
for obj in objects:
    print(f"    {obj[2]:<35} ({obj[3]})")  # name, type

con.close()

print("\n" + "=" * 65)
print("✅  enterprise.duckdb is ready!")
print("=" * 65)
print(f"""
Quick-start examples
────────────────────
Python:
    import duckdb
    con = duckdb.connect(r'I:\\posting_2026\\enterprise.duckdb')

    # Find a company
    con.sql("FROM find_company('华为')").show()

    # Recent job postings
    con.sql("FROM company_jobs(12345678)").show()

    # Ad-hoc SQL
    con.sql(\"\"\"
        SELECT 一级行业, COUNT(*) AS jobs, COUNT(DISTINCT 公司ID) AS firms
        FROM 招聘 z JOIN 工商信息 g ON z.公司ID = g.公司ID
        WHERE 发布日期 >= '2024-01-01'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 15
    \"\"\").show()

DBeaver:
    New Connection → DuckDB → Database file: I:\\posting_2026\\enterprise.duckdb

CLI:
    duckdb I:\\posting_2026\\enterprise.duckdb
    > SHOW TABLES;
    > FROM find_company('阿里巴巴');
""")
