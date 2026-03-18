# -*- coding: utf-8 -*-
"""
07_data_cleaning.py
-------------------
Parses and cleans 10 dirty numeric/semi-structured columns in the
enterprise dataset.  Produces a validated, typed column alongside the
original for every target.

Columns cleaned
───────────────
招聘.工作薪酬         → salary_lo / salary_hi (float, 元/月)
                         salary_type  ('range'|'single'|'daily'|'negotiable')
工商信息.注册资本      → reg_capital_wan (float, 万元CNY)
                         reg_capital_currency ('CNY'|'USD'|'HKD'|'EUR'|…)
年报.参保人数         → insured_n (int)
年报.资产总额         → total_assets_wan (float, 万元)  NULL=not-disclosed/unknown
年报.净利润           → net_profit_wan (float, 万元)    (can be negative)
年报.纳税总额         → total_tax_wan (float, 万元)
股东信息.持股比例      → share_pct (float, 0-100)
股东信息.认缴         → subscribed_wan (float, 万元CNY)
                         subscribed_currency ('CNY'|'USD'|'HKD'|…)
股东信息.实缴         → paid_wan (float, 万元CNY)
                         paid_currency ('CNY'|'USD'|'HKD'|…)
股权出质.出质股权数额  → pledge_amount (float)
                         pledge_unit ('元'|'万元'|'万股')

Strategy
────────
• All cleaning is done in DuckDB via SQL CASE/regex expressions so that
  the heavy data never leaves disk.  We create new columns as VIEWs
  on top of the existing Parquet-backed views — zero data duplication.
• The final artefact is a DuckDB VIEW per table with "_clean" suffix.
• A validation report is written to 07_cleaning_report.md.
• The clean views are also registered in enterprise.duckdb for direct
  SQL access.

Usage
─────
    python 07_data_cleaning.py
    # Runs in ~15-30 min depending on I/O.
    # On completion enterprise.duckdb has views:
    #   招聘_clean, 工商信息_clean, 年报社保财报信息_clean,
    #   股东信息_clean, 股权出质_clean
"""

import sys, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import duckdb
from config import DB_PATH, TMP_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
DB      = str(DB_PATH)
OUT     = Path(__file__).parent / "07_cleaning_report.md"

con = duckdb.connect(DB)
con.execute("SET memory_limit='96GB'")
con.execute("SET threads=8")
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")

lines = [
    "# Data Cleaning Report — posting_2026",
    f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    "Parsed numeric/typed columns created as DuckDB views (`<table>_clean`).\n",
]

print("=" * 65)
print("07_data_cleaning.py")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: run a CREATE OR REPLACE VIEW, time it, print status
# ══════════════════════════════════════════════════════════════════════════════
def create_view(view_name: str, sql: str) -> float:
    t0 = time.time()
    print(f"\n  Creating view [{view_name}] ...", end="", flush=True)
    con.execute(f'DROP VIEW IF EXISTS "{view_name}"')
    con.execute(f'CREATE VIEW "{view_name}" AS\n{sql}')
    elapsed = time.time() - t0
    print(f" done ({elapsed:.1f}s)")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: run a quick validation query and return result rows
# ══════════════════════════════════════════════════════════════════════════════
def val_query(sql: str):
    return con.execute(sql).fetchall()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  招聘.工作薪酬  →  salary_lo, salary_hi (元/月), salary_type
# ══════════════════════════════════════════════════════════════════════════════
#
# Patterns observed (in order of frequency):
#   面议 / 薪资面议          → negotiable      salary_lo=NULL salary_hi=NULL
#   3000-5000               → range (元/月)   lo=3000 hi=5000
#   6000-8000元/月           → range (元/月)
#   5-10K / 3-4K            → range *1000     lo=5000 hi=10000
#   6-8千/月  4.5-6千/月     → range *1000
#   0.8-1万/月               → range *10000
#   1万-1.5万 / 1-1.5万      → range *10000
#   8千-1.2万 / 8千-1万      → mixed 千/万
#   600元/天                 → daily           salary_type='daily'
#   10000以上 / 25000以上    → single lower bound
#   3000以上 / 3000元以上    → single lower bound
#   10-15k·12薪             → range *1000 (ignore "·12薪" annualisation hint)
#   10000-99999999           → treat hi>500000 as open-ended upper  (salary_hi=NULL)
#
# Unit multipliers after stripping suffixes:
#   K / k / K·12薪           ×1000
#   千                        ×1000
#   万                        ×10000
#   (bare number or 元/月)    ×1
#
# We implement this as a large CASE expression operating on the raw string.
# The approach: normalise → extract lo/hi → apply unit multiplier.
# Because the variety is huge, we use a chain of regexp_extract calls.
#
# Implementation note: DuckDB's regexp_extract returns '' on no-match.
# We coerce '' → NULL via NULLIF.

print("\n[1] 招聘.工作薪酬 → salary_lo / salary_hi / salary_type")

SALARY_SQL = r"""
SELECT
    *,
    -- ── classify salary type ──────────────────────────────────────────
    CASE
        WHEN 工作薪酬 IS NULL OR 工作薪酬 = '--'
            THEN 'null'
        WHEN regexp_matches(工作薪酬, '面议|薪资面议')
            THEN 'negotiable'
        WHEN regexp_matches(工作薪酬, '元/天|元/日|/天|/日')
            THEN 'daily'
        WHEN regexp_matches(工作薪酬, '以上|以上$')
            THEN 'lower_bound'
        ELSE 'range'
    END AS salary_type,

    -- ── low end (元/月) ───────────────────────────────────────────────
    CASE
        WHEN 工作薪酬 IS NULL OR 工作薪酬 = '--'
            THEN NULL
        WHEN regexp_matches(工作薪酬, '面议|薪资面议')
            THEN NULL

        -- daily wages: keep raw number, flag via salary_type
        WHEN regexp_matches(工作薪酬, '([0-9]+(?:\.[0-9]+)?)元/天')
            THEN TRY_CAST(
                regexp_extract(工作薪酬, '([0-9]+(?:\.[0-9]+)?)元/天', 1)
            AS DOUBLE)

        -- X万-Y万 / X-Y万  (e.g. "1万-1.5万", "0.8-1万/月")
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[万]?[-–~]([0-9]+(?:\.[0-9]+)?)万')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[万]?[-–~]([0-9]+(?:\.[0-9]+)?)万', 1)
            AS DOUBLE) * 10000

        -- X千-Y万 (e.g. "8千-1.2万")
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)千[-–~]([0-9]+(?:\.[0-9]+)?)万')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)千[-–~]([0-9]+(?:\.[0-9]+)?)万', 1)
            AS DOUBLE) * 1000

        -- X-Y千 / X-Y千/月  (e.g. "6-8千/月")
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)千')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)千', 1)
            AS DOUBLE) * 1000

        -- X-YK / X-Yk  (e.g. "5-10K", "3-4K")
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)[Kk]')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)[Kk]', 1)
            AS DOUBLE) * 1000

        -- XK-YK  (e.g. "4K-6K", "8K-10K")
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[Kk][-–]([0-9]+(?:\.[0-9]+)?)[Kk]')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[Kk][-–]([0-9]+(?:\.[0-9]+)?)[Kk]', 1)
            AS DOUBLE) * 1000

        -- bare X-Y (元/月, optional suffix e.g. "元/月", "元")
        WHEN regexp_matches(工作薪酬,
            '^([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '^([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)', 1)
            AS DOUBLE)

        -- X以上 / X元以上
        WHEN regexp_matches(工作薪酬, '([0-9]+(?:\.[0-9]+)?)元?以上')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)元?以上', 1)
            AS DOUBLE)

        ELSE NULL
    END AS salary_lo,

    -- ── high end (元/月) ──────────────────────────────────────────────
    CASE
        WHEN 工作薪酬 IS NULL OR 工作薪酬 = '--'
            THEN NULL
        WHEN regexp_matches(工作薪酬, '面议|薪资面议')
            THEN NULL
        WHEN regexp_matches(工作薪酬, '元/天|元/日|/天|/日')
            THEN NULL   -- hi unused for daily rate
        WHEN regexp_matches(工作薪酬, '以上')
            THEN NULL   -- open upper bound

        -- X万-Y万
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[万]?[-–~]([0-9]+(?:\.[0-9]+)?)万')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[万]?[-–~]([0-9]+(?:\.[0-9]+)?)万', 2)
            AS DOUBLE) * 10000

        -- X千-Y万
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)千[-–~]([0-9]+(?:\.[0-9]+)?)万')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)千[-–~]([0-9]+(?:\.[0-9]+)?)万', 2)
            AS DOUBLE) * 10000

        -- X-Y千
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)千')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)千', 2)
            AS DOUBLE) * 1000

        -- X-YK
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)[Kk]')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)[Kk]', 2)
            AS DOUBLE) * 1000

        -- XK-YK
        WHEN regexp_matches(工作薪酬,
            '([0-9]+(?:\.[0-9]+)?)[Kk][-–]([0-9]+(?:\.[0-9]+)?)[Kk]')
            THEN TRY_CAST(
                regexp_extract(工作薪酬,
                    '([0-9]+(?:\.[0-9]+)?)[Kk][-–]([0-9]+(?:\.[0-9]+)?)[Kk]', 2)
            AS DOUBLE) * 1000

        -- bare X-Y: if hi > 500000 treat as garbage upper bound → NULL
        WHEN regexp_matches(工作薪酬,
            '^([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)')
            THEN CASE
                WHEN TRY_CAST(
                    regexp_extract(工作薪酬,
                        '^([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)', 2)
                AS DOUBLE) > 500000
                THEN NULL
                ELSE TRY_CAST(
                    regexp_extract(工作薪酬,
                        '^([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)', 2)
                AS DOUBLE)
            END

        ELSE NULL
    END AS salary_hi

FROM "招聘"
"""

create_view("招聘_clean", SALARY_SQL)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  工商信息.注册资本  →  reg_capital_wan (float, 万元), reg_capital_currency
# ══════════════════════════════════════════════════════════════════════════════
#
# Patterns:
#   100.000000万人民币   → 100 万CNY
#   1万元人民币          → 1   万CNY
#   1.0万元              → 1   万CNY
#   100万人民币          → 100 万CNY
#   10万                 → 10  万CNY
#   1(万元)  1（万元）   → 1   万CNY
#   1.0  (bare float)    → interpret as 万元 (most common unit in this field)
#   0                    → 0 万CNY
#   -  /  未公示  /  空格 → NULL
#
# Currency keywords: 人民币→CNY, 美元→USD, 港元/港币→HKD, 欧元→EUR,
#                    英镑→GBP, 日元→JPY  (default CNY)

print("\n[2] 工商信息.注册资本 → reg_capital_wan / reg_capital_currency")

REG_CAP_SQL = r"""
SELECT
    *,
    CASE
        WHEN 注册资本 IS NULL OR 注册资本 = '--' OR TRIM(注册资本) = ''
            THEN NULL
        WHEN TRIM(注册资本) IN ('-', '未公示', '0', '0万元人民币',
                                 '0.000000万人民币', '0.000000万', '0万人民币')
            THEN 0.0
        -- "X万..." patterns (万 present in string)
        WHEN regexp_matches(注册资本, '([0-9]+(?:\.[0-9]+)?)万')
            THEN TRY_CAST(
                regexp_extract(注册资本, '([0-9]+(?:\.[0-9]+)?)万', 1)
            AS DOUBLE)
        -- "X元..." patterns WITHOUT 万 — convert 元→万 (÷10000)
        WHEN regexp_matches(注册资本, '([0-9]+(?:\.[0-9]+)?)元')
            THEN TRY_CAST(
                regexp_extract(注册资本, '([0-9]+(?:\.[0-9]+)?)元', 1)
            AS DOUBLE) / 10000.0
        -- bare number (e.g. "100.000000" or "1.0") — interpret as 万元
        WHEN regexp_matches(注册资本, '^[0-9]+(?:\.[0-9]+)?$')
            THEN TRY_CAST(注册资本 AS DOUBLE)
        ELSE NULL
    END AS reg_capital_wan,

    CASE
        WHEN 注册资本 IS NULL OR 注册资本 = '--' OR TRIM(注册资本) = ''
            THEN NULL
        WHEN regexp_matches(注册资本, '美元') THEN 'USD'
        WHEN regexp_matches(注册资本, '港元|港币') THEN 'HKD'
        WHEN regexp_matches(注册资本, '欧元') THEN 'EUR'
        WHEN regexp_matches(注册资本, '英镑') THEN 'GBP'
        WHEN regexp_matches(注册资本, '日元') THEN 'JPY'
        ELSE 'CNY'
    END AS reg_capital_currency

FROM "工商信息"
"""

create_view("工商信息_clean", REG_CAP_SQL)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  年报社保财报信息  →  insured_n, total_assets_wan, net_profit_wan, total_tax_wan
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] 年报社保财报信息 → insured_n / total_assets_wan / net_profit_wan / total_tax_wan")

# --- 参保人数 patterns:
#   "0人"  "0 人"  "  0 人"   → strip 人 + whitespace → int
#   "0"    "1"    "2"        → bare int
#   "人"   "-"               → NULL (malformed)

# --- 资产总额 / 净利润 / 纳税总额 patterns (same structure):
#   "企业选择不公示"  and variants → NULL (disclosure opt-out)
#   "农民专业合作社选择不公示"      → NULL
#   "个体户选择不公示"             → NULL
#   "个体工商户选择不公示"          → NULL
#   "农专社选择不公示"              → NULL
#   "选择不公示"  "不公示"         → NULL
#   "无"                           → NULL
#   "None万元"                     → NULL
#   "  企业选择不公示"              → NULL (leading whitespace variant)
#   " 万元"  "万元"  (no number)   → NULL
#   "1（万元）"  "1(万元)"         → 1 万元
#   "X万元"                        → X
#   "X 万元"  "X.Y 万元"           → X
#   "0.000000万元"                  → 0
#   "X"  (bare number)              → interpret as 万元
#   negative values are valid (净利润 can be negative)

NULL_KEYWORDS = """
        regexp_matches(val, '选择不公示|不公示|农民专业合作社|个体户|
                个体工商户|农专社|^无$|None万元|^\\s*$|^\\s+万元$|^万元$')
        OR TRIM(val) IN ('无', 'None万元', '万元', ' 万元', '万人民币')
"""

ANNUAL_SQL = r"""
SELECT
    *,

    -- ── 参保人数 → insured_n (integer) ──────────────────────────────
    CASE
        WHEN 参保人数 IS NULL OR 参保人数 = '--'
            THEN NULL
        WHEN TRIM(参保人数) = '人' OR TRIM(参保人数) = '-'
            THEN NULL
        -- strip optional leading spaces, trailing "人" with optional space
        WHEN regexp_matches(参保人数, '^\s*([0-9]+)\s*人?\s*$')
            THEN TRY_CAST(
                regexp_extract(参保人数, '([0-9]+)', 1)
            AS INTEGER)
        ELSE NULL
    END AS insured_n,

    -- ── 资产总额 → total_assets_wan (float, 万元) ───────────────────
    CASE
        WHEN 资产总额 IS NULL OR 资产总额 = '--'
            THEN NULL
        WHEN regexp_matches(TRIM(资产总额),
            '选择不公示|不公示|农民专业合作社|个体户|个体工商户|农专社|None万元')
            THEN NULL
        WHEN TRIM(资产总额) IN ('无', '万元', ' 万元', '万人民币', '1（万元）')
            THEN CASE WHEN TRIM(资产总额) = '1（万元）' THEN 1.0 ELSE NULL END
        WHEN regexp_matches(资产总额, '1[（(]万元[)）]')
            THEN 1.0
        -- "X万元" or "X 万元" or "X.Y万元"
        WHEN regexp_matches(资产总额, '(-?[0-9]+(?:\.[0-9]+)?)\s*万元?')
            THEN TRY_CAST(
                regexp_extract(资产总额, '(-?[0-9]+(?:\.[0-9]+)?)\s*万元?', 1)
            AS DOUBLE)
        -- bare number
        WHEN regexp_matches(资产总额, '^-?[0-9]+(?:\.[0-9]+)?$')
            THEN TRY_CAST(TRIM(资产总额) AS DOUBLE)
        ELSE NULL
    END AS total_assets_wan,

    -- ── 净利润 → net_profit_wan (float, 万元) ───────────────────────
    CASE
        WHEN 净利润 IS NULL OR 净利润 = '--'
            THEN NULL
        WHEN regexp_matches(TRIM(净利润),
            '选择不公示|不公示|农民专业合作社|个体户|个体工商户|农专社|None万元')
            THEN NULL
        WHEN TRIM(净利润) IN ('无', '万元', ' 万元', '万人民币', '万人民币')
            THEN NULL
        WHEN regexp_matches(净利润, '万人民币$')
            THEN TRY_CAST(
                regexp_extract(净利润, '(-?[0-9]+(?:\.[0-9]+)?)\s*万', 1)
            AS DOUBLE)
        WHEN regexp_matches(净利润, '(-?[0-9]+(?:\.[0-9]+)?)\s*万元?')
            THEN TRY_CAST(
                regexp_extract(净利润, '(-?[0-9]+(?:\.[0-9]+)?)\s*万元?', 1)
            AS DOUBLE)
        WHEN regexp_matches(净利润, '^-?[0-9]+(?:\.[0-9]+)?$')
            THEN TRY_CAST(TRIM(净利润) AS DOUBLE)
        ELSE NULL
    END AS net_profit_wan,

    -- ── 纳税总额 → total_tax_wan (float, 万元) ──────────────────────
    CASE
        WHEN 纳税总额 IS NULL OR 纳税总额 = '--'
            THEN NULL
        WHEN regexp_matches(TRIM(纳税总额),
            '选择不公示|不公示|农民专业合作社|个体户|个体工商户|农专社|None万元')
            THEN NULL
        WHEN TRIM(纳税总额) IN ('无', '万元', ' 万元', '万人民币', '不公示', '选择不公示')
            THEN NULL
        WHEN regexp_matches(纳税总额, '万人民币$')
            THEN TRY_CAST(
                regexp_extract(纳税总额, '([0-9]+(?:\.[0-9]+)?)\s*万', 1)
            AS DOUBLE)
        WHEN regexp_matches(纳税总额, '([0-9]+(?:\.[0-9]+)?)\s*万元?')
            THEN TRY_CAST(
                regexp_extract(纳税总额, '([0-9]+(?:\.[0-9]+)?)\s*万元?', 1)
            AS DOUBLE)
        WHEN regexp_matches(纳税总额, '^[0-9]+(?:\.[0-9]+)?$')
            THEN TRY_CAST(TRIM(纳税总额) AS DOUBLE)
        ELSE NULL
    END AS total_tax_wan

FROM "年报社保财报信息"
"""

create_view("年报社保财报信息_clean", ANNUAL_SQL)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  股东信息  →  share_pct, subscribed_wan, paid_wan  (+ currency flags)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] 股东信息 → share_pct / subscribed_wan / paid_wan")

# 持股比例: always "XX.XXXX%"  → strip % → float
# 认缴 / 实缴:  "10.0万人民币"  "0.0万美元"  "0.0万港元"  → float 万 + currency

def currency_case(col: str) -> str:
    """Generate a CASE expression for currency detection."""
    return f"""
        CASE
            WHEN {col} IS NULL OR {col} = '--' THEN NULL
            WHEN regexp_matches({col}, '美元') THEN 'USD'
            WHEN regexp_matches({col}, '港元|港币') THEN 'HKD'
            WHEN regexp_matches({col}, '欧元') THEN 'EUR'
            WHEN regexp_matches({col}, '英镑') THEN 'GBP'
            WHEN regexp_matches({col}, '日元') THEN 'JPY'
            ELSE 'CNY'
        END"""

def wan_amount_case(col: str) -> str:
    """Generate a CASE expression to extract 万-denominated float."""
    return f"""
        CASE
            WHEN {col} IS NULL OR {col} = '--' THEN NULL
            WHEN regexp_matches({col}, '([0-9]+(?:\\.[0-9]+)?)万')
                THEN TRY_CAST(
                    regexp_extract({col}, '([0-9]+(?:\\.[0-9]+)?)万', 1)
                AS DOUBLE)
            WHEN regexp_matches({col}, '^[0-9]+(?:\\.[0-9]+)?$')
                THEN TRY_CAST(TRIM({col}) AS DOUBLE)
            ELSE NULL
        END"""

SHAREHOLDER_SQL = f"""
SELECT
    *,
    -- 持股比例: strip trailing %, parse float
    CASE
        WHEN 持股比例 IS NULL OR 持股比例 = '--'
            THEN NULL
        WHEN regexp_matches(持股比例, '([0-9]+(?:\\.[0-9]+)?)%')
            THEN TRY_CAST(
                regexp_extract(持股比例, '([0-9]+(?:\\.[0-9]+)?)%', 1)
            AS DOUBLE)
        WHEN regexp_matches(持股比例, '^[0-9]+(?:\\.[0-9]+)?$')
            THEN TRY_CAST(持股比例 AS DOUBLE)
        ELSE NULL
    END AS share_pct,

    {wan_amount_case('认缴')} AS subscribed_wan,
    {currency_case('认缴')}   AS subscribed_currency,

    {wan_amount_case('实缴')} AS paid_wan,
    {currency_case('实缴')}   AS paid_currency

FROM "股东信息"
"""

create_view("股东信息_clean", SHAREHOLDER_SQL)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  股权出质.出质股权数额  →  pledge_amount (float), pledge_unit
# ══════════════════════════════════════════════════════════════════════════════
#
# Patterns:
#   "1000.0万元"  → amount=1000, unit='万元'
#   "1000.0万股"  → amount=1000, unit='万股'
#   "100万元"     → 100, '万元'
#   "20万股"      → 20,  '万股'
#   bare number   → amount=X, unit='元' (raw count, usually small)

print("\n[5] 股权出质.出质股权数额 → pledge_amount / pledge_unit")

PLEDGE_SQL = r"""
SELECT
    *,
    CASE
        WHEN 出质股权数额 IS NULL OR 出质股权数额 = '--'
            THEN NULL
        WHEN regexp_matches(出质股权数额, '([0-9]+(?:\.[0-9]+)?)万[元股]?')
            THEN TRY_CAST(
                regexp_extract(出质股权数额, '([0-9]+(?:\.[0-9]+)?)万', 1)
            AS DOUBLE)
        WHEN regexp_matches(出质股权数额, '^[0-9]+(?:\.[0-9]+)?$')
            THEN TRY_CAST(出质股权数额 AS DOUBLE)
        ELSE NULL
    END AS pledge_amount,

    CASE
        WHEN 出质股权数额 IS NULL OR 出质股权数额 = '--'
            THEN NULL
        WHEN regexp_matches(出质股权数额, '万股') THEN '万股'
        WHEN regexp_matches(出质股权数额, '万元') THEN '万元'
        WHEN regexp_matches(出质股权数额, '万')   THEN '万元'   -- default if just 万
        WHEN regexp_matches(出质股权数额, '^[0-9]+(?:\.[0-9]+)?$') THEN '元'
        ELSE NULL
    END AS pledge_unit

FROM "股权出质"
"""

create_view("股权出质_clean", PLEDGE_SQL)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Validation")
print("=" * 65)

lines += ["\n---\n## Validation Results\n"]

checks = [
    # (label, SQL returning (pass_count, fail_count or value))
    (
        "招聘_clean — salary_type distribution (sample 10M)",
        """
        SELECT salary_type, COUNT(*) AS n
        FROM "招聘_clean"
        USING SAMPLE 10000000
        GROUP BY 1 ORDER BY 2 DESC
        """
    ),
    (
        "招聘_clean — salary_lo sanity (range rows: lo < hi)",
        """
        SELECT
            COUNT(*) FILTER(WHERE salary_type='range'
                AND salary_lo IS NOT NULL AND salary_hi IS NOT NULL) AS both_present,
            COUNT(*) FILTER(WHERE salary_type='range'
                AND salary_lo IS NOT NULL AND salary_hi IS NOT NULL
                AND salary_lo < salary_hi) AS lo_lt_hi,
            COUNT(*) FILTER(WHERE salary_type='range'
                AND salary_lo IS NOT NULL AND salary_hi IS NOT NULL
                AND salary_lo >= salary_hi) AS lo_gte_hi_warn
        FROM "招聘_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "工商信息_clean — reg_capital_wan null rate",
        """
        SELECT
            COUNT(*) AS total,
            COUNT(reg_capital_wan) AS non_null,
            ROUND(COUNT(reg_capital_wan)*100.0/COUNT(*),2) AS pct_parsed
        FROM "工商信息_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "工商信息_clean — currency distribution",
        """
        SELECT reg_capital_currency, COUNT(*) AS n
        FROM "工商信息_clean"
        WHERE reg_capital_wan IS NOT NULL
        USING SAMPLE 5000000
        GROUP BY 1 ORDER BY 2 DESC
        """
    ),
    (
        "年报_clean — insured_n: parsed vs null (sample)",
        """
        SELECT
            COUNT(*) AS total,
            COUNT(insured_n) AS non_null,
            ROUND(COUNT(insured_n)*100.0/COUNT(*),2) AS pct_parsed,
            MIN(insured_n) AS min_val,
            MAX(insured_n) AS max_val
        FROM "年报社保财报信息_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "年报_clean — total_assets_wan: parsed vs null (sample)",
        """
        SELECT
            COUNT(*) AS total,
            COUNT(total_assets_wan) AS non_null,
            ROUND(COUNT(total_assets_wan)*100.0/COUNT(*),2) AS pct_parsed
        FROM "年报社保财报信息_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "年报_clean — net_profit_wan: negative check",
        """
        SELECT
            COUNT(*) FILTER(WHERE net_profit_wan < 0) AS negative_count,
            COUNT(net_profit_wan) AS total_parsed,
            MIN(net_profit_wan) AS min_val,
            MAX(net_profit_wan) AS max_val
        FROM "年报社保财报信息_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "股东信息_clean — share_pct range check",
        """
        SELECT
            COUNT(*) AS total,
            COUNT(share_pct) AS parsed,
            COUNT(*) FILTER(WHERE share_pct < 0 OR share_pct > 100) AS out_of_range
        FROM "股东信息_clean"
        USING SAMPLE 5000000
        """
    ),
    (
        "股东信息_clean — subscribed_currency distribution",
        """
        SELECT subscribed_currency, COUNT(*) AS n
        FROM "股东信息_clean"
        WHERE subscribed_wan IS NOT NULL
        USING SAMPLE 5000000
        GROUP BY 1 ORDER BY 2 DESC
        """
    ),
    (
        "股权出质_clean — pledge_unit distribution",
        """
        SELECT pledge_unit, COUNT(*) AS n
        FROM "股权出质_clean"
        GROUP BY 1 ORDER BY 2 DESC
        """
    ),
]

for label, sql in checks:
    t0 = time.time()
    print(f"\n  [{label}]")
    try:
        rows = val_query(sql)
        elapsed = time.time() - t0
        lines += [f"\n### {label}\n```"]
        for r in rows:
            row_str = "  ".join(str(x) for x in r)
            print(f"    {row_str}")
            lines.append(f"  {row_str}")
        lines += [f"```\n_({elapsed:.1f}s)_"]
        print(f"    ({elapsed:.1f}s)")
    except Exception as e:
        print(f"    ERROR: {e}")
        lines += [f"\n### {label}\n```\nERROR: {e}\n```"]

# ══════════════════════════════════════════════════════════════════════════════
# WRITE REPORT
# ══════════════════════════════════════════════════════════════════════════════
lines += ["\n---\n## Clean Views Created\n"]
for vname in ["招聘_clean", "工商信息_clean", "年报社保财报信息_clean",
              "股东信息_clean", "股权出质_clean"]:
    lines.append(f"- `{vname}`")

lines += [
    "\n\n## New Columns Summary\n",
    "| View | New Columns |\n|------|-------------|",
    "| 招聘_clean | salary_lo (元/月), salary_hi (元/月), salary_type |",
    "| 工商信息_clean | reg_capital_wan (万元), reg_capital_currency |",
    "| 年报社保财报信息_clean | insured_n (人), total_assets_wan (万元), net_profit_wan (万元), total_tax_wan (万元) |",
    "| 股东信息_clean | share_pct (%), subscribed_wan (万元), subscribed_currency, paid_wan (万元), paid_currency |",
    "| 股权出质_clean | pledge_amount, pledge_unit (元/万元/万股) |",
]

OUT.write_text("\n".join(lines), encoding="utf-8")
con.close()

print(f"\n{'=' * 65}")
print(f"✅  All clean views created in enterprise.duckdb")
print(f"✅  Report written → {OUT}")
print(f"{'=' * 65}")
print("""
Usage examples
──────────────
import duckdb
con = duckdb.connect(r'I:\\posting_2026\\enterprise.duckdb')

# salary distribution for tech jobs in Beijing
con.sql(\"\"\"
    SELECT
        ROUND(salary_lo/1000)*1000 AS lo_bucket,
        COUNT(*) AS n
    FROM 招聘_clean
    WHERE 工作名称 LIKE '%工程师%'
      AND 公司所在区域 LIKE '%北京%'
      AND salary_type = 'range'
      AND salary_lo IS NOT NULL
    GROUP BY 1 ORDER BY 1
\"\"\").show()

# registered capital histogram (CNY companies only)
con.sql(\"\"\"
    SELECT
        ROUND(LOG10(reg_capital_wan + 0.001)) AS log10_capital_bucket,
        COUNT(*) AS n
    FROM 工商信息_clean
    WHERE reg_capital_currency = 'CNY'
      AND reg_capital_wan > 0
    GROUP BY 1 ORDER BY 1
\"\"\").show()
""")
