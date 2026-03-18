# -*- coding: utf-8 -*-
"""
Estimate title-only share using DuckDB TABLESAMPLE directly on parquet.
Opens the DB read-only to avoid lock conflicts.
"""
import duckdb, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARQUET = r"I:\posting_2026\parquet\招聘\data.parquet"

con = duckdb.connect(r"I:\posting_2026\enterprise.duckdb", read_only=True)

# ── 1. TABLESAMPLE 0.2% (~1.3M rows) — fast random rows ─────────────────────
print("Running TABLESAMPLE 0.2% on 招聘 parquet ...")
con.sql(f"""
    SELECT
        COUNT(*)                                                                   AS sample_n,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 职责描述 IS NULL OR 职责描述 = '--')
              / COUNT(*), 2)                                                       AS no_desc_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 职责描述 IS NULL)
              / COUNT(*), 2)                                                       AS desc_null_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 职责描述 = '--')
              / COUNT(*), 2)                                                       AS desc_dash_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 工作名称 IS NULL)
              / COUNT(*), 3)                                                       AS title_null_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE (职责描述 IS NULL OR 职责描述 = '--')
                                          AND 工作名称 IS NOT NULL)
              / COUNT(*), 2)                                                       AS title_only_pct,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 职责描述 IS NOT NULL
                                          AND 职责描述 != '--'
                                          AND 工作名称 IS NOT NULL)
              / COUNT(*), 2)                                                       AS both_present_pct
    FROM read_parquet('{PARQUET}') TABLESAMPLE(0.2 PERCENT)
""").show()

# ── 2. Top 10 desc values among no-desc rows ─────────────────────────────────
print("\nTop 10 职责描述 values in rows where desc IS NULL or = '--' (TABLESAMPLE 0.5%):")
con.sql(f"""
    SELECT 职责描述, COUNT(*) AS n
    FROM read_parquet('{PARQUET}') TABLESAMPLE(0.5 PERCENT)
    WHERE 职责描述 IS NULL OR 职责描述 = '--'
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 10
""").show()

# ── 3. No-desc rate by 数据来源 (platform) ────────────────────────────────────
print("\nNo-desc rate by platform (TABLESAMPLE 0.5%):")
con.sql(f"""
    SELECT
        数据来源,
        COUNT(*)                                                                   AS n,
        ROUND(100.0 * COUNT(*) FILTER (WHERE 职责描述 IS NULL OR 职责描述 = '--')
              / COUNT(*), 1)                                                       AS no_desc_pct
    FROM read_parquet('{PARQUET}') TABLESAMPLE(0.5 PERCENT)
    GROUP BY 1
    ORDER BY n DESC
    LIMIT 15
""").show()

con.close()
