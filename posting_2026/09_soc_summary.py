# -*- coding: utf-8 -*-
"""
09_soc_summary.py — Summary statistics on 招聘_soc_labels.parquet
=================================================================
Queries the labeled subsample (575,707 rows) produced by step 6 and
prints a comprehensive profile: confidence distribution, top/bottom SOC
codes, temporal trends, and company coverage.

Usage
-----
    python 09_soc_summary.py
"""

import sys
from pathlib import Path

PARQUET  = Path(r"I:\posting_2026\parquet\招聘_soc_labels.parquet")
DB_PATH  = Path(r"I:\posting_2026\enterprise.duckdb")

# ── helpers ────────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def run(con, sql: str, show: bool = True):
    rel = con.sql(sql)
    if show:
        rel.show(max_rows=40)
    return rel

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import duckdb

    con = duckdb.connect(str(DB_PATH))

    # Ensure view is live
    con.execute(f"""
        CREATE OR REPLACE VIEW 招聘_labeled AS
        SELECT * FROM read_parquet('{PARQUET.as_posix()}')
    """)

    # ── 1. Overview ────────────────────────────────────────────────────────────
    banner("1. OVERVIEW")
    run(con, """
        SELECT
            COUNT(*)                                  AS total_rows,
            COUNT(DISTINCT soc_code_pred)             AS unique_soc_codes,
            COUNT(DISTINCT 公司ID)                    AS unique_companies,
            ROUND(MIN(soc_prob)*100, 1)               AS min_conf_pct,
            ROUND(MAX(soc_prob)*100, 1)               AS max_conf_pct,
            ROUND(AVG(soc_prob)*100, 2)               AS avg_conf_pct,
            ROUND(MEDIAN(soc_prob)*100, 2)            AS median_conf_pct,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY soc_prob)*100, 1) AS p5_conf_pct,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY soc_prob)*100, 1) AS p25_conf_pct,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY soc_prob)*100, 1) AS p75_conf_pct,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY soc_prob)*100, 1) AS p95_conf_pct
        FROM 招聘_labeled
    """)

    # ── 2. Confidence band distribution ────────────────────────────────────────
    banner("2. CONFIDENCE BANDS")
    run(con, """
        SELECT
            CASE
                WHEN soc_prob < 0.50 THEN '< 50%'
                WHEN soc_prob < 0.70 THEN '50–70%'
                WHEN soc_prob < 0.80 THEN '70–80%'
                WHEN soc_prob < 0.90 THEN '80–90%'
                ELSE                      '≥ 90%'
            END                             AS confidence_band,
            COUNT(*)                        AS n_rows,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM 招聘_labeled
        GROUP BY 1
        ORDER BY MIN(soc_prob)
    """)

    # ── 3. Top-20 SOC codes ────────────────────────────────────────────────────
    banner("3. TOP-20 SOC CODES BY VOLUME")
    run(con, """
        SELECT
            soc_code_pred,
            COUNT(*)                           AS n_rows,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct,
            ROUND(AVG(soc_prob)*100, 2)        AS avg_conf_pct,
            ROUND(MEDIAN(soc_prob)*100, 2)     AS median_conf_pct
        FROM 招聘_labeled
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
    """)

    # ── 4. Bottom-20 SOC codes (rarest) ───────────────────────────────────────
    banner("4. BOTTOM-20 SOC CODES BY VOLUME")
    run(con, """
        SELECT
            soc_code_pred,
            COUNT(*)                           AS n_rows,
            ROUND(AVG(soc_prob)*100, 2)        AS avg_conf_pct
        FROM 招聘_labeled
        GROUP BY 1
        ORDER BY 2 ASC
        LIMIT 20
    """)

    # ── 5. Lowest-confidence SOC codes (avg) ─────────────────────────────────
    banner("5. LEAST CONFIDENT SOC CODES (min 100 rows)")
    run(con, """
        SELECT
            soc_code_pred,
            COUNT(*)                          AS n_rows,
            ROUND(AVG(soc_prob)*100, 2)       AS avg_conf_pct,
            ROUND(MEDIAN(soc_prob)*100, 2)    AS median_conf_pct
        FROM 招聘_labeled
        GROUP BY 1
        HAVING COUNT(*) >= 100
        ORDER BY avg_conf_pct ASC
        LIMIT 20
    """)

    # ── 6. Highest-confidence SOC codes (avg) ─────────────────────────────────
    banner("6. MOST CONFIDENT SOC CODES (min 100 rows)")
    run(con, """
        SELECT
            soc_code_pred,
            COUNT(*)                          AS n_rows,
            ROUND(AVG(soc_prob)*100, 2)       AS avg_conf_pct,
            ROUND(MEDIAN(soc_prob)*100, 2)    AS median_conf_pct
        FROM 招聘_labeled
        GROUP BY 1
        HAVING COUNT(*) >= 100
        ORDER BY avg_conf_pct DESC
        LIMIT 20
    """)

    # ── 7. Temporal distribution by year ─────────────────────────────────────
    banner("7. BY YEAR")
    run(con, """
        SELECT
            YEAR(TRY_CAST(发布日期 AS DATE))           AS yr,
            COUNT(*)                                   AS n_rows,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
            ROUND(AVG(soc_prob)*100, 2)                AS avg_conf_pct,
            COUNT(DISTINCT soc_code_pred)              AS distinct_soc_codes,
            COUNT(DISTINCT 公司ID)                     AS distinct_companies
        FROM 招聘_labeled
        WHERE 发布日期 IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)

    # ── 8. SOC code concentration ─────────────────────────────────────────────
    banner("8. SOC CODE CONCENTRATION")
    run(con, """
        WITH ranked AS (
            SELECT
                soc_code_pred,
                COUNT(*) AS n
            FROM 招聘_labeled
            GROUP BY 1
        ),
        total AS (SELECT SUM(n) AS t FROM ranked)
        SELECT
            '406 total classes'                              AS note,
            COUNT(*)                                         AS codes_with_any_rows,
            COUNT(*) FILTER (WHERE n >= 100)                 AS codes_ge_100,
            COUNT(*) FILTER (WHERE n >= 1000)                AS codes_ge_1000,
            ROUND(SUM(n) FILTER (WHERE n = (SELECT MAX(n) FROM ranked)) * 100.0
                  / (SELECT t FROM total), 2)                AS top1_share_pct
        FROM ranked
    """)

    # Top-5 share
    run(con, """
        WITH ranked AS (
            SELECT soc_code_pred, COUNT(*) AS n
            FROM 招聘_labeled
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 5
        )
        SELECT
            ROUND(SUM(n) * 100.0 / (SELECT COUNT(*) FROM 招聘_labeled), 2) AS top5_share_pct
        FROM ranked
    """)

    # ── 9. Low-confidence rows sample ─────────────────────────────────────────
    banner("9. LOW-CONFIDENCE ROWS SAMPLE (soc_prob < 0.40, up to 10)")
    run(con, """
        SELECT
            表ID,
            工作名称,
            soc_code_pred,
            ROUND(soc_prob*100, 1) AS conf_pct,
            YEAR(TRY_CAST(发布日期 AS DATE)) AS yr
        FROM 招聘_labeled
        WHERE soc_prob < 0.40
        ORDER BY soc_prob ASC
        LIMIT 10
    """)

    # ── 10. Company coverage ──────────────────────────────────────────────────
    banner("10. COMPANY COVERAGE")
    run(con, """
        SELECT
            COUNT(DISTINCT 公司ID)                         AS companies_in_sample,
            ROUND(AVG(cnt), 1)                             AS avg_postings_per_company,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY cnt) AS median_postings,
            MAX(cnt)                                       AS max_postings
        FROM (
            SELECT 公司ID, COUNT(*) AS cnt
            FROM 招聘_labeled
            GROUP BY 公司ID
        )
    """)

    # Top-10 companies by labeled postings
    run(con, """
        SELECT
            公司ID,
            COUNT(*)                          AS n_postings,
            COUNT(DISTINCT soc_code_pred)     AS distinct_soc_codes,
            ROUND(AVG(soc_prob)*100, 2)       AS avg_conf_pct
        FROM 招聘_labeled
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 10
    """)

    banner("DONE")
    con.close()


if __name__ == "__main__":
    main()
