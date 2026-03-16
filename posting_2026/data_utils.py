# -*- coding: utf-8 -*-
"""
data_utils.py  —  Unified data access layer for posting_2026
=============================================================
Provides:
  • Table registry          — canonical paths for all 15 Parquet tables
  • query()                 — run any DuckDB SQL, return Polars / pandas / list
  • count()                 — fast row count without full scan
  • sample()                — stratified or random sample → Polars DataFrame
  • load_table()            — load a whole (small) table into Polars
  • filter_by_company()     — filter any table by a list of 公司ID
  • filter_by_date()        — filter tables that have a date column
  • join_posting_to_firm()  — 招聘 ⋈ 工商信息 with optional extra fields
  • save_pickle()           — save a DataFrame to the Pickle folder
  • load_pickle()           — load a pickle back to Polars / pandas

Usage example
-------------
    from data_utils import query, sample, join_posting_to_firm, save_pickle

    # Count active postings in 2024
    n = query("SELECT COUNT(*) FROM {招聘} WHERE 发布日期 >= '2024-01-01'", scalar=True)

    # Sample 500K rows from 招聘 for model fine-tuning
    df = sample("招聘", n=500_000, seed=42)
    save_pickle(df, "招聘_sample_500k")

    # Join postings with firm info
    df = join_posting_to_firm(
        posting_cols=["公司ID", "工作名称", "工作薪酬", "发布日期"],
        firm_cols=["省份", "一级行业", "经营状态", "注册资本"],
        where="z.发布日期 >= '2023-01-01' AND g.经营状态 LIKE '%在营%'"
    )
"""

import sys, re, time, pickle
from pathlib import Path
from typing import Union, Optional

import duckdb
import polars as pl

from config import PARQUET_DIR, PICKLE_DIR, TMP_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Table registry ─────────────────────────────────────────────────────────────
def _glob(name: str) -> str:
    """Return the DuckDB-readable path (glob) for a table by name."""
    d = PARQUET_DIR / name
    f = PARQUET_DIR / f"{name}.parquet"
    if d.is_dir():
        return (d / "*.parquet").as_posix()
    if f.exists():
        return f.as_posix()
    raise FileNotFoundError(f"Table '{name}' not found in {PARQUET_DIR}")

# All 15 tables — keys used as placeholders in SQL strings  {tablename}
TABLES = {
    "招聘":           _glob("招聘"),
    "工商信息":        _glob("工商信息"),
    "工商变更":        _glob("工商变更"),
    "年报社保财报信息": _glob("年报社保财报信息"),
    "主要人员":        _glob("主要人员"),
    "股东信息":        _glob("股东信息"),
    "对外投资":        _glob("对外投资"),
    "分支机构":        _glob("分支机构"),
    "产品许可":        _glob("产品许可"),
    "动产抵押":        _glob("动产抵押"),
    "动产抵押-抵押物":  _glob("动产抵押-抵押物"),
    "动产抵押-抵押人":  _glob("动产抵押-抵押人"),
    "动产抵押-抵押变更": _glob("动产抵押-抵押变更"),
    "股权出质":        _glob("股权出质"),
    "融资历史":        _glob("融资历史"),
}

def _resolve_sql(sql: str) -> str:
    """Replace {tablename} placeholders with actual read_parquet(...) expressions."""
    for name, path in TABLES.items():
        placeholder = "{" + name + "}"
        if placeholder in sql:
            sql = sql.replace(placeholder, f"read_parquet('{path}')")
    return sql

def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-process DuckDB connection with tuned memory settings."""
    con = duckdb.connect()
    con.execute("SET memory_limit='96GB'")        # leave headroom for OS
    con.execute("SET threads=8")                  # adjust to your CPU
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    return con

# ── Core: query() ──────────────────────────────────────────────────────────────
def query(
    sql: str,
    return_type: str = "polars",   # "polars" | "pandas" | "list" | "arrow"
    scalar: bool = False,
    verbose: bool = True,
) -> Union[pl.DataFrame, list, object]:
    """
    Run a DuckDB SQL query against the Parquet tables.

    Use {tablename} placeholders instead of read_parquet(...) calls:
        query("SELECT * FROM {招聘} LIMIT 10")

    Parameters
    ----------
    sql         : SQL string, may contain {tablename} placeholders
    return_type : "polars" (default), "pandas", "list", or "arrow"
    scalar      : if True, return the first cell only (e.g. COUNT result)
    verbose     : print elapsed time

    Returns
    -------
    Polars DataFrame / pandas DataFrame / list of tuples / scalar
    """
    sql = _resolve_sql(sql)
    con = _get_con()
    t0  = time.time()
    rel = con.sql(sql)
    if scalar:
        result = rel.fetchone()[0]
    elif return_type == "polars":
        result = rel.pl()
    elif return_type == "pandas":
        result = rel.df()
    elif return_type == "arrow":
        result = rel.arrow()
    else:
        result = rel.fetchall()
    elapsed = time.time() - t0
    if verbose:
        n = len(result) if hasattr(result, "__len__") and not scalar else ""
        print(f"  query done in {elapsed:.1f}s  rows={n if n != '' else result}")
    return result

# ── count() ────────────────────────────────────────────────────────────────────
def count(table: str, where: str = "") -> int:
    """Fast row count for a table, with optional WHERE clause."""
    w   = f"WHERE {where}" if where else ""
    sql = f"SELECT COUNT(*) FROM {{{table}}} {w}"
    return query(sql, scalar=True, verbose=False)

# ── sample() ───────────────────────────────────────────────────────────────────
def sample(
    table: str,
    n: int = 100_000,
    where: str = "",
    cols: Optional[list] = None,
    seed: int = 42,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Draw a random sample of n rows from a table.

    Uses DuckDB's USING SAMPLE clause (reservoir sampling) — very fast
    even on 600M-row tables.

    Parameters
    ----------
    table  : table name (key in TABLES dict)
    n      : number of rows to return
    where  : optional SQL WHERE filter applied before sampling
    cols   : list of column names to select (None = all)
    seed   : random seed for reproducibility
    """
    col_str = ", ".join(cols) if cols else "*"
    w       = f"WHERE {where}" if where else ""
    sql = f"""
        SELECT {col_str}
        FROM {{{table}}} {w}
        USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})
    """
    if verbose:
        print(f"  Sampling {n:,} rows from '{table}' (seed={seed}) ...")
    return query(sql, return_type="polars", verbose=verbose)

# ── load_table() ───────────────────────────────────────────────────────────────
def load_table(
    table: str,
    cols: Optional[list] = None,
    where: str = "",
) -> pl.DataFrame:
    """
    Load an entire table (or filtered subset) into a Polars DataFrame.
    Only use for tables < 20 GB. For larger tables, use sample() or query().
    """
    col_str = ", ".join(cols) if cols else "*"
    w       = f"WHERE {where}" if where else ""
    sql = f"SELECT {col_str} FROM {{{table}}} {w}"
    print(f"  Loading '{table}' ...")
    return query(sql, return_type="polars", verbose=True)

# ── filter_by_company() ────────────────────────────────────────────────────────
def filter_by_company(
    table: str,
    company_ids: list,
    cols: Optional[list] = None,
    id_col: str = "公司ID",
) -> pl.DataFrame:
    """
    Filter any table to rows matching a list of 公司ID values.

    Parameters
    ----------
    table       : table name
    company_ids : list of company ID strings/ints
    cols        : columns to return (None = all)
    id_col      : name of the company ID column (default '公司ID')
    """
    ids_str = ", ".join(f"'{i}'" for i in company_ids)
    col_str = ", ".join(cols) if cols else "*"
    sql = f"""
        SELECT {col_str}
        FROM {{{table}}}
        WHERE {id_col} IN ({ids_str})
    """
    print(f"  Filtering '{table}' by {len(company_ids):,} company IDs ...")
    return query(sql, return_type="polars", verbose=True)

# ── filter_by_date() ───────────────────────────────────────────────────────────
def filter_by_date(
    table: str,
    date_col: str,
    start: Optional[str] = None,
    end:   Optional[str] = None,
    cols: Optional[list] = None,
) -> pl.DataFrame:
    """
    Filter a table by a date range.

    Parameters
    ----------
    table    : table name
    date_col : name of the date column
    start    : start date string 'YYYY-MM-DD' (inclusive, optional)
    end      : end date string 'YYYY-MM-DD'   (inclusive, optional)
    cols     : columns to return (None = all)

    Example
    -------
        df = filter_by_date("招聘", "发布日期", start="2023-01-01", end="2024-12-31")
    """
    conditions = []
    if start:
        conditions.append(f"{date_col} >= '{start}'")
    if end:
        conditions.append(f"{date_col} <= '{end}'")
    where = " AND ".join(conditions) if conditions else "1=1"
    col_str = ", ".join(cols) if cols else "*"
    sql = f"""
        SELECT {col_str}
        FROM {{{table}}}
        WHERE {where}
    """
    print(f"  Filtering '{table}' by {date_col} [{start} → {end}] ...")
    return query(sql, return_type="polars", verbose=True)

# ── join_posting_to_firm() ─────────────────────────────────────────────────────
def join_posting_to_firm(
    posting_cols: Optional[list] = None,
    firm_cols:    Optional[list] = None,
    where:        str = "",
    n_sample:     Optional[int] = None,
    seed:         int = 42,
) -> pl.DataFrame:
    """
    Join 招聘 (job postings) to 工商信息 (firm registry) on 公司ID.

    Parameters
    ----------
    posting_cols : columns from 招聘 to include (None = all)
    firm_cols    : columns from 工商信息 to include (None = common useful ones)
    where        : extra WHERE clause applied before the join
    n_sample     : if set, sample this many rows from 招聘 before joining
    seed         : random seed for sampling

    Common 招聘 columns:
        公司ID, 工作名称, 工作薪酬, 发布日期, 工作城市, 工作经验,
        最低学历, 数据来源

    Common 工商信息 columns:
        省份, 城市, 一级行业, 二级行业, 经营状态, 注册资本,
        成立日期, 企业类型, 员工人数
    """
    default_firm_cols = [
        "g.省份", "g.城市", "g.一级行业", "g.二级行业",
        "g.经营状态", "g.注册资本", "g.企业类型"
    ]
    p_cols = ["z." + c for c in posting_cols] if posting_cols else ["z.*"]
    f_cols = ["g." + c for c in firm_cols]    if firm_cols    else default_firm_cols
    col_str = ", ".join(p_cols + f_cols)

    w = f"WHERE {where}" if where else ""

    if n_sample:
        posting_src = f"""(
            SELECT * FROM {{招聘}}
            USING SAMPLE reservoir({n_sample} ROWS) REPEATABLE ({seed})
        )"""
    else:
        posting_src = "{招聘}"

    sql = f"""
        SELECT {col_str}
        FROM {posting_src} z
        JOIN {{工商信息}} g ON z.公司ID = g.公司ID
        {w}
    """
    label = f"招聘({n_sample:,} sample)" if n_sample else "招聘(all)"
    print(f"  Joining {label} ⋈ 工商信息 ...")
    return query(sql, return_type="polars", verbose=True)

# ── save_pickle() / load_pickle() ─────────────────────────────────────────────
def save_pickle(
    df: Union[pl.DataFrame, object],
    name: str,
    as_pandas: bool = False,
) -> Path:
    """
    Save a Polars (or pandas) DataFrame to the Pickle folder.

    Parameters
    ----------
    df        : Polars or pandas DataFrame (or any picklable object)
    name      : filename without extension  (e.g. "招聘_sample_500k")
    as_pandas : if True, convert Polars → pandas before pickling

    Returns
    -------
    Path to the saved pickle file
    """
    if as_pandas and isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    path = PICKLE_DIR / f"{name}.pkl"
    t0   = time.time()
    with open(path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.time() - t0
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved → {path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return path

def load_pickle(
    name: str,
    to_polars: bool = True,
) -> Union[pl.DataFrame, object]:
    """
    Load a pickle from the Pickle folder.

    Parameters
    ----------
    name      : filename without extension (e.g. "招聘_sample_500k")
    to_polars : if True and the object is a pandas DataFrame, convert to Polars
    """
    path = PICKLE_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    t0 = time.time()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    elapsed = time.time() - t0
    if to_polars:
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                obj = pl.from_pandas(obj)
        except ImportError:
            pass
    print(f"  Loaded ← {path}  ({elapsed:.1f}s)")
    return obj

# ── Convenience: list available tables & pickles ───────────────────────────────
def list_tables() -> None:
    """Print all available Parquet tables with their sizes."""
    print(f"\n{'Table':<30} {'Path':<55} {'Size GB':>8}")
    print("-" * 95)
    for name, path in TABLES.items():
        p = Path(path.replace("/*.parquet", ""))
        if p.is_dir():
            sz = sum(f.stat().st_size for f in p.glob("*.parquet")) / 1e9
        else:
            sz = p.stat().st_size / 1e9 if p.exists() else 0
        print(f"  {name:<28} {path:<55} {sz:>7.2f}")

def list_pickles() -> None:
    """Print all saved pickle files with sizes."""
    pkls = sorted(PICKLE_DIR.glob("*.pkl"))
    if not pkls:
        print("  No pickle files found.")
        return
    print(f"\n{'Name':<45} {'Size MB':>10}")
    print("-" * 58)
    for p in pkls:
        print(f"  {p.stem:<45} {p.stat().st_size/1e6:>9.1f}")

# ── Quick self-test (run this file directly to verify) ────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("data_utils.py — self-test")
    print("=" * 60)

    list_tables()

    print("\n[1] Count rows in 招聘 ...")
    n = query("SELECT COUNT(*) FROM {招聘}", scalar=True)
    print(f"    招聘 rows: {n:,}")

    print("\n[2] Sample 1,000 rows from 融资历史 ...")
    df = sample("融资历史", n=1_000, seed=0)
    print(df.head(3))

    print("\n[3] filter_by_date: 招聘 in 2024 ...")
    df2 = filter_by_date("招聘", "发布日期", start="2024-01-01", end="2024-12-31",
                          cols=["公司ID", "工作名称", "工作薪酬", "发布日期"])
    print(f"    rows: {len(df2):,}")
    print(df2.head(3))

    print("\n[4] join_posting_to_firm (sample 5,000) ...")
    df3 = join_posting_to_firm(
        posting_cols=["公司ID", "工作名称", "工作薪酬", "发布日期"],
        firm_cols=["省份", "一级行业", "经营状态"],
        where="z.发布日期 >= '2024-01-01'",
        n_sample=5_000,
    )
    print(df3.head(3))

    print("\n[5] save_pickle & load_pickle ...")
    path = save_pickle(df3, "_test_join_sample")
    df4  = load_pickle("_test_join_sample")
    print(f"    Roundtrip OK — shape {df4.shape}")
    path.unlink()   # clean up test pickle

    print("\n✅ All self-tests passed!")
    list_pickles()
