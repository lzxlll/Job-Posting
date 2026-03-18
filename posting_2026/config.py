# -*- coding: utf-8 -*-
"""
config.py  —  Central path configuration for posting_2026 pipeline
===================================================================
Change BASE_DIR here if you ever move the data drive or folder.
All scripts import from this module.
"""

from pathlib import Path

# ── Change this one line to relocate everything ────────────────────────────────
BASE_DIR = Path(r"I:\posting_2026")

# ── Derived paths (do not edit) ───────────────────────────────────────────────
PARQUET_DIR = BASE_DIR / "parquet"
PICKLE_DIR  = BASE_DIR / "Pickle"
DB_PATH     = BASE_DIR / "enterprise.duckdb"
TMP_DIR     = BASE_DIR / "duckdb_tmp"
RAW_DIR     = BASE_DIR / "定制数据"
