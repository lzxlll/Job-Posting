# Data Pipeline Plan: 2026 Enterprise Dataset (定制数据)

## Project Overview

**Objective**: Build a scalable data pipeline for ~1 TB of Chinese enterprise data, supporting data merging, filtering, descriptive analytics, and ML/fine-tuning workflows.

**Date**: 2026-03-12

---

## Directory Structure

```
I:\posting_2026\                          ← Working data root
├── 定制数据\                              ← Raw CSV files (~977 GB)
├── Pickle\                               ← Generated Pickle files (intermediate storage)
└── parquet\                              ← Parquet files (to be created, ~100-200 GB)

D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\   ← Code & documentation
├── 00_data_pipeline_plan.md              ← This file
├── 01_csv_to_parquet.py                  ← CSV → Parquet conversion script
├── 02_schema_discovery.py                ← Schema & data dictionary generation
└── ...                                   ← Future analysis/modeling scripts
```

---

## Dataset Inventory

| # | File | Size (GB) | Tier | Notes |
|---|------|-----------|------|-------|
| 1 | 招聘.csv | 421.0 | Giant | Job postings — likely NLP target |
| 2 | 交易信息.csv | 208.4 | Giant | Transaction records |
| 3 | 工商变更.csv | 184.0 | Giant | Business registration changes |
| 4 | 年报年报保险信息.csv | 113.2 | Giant | Annual report & insurance info |
| 5 | 股东信息.csv | 24.9 | Large | Shareholder information |
| 6 | 主要人员.csv | 18.6 | Large | Key personnel |
| 7 | 资质投标.csv | 2.6 | Medium | Qualifications & bidding |
| 8 | 收支明细.csv | 2.3 | Medium | Income & expenditure details |
| 9 | 关联投资-投资方.csv | 2.1 | Medium | Related investment — investor side |
| 10 | 品牌商标.csv | 0.43 | Small | Brand & trademarks |
| 11 | 关联投资.csv | 0.33 | Small | Related investments |
| 12 | 股权出质.csv | 0.18 | Small | Equity pledges |
| 13 | 关联投资-投资对象.csv | 0.16 | Small | Related investment — investee |
| 14 | 变更历史.csv | 0.07 | Small | Change history |
| 15 | 关联投资-投资企业.csv | 0.01 | Small | Related investment — enterprises |
| | **Total** | **~977** | | |

---

## Environment

- **Machine**: DESKTOP-M7MRSEF — Intel i9-10980XE, 128 GB RAM, Windows 11 64-bit
- **GPU**: NVIDIA Quadro RTX 5000, 16 GB VRAM, CUDA 12.8 ✅
- **Python**: 3.13.7 at `C:\Python313`
- **Key packages**: `polars`, `duckdb`, `pyarrow`, `connectorx`, `datasets`, `transformers 5.3.0`, `torch 2.10.0+cu128`, `accelerate`
- **Storage**: I:\ drive (HDD, 3.64 TB) for data; D:\ (SSD) for code/Dropbox
- **Cloud**: SQL Server / Azure available but skipped — local DuckDB sufficient

---

## Architecture

```
CSV files (977 GB, slow, untyped, GB18030 encoded)
  │
  ▼  [Step 1: one-time streaming conversion via Polars]
Parquet files (est. 100-200 GB, columnar, compressed, typed)
  │
  ├──▶ DuckDB    — SQL queries directly on Parquet (joins, agg, filters)
  ├──▶ Polars    — Python lazy DataFrame ops (streaming, multi-threaded)
  ├──▶ SQL Server — Shared access, indexed tables for key subsets
  └──▶ ML Pipeline — Filtered subsets → PyTorch / HuggingFace datasets
```

---

## Implementation Steps

### Step 1: CSV → Parquet Conversion (HIGHEST PRIORITY)

**Why Parquet over CSV?**
- 5-10x smaller (compression)
- Column pruning: read only the columns you need
- Predicate pushdown: filter at I/O level
- Type preservation: no re-parsing every time

**Tool**: Polars streaming mode (`scan_csv → sink_parquet`)
- Never loads full file into memory
- Handles GB18030 encoding natively
- Giant files (100 GB+): partition by key column (year/region) into multiple .parquet files

**Execution order**: Small → Medium → Large → Giant (pilot first, scale later)

### Step 2: Schema Discovery & Data Dictionary

- Extract column names, types, cardinality, null rates per table
- Identify join keys across tables (统一社会信用代码? 企业名称?)
- Sample 10,000 rows per file for distribution overview
- Output: Excel or markdown data dictionary

### Step 3: Tool Selection by Task

| Task | Tool | Rationale |
|------|------|-----------|
| Ad-hoc filtering & aggregation | Polars (lazy) | Streaming, never OOM |
| Multi-table joins | DuckDB or SQL Server | SQL-native, reads Parquet directly |
| Descriptive analytics | Polars or DuckDB | Group-by, pivot, window functions |
| ML subset extraction | Polars/DuckDB → subset Parquet → HuggingFace | Filter first, train on small subset |
| SAS workflows | SAS reads Parquet (9.4M6+) or filtered CSV export | Native Parquet support in modern SAS |

### Step 4: DuckDB as Local Query Engine

- Create a persistent DuckDB database at `I:\posting_2026\enterprise.duckdb`
- Register Parquet files as views for convenient SQL access
- Supports joins, CTEs, window functions across all tables
- Spills to disk when exceeding RAM

### Step 5: SQL Server (selective use)

- Bulk-load Small + Medium tables (~8 GB) into SQL Server
- For Giant tables: load only filtered/materialized subsets
- Use for shared team access and scheduled pipelines

### Step 6: ML / Fine-tuning Pipeline

- Filter subsets via DuckDB/Polars (by date, region, industry)
- Export as Parquet subset (1-10 GB)
- Load via HuggingFace `datasets` (supports Parquet streaming)
- Train on GPU (local or Azure ML)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| CSV encoding (GB18030 vs UTF-8) | Test with 变更历史.csv (77 MB) first; specify `encoding="gb18030"` |
| I: drive I/O bottleneck | Move Parquet to SSD if available; partition Giant files |
| Join key ambiguity | Schema discovery must happen before any merge |
| SAS compatibility | SAS 9.4M6+ reads Parquet; fallback to filtered CSV export |
| Conversion time for Giant files | Expect 1-3 hours per Giant file; run overnight if needed |

---

## Progress Log

| Date | Step | Status | Notes |
|------|------|--------|-------|
| 2026-03-12 | Planning | ✅ Done | This document created |
| 2026-03-12 | Environment setup | ✅ Done | Python 3.13.7; installed polars, duckdb, pyarrow, connectorx, datasets |
| 2026-03-12 | Pilot test | ✅ Done | `01_pilot_csv_to_parquet.py` — validated encoding (UTF-8), Parquet write OK |
| 2026-03-12 | Step 1: Parquet conversion | ✅ Done | `02_convert_all_to_parquet.py` — all 15/15 files converted (resume-safe) |
| 2026-03-13 | Step 2: Schema discovery | ✅ Done | `03_schema_discovery.py` → `schema_dictionary.md` — all 15 tables documented |
| 2026-03-13 | Step 3: DuckDB validation | ✅ Done | `04_duckdb_validation.py` → `04_validation_report.md` — all checks passed |
| 2026-03-13 | Data quality investigation | ✅ Done | `对外投资` column names are correct — `认缴投资金额` = amounts, `认缴投资时间` = dates. Initial false alarm was caused by `'--'` (null placeholder) matching `LIKE '%-%-%'` heuristic. No fix needed. |
| 2026-03-13 | Data access utilities | ✅ Done | `data_utils.py` — all self-tests passed |
| 2026-03-13 | Step 4: DuckDB persistent DB | ✅ Done | `05_setup_duckdb.py` — `enterprise.duckdb` created, 15 views + 5 macros (find_company, company_jobs, company_shareholders, company_personnel, company_financials) |
| 2026-03-13 | Step 5: SQL Server | ⏭️ Skipped | Local-only workflow — DuckDB sufficient for all query needs |
| 2026-03-13 | Dirty column profiling | ✅ Done | `06a_profile_dirty_cols.py` → `06a_dirty_col_profile.md` — top-200 values for 10 columns profiled |
| 2026-03-13 | Data cleaning (numeric parsing) | ✅ Done | `07_data_cleaning.py` → 5 clean views in `enterprise.duckdb` (`招聘_clean`, `工商信息_clean`, `年报社保财报信息_clean`, `股东信息_clean`, `股权出质_clean`) + `07_cleaning_report.md` |
| 2026-03-15 | Central path config | ✅ Done | `config.py` created — single `BASE_DIR = I:\posting_2026`; all scripts import from it |
| 2026-03-15 | CUDA GPU setup | ✅ Done | Reinstalled PyTorch 2.10.0+cu128 (was +cpu); installed `accelerate`; verified: CUDA available, Quadro RTX 5000, 16GB VRAM |
| 2026-03-16 | Step 6: ML inference | ✅ Done | `08_bert_inference.py` — SOC code labeling via fine-tuned BERT (406 classes); 1M-row local test; output: `招聘_soc_labels.parquet` + DuckDB view `招聘_labeled` |

---

## Actual Parquet Inventory (as of 2026-03-13)

> Source: 1,050.3 GB CSV → **184.7 GB Parquet (82% reduction)**

| Table | CSV (GB) | Parquet (GB) | Ratio | Rows |
|-------|----------|--------------|-------|------|
| 招聘 | 451.96 | 90.31 | 5.0x | 641,831,221 |
| 工商信息 | 223.76 | 34.53 | 6.5x | 405,180,231 |
| 工商变更 | 197.54 | 34.35 | 5.8x | 601,760,211 |
| 年报社保财报信息 | 121.54 | 13.33 | 9.1x | 727,206,127 |
| 股东信息 | 26.74 | 5.68 | 4.7x | 197,574,397 |
| 主要人员 | 20.01 | 4.71 | 4.2x | 238,180,109 |
| 对外投资 | 2.80 | 0.77 | 3.7x | 20,246,223 |
| 分支机构 | 2.44 | 0.65 | 3.8x | 12,274,728 |
| 动产抵押-抵押物 | 2.29 | 0.21 | 10.7x | 12,099,509 |
| 产品许可 | 0.46 | 0.04 | 11.4x | 3,423,868 |
| 动产抵押 | 0.35 | 0.04 | 7.8x | 1,236,423 |
| 股权出质 | 0.19 | 0.03 | 6.2x | 1,267,083 |
| 动产抵押-抵押人 | 0.17 | 0.02 | 7.1x | 1,220,323 |
| 融资历史 | 0.08 | 0.02 | 5.0x | 427,337 |
| 动产抵押-抵押变更 | 0.01 | 0.00 | 6.9x | 64,189 |
| **TOTAL** | **1,050.3** | **184.7** | **5.7x avg** | **~2.86 billion** |

---

## Key Findings from Validation

| Finding | Detail |
|---------|--------|
| Join key quality | 97–100% 公司ID match rate across all tables → joins are reliable |
| 招聘 date range | 2014-01-01 → 2025-12-31 (11 years) |
| 招聘 sources | 4,072 distinct data sources |
| 对外投资 column order | `认缴投资时间` precedes `认缴投资金额` in schema — both values are correct |
| 工商变更 `变更类型` | 100% NULL — column is unused/reserved |
| 产品许可 `截止日期` | 100% NULL — column is unused/reserved |
| `--` convention | Used throughout as null/unknown placeholder (not empty string) |

---

## Scripts & Files Created

| File | Purpose |
|------|---------|
| `00_data_pipeline_plan.md` | This document — plan + progress log |
| `01_pilot_csv_to_parquet.py` | Pilot test script (single file) |
| `02_convert_all_to_parquet.py` | Batch CSV → Parquet conversion (resume-safe) |
| `02_convert_log.json` | Machine-readable conversion log (sizes, timings) |
| `03_schema_discovery.py` | Schema + data dictionary generator |
| `schema_dictionary.md` | Human-readable data dictionary for all 15 tables |
| `04_duckdb_validation.py` | DuckDB validation queries across all tables |
| `04_validation_report.md` | Validation results (row counts, join coverage, QA) |
| `check_status.py` | Quick status checker — run any time to see pipeline state |
| `config.py` | **Central path config** — change `BASE_DIR` here to relocate all data paths |
| `data_utils.py` | **Main data access layer** — query, sample, filter, join, pickle |
| `05_setup_duckdb.py` | Creates `enterprise.duckdb` with 15 views + 5 macros; re-run to rebuild |
| `06a_profile_dirty_cols.py` | Profiles 10 dirty columns — top-200 value frequencies per column |
| `06a_dirty_col_profile.md` | Output of dirty column profiling |
| `07_data_cleaning.py` | Cleans 10 columns → 5 `_clean` views in `enterprise.duckdb` |
| `07_cleaning_report.md` | Cleaning validation report — sample counts and parse rates |
| `08_bert_inference.py` | **Step 6** — SOC code inference on 招聘_clean using fine-tuned BERT (406 classes); outputs `招聘_soc_labels.parquet` + DuckDB view `招聘_labeled` |

---

## Next Steps

| Step | Description | Priority | Status |
|------|-------------|----------|--------|
| Step 4: DuckDB persistent DB | Create `enterprise.duckdb` with Parquet views for fast SQL | High | ✅ Done — `05_setup_duckdb.py`, 15 views + 5 macros |
| Step 5: SQL Server load | ~~Bulk-load small/medium tables; load filtered subsets of giants~~ | ~~Medium~~ | ⏭️ **Skipped** — local-only workflow, DuckDB covers all query needs |
| Step 6: ML inference | Inference via fine-tuned BERT (406 SOC classes) → `招聘_soc_labels.parquet` + DuckDB view | High | ✅ Done — `08_bert_inference.py` |
| Data cleaning | Parse `注册资本`, `工作薪酬`, `参保人数` into numeric types | High | ✅ Done — `07_data_cleaning.py`, 5 clean views in enterprise.duckdb |
| SAS access | Point SAS LIBNAME at Parquet dir (SAS 9.4M6+) | Low | As needed |

### Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-13 | Skip SQL Server (Step 5) | Single-user local workflow; DuckDB reads Parquet directly with equivalent speed; no need for shared/network access or BI tool integration |
| 2026-03-13 | DuckDB as sole query engine | 128 GB RAM + Parquet columnar storage sufficient for all analytical tasks including joins across 2.86B rows |
