import json, os

log_path    = r'D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\02_convert_log.json'
parquet_dir = r'I:\posting_2026\parquet'
csv_dir     = r'I:\posting_2026\定制数据'
schema_rpt  = r'D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\03_schema_report.json'

with open(log_path, 'r', encoding='utf-8') as f:
    log = json.load(f)

done, pending = [], []
for k, v in sorted(log.items(), key=lambda x: x[1].get('csv_bytes', 0)):
    if v.get('status') == 'done':
        done.append((k, v))
    else:
        pending.append((k, v))

print(f'=== CONVERSION STATUS ===')
print(f'Done: {len(done)}  |  Pending: {len(pending)}\n')

total_csv  = sum(v.get('csv_bytes', 0)     for k, v in done)
total_pq   = sum(v.get('parquet_bytes', 0) for k, v in done)

print('DONE:')
for k, v in done:
    print(f'  [OK] {k:<35}  CSV={v["csv_bytes"]/1e9:7.2f} GB  '
          f'PQ={v["parquet_bytes"]/1e9:6.2f} GB  '
          f'{v["compression_x"]}x  ({v["elapsed_s"]:.0f}s)')

if pending:
    print('\nPENDING (not yet converted):')
    for k, v in pending:
        csv_gb = v.get('csv_bytes', 0) / 1e9
        print(f'  [..] {k:<35}  CSV={csv_gb:7.2f} GB  status={v.get("status")}')
else:
    print('\nAll files converted!')

print(f'\nTotal CSV converted : {total_csv/1e9:7.1f} GB')
print(f'Total Parquet size  : {total_pq/1e9:7.1f} GB')
savings = total_csv - total_pq
if total_csv > 0:
    print(f'Space saved         : {savings/1e9:7.1f} GB  ({savings/total_csv*100:.0f}% reduction)')

# Also check actual parquet files on disk
if os.path.exists(parquet_dir):
    print(f'\nParquet files on disk ({parquet_dir}):')
    for f in sorted(os.listdir(parquet_dir)):
        fpath = os.path.join(parquet_dir, f)
        if os.path.isdir(fpath):
            sz = sum(os.path.getsize(os.path.join(fpath, x))
                     for x in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, x)))
        else:
            sz = os.path.getsize(fpath)
        print(f'  {f:<45}  {sz/1e9:6.2f} GB')

# Cross-check: which CSVs are still missing from log?
if os.path.exists(csv_dir):
    csv_stems = {os.path.splitext(fn)[0] for fn in os.listdir(csv_dir) if fn.endswith('.csv')}
    missing = csv_stems - set(log.keys())
    print(f'\nCSVs missing from conversion log ({len(missing)}): {missing if missing else "None - all accounted for!"}')

# Schema report status
print(f'\n=== SCHEMA REPORT ===')
if os.path.exists(schema_rpt):
    with open(schema_rpt, encoding='utf-8') as f:
        sr = json.load(f)
    print(f'03_schema_report.json exists: {len(sr)} tables documented')
    for tbl, info in sorted(sr.items()):
        ncols = len(info.get('columns', []))
        nrows = info.get('row_count', 'N/A')
        print(f'  {tbl:<35}  {ncols:3d} cols  {nrows:>15,} rows')
else:
    print('03_schema_report.json: NOT found (schema discovery not run yet)')
