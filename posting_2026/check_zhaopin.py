import os, json

src_dir    = r'I:\posting_2026\定制数据'
parquet_dir = r'I:\posting_2026\parquet'
log_path   = r'D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026\02_convert_log.json'

print("=== 1. Source CSV check ===")
for f in os.listdir(src_dir):
    if '招聘' in f:
        fpath = os.path.join(src_dir, f)
        sz = os.path.getsize(fpath)
        print(f"  FOUND: {f}  ({sz/1e9:.2f} GB)")
if not any('招聘' in f for f in os.listdir(src_dir)):
    print("  招聘.csv NOT found in source directory!")
    print("  All files in source:")
    for f in sorted(os.listdir(src_dir)):
        sz = os.path.getsize(os.path.join(src_dir, f))
        print(f"    {f}  ({sz/1e9:.2f} GB)")

print("\n=== 2. Parquet output check ===")
for entry in os.listdir(parquet_dir):
    if '招聘' in entry:
        fpath = os.path.join(parquet_dir, entry)
        if os.path.isdir(fpath):
            files = os.listdir(fpath)
            total = sum(os.path.getsize(os.path.join(fpath, x)) for x in files)
            print(f"  DIR : {entry}/  contains {len(files)} files, {total/1e9:.2f} GB")
            for x in files[:5]:
                print(f"         {x}")
        else:
            sz = os.path.getsize(fpath)
            print(f"  FILE: {entry}  ({sz/1e9:.2f} GB)")

print("\n=== 3. Log check ===")
with open(log_path, 'r', encoding='utf-8') as f:
    log = json.load(f)
if '招聘' in log:
    print(f"  招聘 IS in log: {log['招聘']}")
else:
    print("  招聘 is NOT in the log — was never processed!")
    print(f"  Log has {len(log)} entries: {list(log.keys())}")
