# Cloud GPU Inference Guide — SOC Labeling at Scale

## Context
- Task: BERT-based SOC classification on 641M Chinese job posting rows
- Script: `10_full_inference_linux.py` (Phase A: title dedup, Phase B: full-text inference)
- Local machine: NVIDIA Quadro RTX 5000 (16GB), ~220 seq/sec — too slow (~30 days)

---

## 1. Bottleneck Diagnosis

| Observation | Root Cause |
|---|---|
| seq/sec flat across batch sizes 64/128/512 | GPU memory-bandwidth-bound, not compute-bound |
| GPU compute utilization 22% | BERT attention is O(seq²) — memory controller busy, compute units wait |
| Multi-threading (tok_workers) only gave ~10% gain | Tokenizer was only ~10% of wall time; GPU BW was 90% |
| Sort-by-length optimization hurt locally | Front-loaded all 512-token sequences → GPU thermal stress |

**Key insight:** For BERT inference with long sequences (512 tokens), the bottleneck is **GPU memory bandwidth**, not compute or CPU tokenization. The fix is a faster GPU (more GB/s), not more CPU threads.

---

## 2. Cloud Platform Choice

### Why Alibaba Cloud (Aliyun) over alternatives
| Option | Problem |
|---|---|
| Google Colab Pro+ ($50/mo) | 24-hour session limit — cannot run multi-day jobs |
| GCP A100 VM | A100 not available in region |
| **Aliyun gn7i (A10)** | ✓ Available, dedicated, persistent disk |

### Instance chosen: `ecs.gn7i-8x.32xlarge`
- 4× NVIDIA A10 GPUs (24GB VRAM each, 600 GB/s memory bandwidth)
- 128 vCPU, 512 GiB RAM
- OS: Alibaba Cloud Linux 3.2104 LTS (pre-installed NVIDIA drivers + CUDA)

### GPU performance vs local
| GPU | Memory BW | Expected seq/sec |
|---|---|---|
| RTX 5000 (local) | 448 GB/s | ~220 |
| A10 (Aliyun gn7i) | 600 GB/s | ~340 |
| A100 40GB | 1,555 GB/s | ~760 |

---

## 3. Storage Setup

### Disk configuration
- **System disk only**: 500 GiB ESSD PL0
- **云盘释放行为: 不随实例释放** ← CRITICAL: uncheck "release with instance" so data survives spot preemption
- No separate data disk needed — disk I/O is never the bottleneck for this workload

### Directory structure on VM
```
/home/ecs-user/data/          ← actual data location
/data/                        ← symlink → /home/ecs-user/data/  (matches hardcoded script paths)

/data/parquet/招聘/data.parquet          ← source (90 GB)
/data/parquet/招聘_soc_full/             ← output chunks + checkpoints
/data/model/chinese-bert-wwm/            ← tokenizer
/data/model/bert_soc/                    ← fine-tuned classifier
/data/model/est_sample.csv               ← label map
/data/10_full_inference_linux.py         ← script
```

Create symlink:
```bash
sudo ln -s /home/ecs-user/data /data
mkdir -p /home/ecs-user/data/parquet/招聘 \
          /home/ecs-user/data/parquet/招聘_soc_full \
          /home/ecs-user/data/model \
          /home/ecs-user/data/logs
```

---

## 4. Data Transfer: Windows → VM via OSS

### Why OSS (not scp)
- VM public bandwidth: 100 Mbps → 90 GB parquet takes ~2 hours via scp
- OSS internal endpoint: ~Gbps → 90 GB in seconds once in OSS
- Upload Windows → OSS limited only by home internet speed

### Steps

**On Windows — install ossutil:**
```
Download: https://www.alibabacloud.com/help/en/oss/developer-reference/install-ossutil
Place at: D:\ossutil\ossutil64.exe
```

**Configure ossutil (Windows):**
```bat
D:\ossutil\ossutil64.exe config -e oss-cn-chengdu.aliyuncs.com -i <AccessKeyID> -k <AccessKeySecret>
```

**Upload from Windows:**
```bat
cd /d "I:\posting_2026\parquet\招聘"
D:\ossutil\ossutil64.exe cp data.parquet oss://posting-china/parquet/招聘/ --jobs 8

cd /d "D:\Dropbox\Dropbox\vs_cloud\HPC"
D:\ossutil\ossutil64.exe cp -r chinese-bert-wwm oss://posting-china/model/chinese-bert-wwm/ --jobs 4

cd /d "G:\Data\job_posting\processed"
D:\ossutil\ossutil64.exe cp -r model oss://posting-china/model/bert_soc/ --jobs 4
D:\ossutil\ossutil64.exe cp finetune\est_sample.csv oss://posting-china/model/

cd /d "I:\posting_2026\parquet"
D:\ossutil\ossutil64.exe cp 招聘_soc_title_lookup.parquet oss://posting-china/parquet/
D:\ossutil\ossutil64.exe cp -r 招聘_soc_full oss://posting-china/parquet/招聘_soc_full/ --jobs 4

cd /d "D:\Dropbox\Dropbox\vs_cloud\Job_posting_data\posting_2026"
D:\ossutil\ossutil64.exe cp 10_full_inference_linux.py oss://posting-china/
```

**Install ossutil on VM:**
```bash
# Get correct URL from: https://www.alibabacloud.com/help/en/oss/developer-reference/install-ossutil
wget <linux-ossutil-url> -O ossutil64
chmod +x ossutil64 && sudo mv ossutil64 /usr/local/bin/ossutil

# Configure with INTERNAL endpoint (free + ~Gbps)
ossutil config -e oss-cn-chengdu-internal.aliyuncs.com -i <AccessKeyID> -k <AccessKeySecret>
```

**Download to VM (fast via internal network):**
```bash
ossutil cp -r oss://posting-china/ /home/ecs-user/data/ --jobs 8
```

### OSS bucket settings
- Region: **西南1（成都）** — must match VM region
- Storage type: 标准存储 (Standard)
- Redundancy: 本地冗余 (LRS)
- Access: 私有 (Private)
- **Do NOT buy resource packages** — pay-as-you-go costs ~¥1 for a one-time 90 GB transfer
- Delete files from OSS after transfer to VM to avoid ongoing storage costs

---

## 5. Python Environment

Python 3.10 is pre-installed at `/usr/local/bin/python3.10`.

```bash
# Add to PATH
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install packages
python3.10 -m pip install torch transformers polars pyarrow duckdb pandas numpy
```

---

## 6. Parallel GPU Inference — 4 × A10

### Key principle
Run 4 separate Python processes, each assigned to one GPU via `CUDA_VISIBLE_DEVICES`. No DataParallel needed. Each process handles its own chunk range and writes to its own checkpoint file.

### CRITICAL: must set CUDA_VISIBLE_DEVICES
Without it, all processes default to GPU 0 → OOM crash.

### Launch commands (use tmux for SSH disconnect safety)

```bash
# Install tmux if needed
sudo yum install -y tmux   # or sudo apt install tmux

# Create persistent session
tmux new-session -s inference

# GPU 0 (chunks 3–82)
CUDA_VISIBLE_DEVICES=0 python3.10 /home/ecs-user/data/10_full_inference_linux.py \
  --phase b --gpu_id 0 --start_chunk 3 --end_chunk 83 \
  --batch_full 1024 --tok_workers 8

# GPU 1 (chunks 83–162) — open new tmux window: Ctrl+B then C
CUDA_VISIBLE_DEVICES=1 python3.10 /home/ecs-user/data/10_full_inference_linux.py \
  --phase b --gpu_id 1 --start_chunk 83 --end_chunk 163 \
  --batch_full 1024 --tok_workers 8

# GPU 2 (chunks 163–242)
CUDA_VISIBLE_DEVICES=2 python3.10 /home/ecs-user/data/10_full_inference_linux.py \
  --phase b --gpu_id 2 --start_chunk 163 --end_chunk 243 \
  --batch_full 1024 --tok_workers 8

# GPU 3 (chunks 243–321)
CUDA_VISIBLE_DEVICES=3 python3.10 /home/ecs-user/data/10_full_inference_linux.py \
  --phase b --gpu_id 3 --start_chunk 243 --end_chunk 321 \
  --batch_full 1024 --tok_workers 8
```

### Tmux navigation
```
Ctrl+B then C        → new window
Ctrl+B then 0/1/2/3  → switch windows
Ctrl+B then D        → detach (leave running)
tmux attach -t inference  → reattach after SSH reconnect
```

### Chunk range formula for N GPUs
```
Total chunks: 321 (0-indexed)
Done already: chunks 0, 1, 2 (Phase B on local)
Remaining: chunks 3–320 = 318 chunks

Per GPU (4 GPUs): ~80 chunks each
GPU 0: 3–82    (80 chunks)
GPU 1: 83–162  (80 chunks)
GPU 2: 163–242 (80 chunks)
GPU 3: 243–321 (78 chunks)
```

---

## 7. Performance Results

### Observed on A10 (single GPU, batch_full=1024, tok_workers=8)
- GPU utilization: **100%** ✓
- GPU VRAM usage: ~76% (18GB of 24GB)
- CPU usage: ~13% (not the bottleneck)
- Stabilized seq/sec: **~340 per GPU**
- 4 GPUs combined: **~1,360 seq/sec** (~6× faster than local)

### Why GPU utilization reached 100%
- A10 has 600 GB/s memory bandwidth vs RTX 5000's 448 GB/s
- Larger VRAM (24GB) allows batch_full=1024 vs local limit of 512
- Larger batches amortize model weight loading, improving efficiency

### Why further tuning is unnecessary at 100% GPU util
- More tok_workers: useless (CPU not bottleneck)
- Larger batch_full: won't help (GPU already maxed), risks OOM
- Sort-by-length: beneficial for A10 in theory but not needed when GPU is already at 100%

---

## 8. Fault Tolerance

- **Spot preemption**: disk set to 不随实例释放 → all data survives
- **Checkpoint files**: one per GPU (`checkpoint_gpu{N}.json`) → resume from last completed chunk
- **Recovery**: relaunch same 4 commands → each process skips already-done chunks automatically

---

## 9. Cost Estimate

- Instance type: ecs.gn7i-8x.32xlarge (pay-as-you-go)
- OSS transfer: ~¥1 one-time for 90 GB
- Delete OSS files after transfer to avoid ongoing storage costs
- Stop/release instance immediately after job completes

---

*Last updated: 2026-03-18*
