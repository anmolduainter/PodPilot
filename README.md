<p align="center">
  <img src="assets/banner.svg" alt="PodPilot Banner" width="100%"/>
</p>

<p align="center">
  <strong>Autopilot for your RunPod GPU infrastructure.</strong><br>
  <sub>Smart GPU selection. One-command provisioning. Beautiful terminal output.</sub>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/pip_install-podpilot-00d2ff?style=for-the-badge" alt="pip install"/></a>
  <a href="#-features"><img src="https://img.shields.io/badge/RunPod-SDK-7b2ff7?style=for-the-badge" alt="RunPod"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License"/></a>
</p>

<br>

---

**Stop babysitting GPUs.** PodPilot wraps the RunPod API into a single intelligent interface that discovers the best GPU for your workload, launches pods with automatic retry and live progress, provisions network volumes in the optimal data center, and tracks your spend — all with rich, color-coded terminal output.

```python
from runpod_manager import RunPodManager

mgr = RunPodManager()  # reads RUNPOD_API_KEY from env

# One line: find best GPU, create pod, wait until ready
pod = mgr.launch("my-training", image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel", min_vram=48)
```

<br>

## Why PodPilot?

| Without PodPilot | With PodPilot |
|:---|:---|
| Manually browse RunPod dashboard for GPU availability | `mgr.best_gpu(min_vram=48)` — auto-picks with scoring |
| Guess which data center has stock | `mgr.recommend(vram_needed=80)` — ranked recommendations |
| SSH in to set up volumes, download models | `mgr.provision_volume(...)` — fully automated |
| Forget a pod running overnight, $50 gone | `mgr.status()` — spend rate + hours remaining |
| Pod creation fails silently | Live progress bar with retry and timeout handling |
| Write boilerplate for every project | 3 lines to go from zero to running pod |

<br>

## Features

### Smart GPU Discovery & Selection

PodPilot scores every available GPU using a composite algorithm that balances **VRAM fit**, **price**, and **availability** — so you always get the best value, not just the cheapest or biggest card.

```python
# See all available GPUs in a beautiful table
mgr.gpus()

# Filter: 48GB+ VRAM, available only, under $2/hr
mgr.gpus(min_vram=48, available_only=True, max_price=2.0)

# Auto-select the single best GPU for your workload
mgr.best_gpu(min_vram=48)
# ╭─── Best GPU ───╮
# │ A100 80GB  $1.64/hr │
# │ 80GB fits 48GB need, available in secure+community │
# ╰─────────────────╯

# Top 5 recommendations with explanations
mgr.recommend(vram_needed=24, budget=1.5)
```

**Sorting strategies:**

| Strategy | Best for |
|:--|:--|
| `SortBy.SMART` | Balanced composite score (default) |
| `SortBy.PRICE` | Cheapest available |
| `SortBy.VRAM` | Maximum memory |
| `SortBy.VRAM_PER_DOLLAR` | Best bang for buck |
| `SortBy.AVAILABILITY` | Highest stock |

---

### Pod Lifecycle Management

Full pod lifecycle with smart defaults, duplicate detection, and live progress.

```python
# Launch with auto GPU selection
pod = mgr.launch(
    "stable-diffusion",
    image="stabilityai/stable-diffusion:latest",
    min_vram=24,
    max_price=1.0,        # budget cap
    gpu_count=1,
    container_disk_gb=50,
    ports="8188/http,22/tcp",
    env={"HF_TOKEN": "hf_..."},
)

# Relaunch? PodPilot detects the existing pod
pod = mgr.launch("stable-diffusion", ...)
# → "Pod 'stable-diffusion' already running: abc123"

# Stop (preserves data) → Resume → Terminate
mgr.stop(name="stable-diffusion")
mgr.resume(name="stable-diffusion")
mgr.terminate(name="stable-diffusion")

# Bulk operations
mgr.stop_all()       # stop every running pod
mgr.cleanup()        # terminate all stopped pods
mgr.terminate_all()  # nuclear option
```

**Live startup progress:**

```
  Creating pod stable-diffusion
  ├── GPU:              RTX 4090 (24GB, $0.44/hr)
  ├── Image:            stabilityai/stable-diffusion:latest
  └── Container disk:   50GB

     0s  Status: CREATED
    12s  Status: STARTING
  Provisioning GPU...  ████████████░░░░░░░░  38s / 900s
    52s  Status: RUNNING
  Pod ready! abc123def
```

---

### One-Command Volume Provisioning

The killer feature. Create a network volume, auto-place it in a data center that has your target GPU, spin up a cheap pod to download your models, and tear it down when done — **all in one call**.

```python
# Full automated provisioning pipeline
vol = mgr.provision_volume(
    "sd-models",
    size_gb=60,
    download_script="download_models.sh",  # your local .sh file
    gpu_vram=48,  # places volume where 48GB GPUs exist
)
# 1. Finds best data center with 48GB+ GPUs
# 2. Creates 60GB network volume there
# 3. Spins up cheapest GPU pod (<$1/hr) in same DC
# 4. SCPs your script, runs it via SSH with live output
# 5. Terminates the pod when done
# → Volume ready to attach to any pod in that DC

# Monitor a running provision (safe to disconnect & reconnect)
mgr.check_provision()
mgr.provision_logs(lines=50)

# When done manually (if you disconnected)
mgr.finish_provision()
```

**Also handles volumes individually:**

```python
mgr.volumes()                                         # list all
mgr.create_volume("my-data", 100, min_vram=48)        # auto-pick DC
mgr.create_volume("my-data", 100, data_center_id="US-TX-3")
mgr.resize_volume(name="my-data", size_gb=200)
mgr.delete_volume(name="my-data")

# Where should I create a volume for 80GB GPU workloads?
mgr.recommend_volume_location(min_vram=80)
```

---

### Cost Tracking & Account Status

Never get surprised by a bill again.

```python
mgr.balance()
# ╭─── Account Balance ───╮
# │ Credits:       $47.23  │
# │ Spend rate:    $0.4400/hr │
# │ Remaining:     107.3h  │
# │ Referrals:     $5.00   │
# ╰────────────────────────╯

mgr.status()  # everything at a glance
# ╭─── RunPod Status ───╮
# │ Balance: $47.23   Spend: $0.4400/hr │
# │ Hours left: 107.3h │
# │ Pods: 2 running, 1 stopped │
# │   ● training-run    A100 80GB  $1.64/hr │
# │   ● inference-api   RTX 4090   $0.44/hr │
# ╰──────────────────────╯
```

---

### Rich Terminal Output

Every command produces clean, readable output using [Rich](https://github.com/Textualize/rich) — tables, panels, spinners, color-coded status, and progress bars. Designed to look great in Jupyter notebooks, terminals, and SSH sessions alike.

---

## Installation

```bash
pip install runpod-manager
```

**Requirements:** Python 3.10+ &nbsp;|&nbsp; Dependencies: `runpod`, `requests`, `rich`

**Set your API key:**

```bash
export RUNPOD_API_KEY="your-key-here"
```

Or pass it directly:

```python
mgr = RunPodManager(api_key="your-key-here")
```

---

## Quick Start

```python
from runpod_manager import RunPodManager, SortBy

mgr = RunPodManager()

# 1. Explore available GPUs
mgr.gpus(min_vram=24, available_only=True)

# 2. Get smart recommendation
mgr.recommend(vram_needed=48, budget=2.0)

# 3. Launch a pod
pod = mgr.launch(
    "my-project",
    image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
    min_vram=24,
)

# 4. Check your spend
mgr.status()

# 5. Done for the day
mgr.stop(name="my-project")

# 6. Pick up tomorrow
mgr.resume(name="my-project")
```

---

## API Reference

### `RunPodManager(api_key=None)`

Main entry point. Reads `RUNPOD_API_KEY` from environment if not provided.

#### GPU Methods

| Method | Description |
|:--|:--|
| `gpus(min_vram, max_vram, available_only, max_price, sort_by, cloud_type)` | List & filter GPUs |
| `best_gpu(min_vram, max_price, prefer)` | Auto-select best GPU |
| `recommend(vram_needed, budget)` | Top 5 ranked recommendations |

#### Pod Methods

| Method | Description |
|:--|:--|
| `launch(name, image, min_vram, max_price, ...)` | Smart pod creation with auto GPU selection |
| `pods()` | List all pods |
| `stop(name=)` / `stop(pod_id=)` | Stop a pod (preserves data) |
| `resume(name=)` | Resume a stopped pod |
| `terminate(name=)` | Permanently destroy a pod |
| `stop_all()` | Stop all running pods |
| `cleanup()` | Terminate all stopped pods |
| `terminate_all()` | Terminate everything |

#### Volume Methods

| Method | Description |
|:--|:--|
| `volumes()` | List all network volumes |
| `create_volume(name, size_gb, data_center_id=, min_vram=)` | Create volume (auto DC selection) |
| `resize_volume(name=, size_gb=)` | Resize a volume (increase only) |
| `delete_volume(name=)` | Delete a volume |
| `provision_volume(name, size_gb, download_script, gpu_vram)` | Full automated provisioning pipeline |
| `check_provision()` | Check provisioning progress |
| `provision_logs(lines=30)` | View download logs |
| `finish_provision()` | Terminate download pod |
| `recommend_volume_location(min_vram)` | Best data centers for your GPU needs |

#### Account Methods

| Method | Description |
|:--|:--|
| `balance()` | Credits, spend rate, hours remaining |
| `status()` | Full dashboard: balance + all pods |

---

## Architecture

```
runpod_manager/
├── manager.py      # RunPodManager — top-level orchestrator
├── gpu.py          # GPUManager — discovery, filtering, composite scoring
├── pods.py         # PodManager — lifecycle, live progress, retry
├── volumes.py      # VolumeManager — CRUD, DC discovery, SSH provisioning
└── console.py      # Rich console, retry decorator, scoring algorithm
```

**Design principles:**
- **Smart defaults, full control** — every auto-pick can be overridden
- **Resilient** — all API calls retry with exponential backoff
- **Cached** — GPU and data center lists cached for 5 minutes
- **Composable** — use `RunPodManager` for convenience, or `GPUManager`/`PodManager`/`VolumeManager` directly
- **Observable** — rich output for every operation, no silent failures

---

## License

MIT

---

<p align="center">
  <sub>Built for ML engineers who'd rather train models than manage infrastructure.</sub><br>
  <sub>If PodPilot saved you time, consider giving it a star.</sub>
</p>
