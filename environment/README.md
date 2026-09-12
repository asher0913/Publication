# Formal experiment environment

Create the Linux/CUDA environment from the repository root:

```bash
conda env create -f environment/conda-linux-64.yml
conda activate publication-cem
python scripts/prefetch_evaluation_assets.py
python scripts/capture_environment.py
bash scripts/run_checks.sh
```

The formal run must use Python 3.11, PyTorch 2.2.2, torchvision 0.17.2 and
CUDA 12.1 as frozen above. Do not reuse the macOS development interpreter for
reported latency or GPU-memory measurements. `evaluation_assets.json` must be
created before launching attacks so all workers use identical frozen LPIPS and
FaceNet weights.
