# Reproducibility checklist

## Before training

- Record the Git commit and a clean/dirty worktree status.
- Capture Python, PyTorch, torchvision, CUDA, cuDNN, driver, and GPU versions.
- Hash the dataset manifest and every initialization checkpoint.
- Freeze target seeds, attacker seeds, data split, transforms, model capacity,
  epoch budget, optimizer, scheduler, and evaluation metrics.
- Keep model selection validation-only. Do not tune on the final test set.

## During training

- Use a separate output directory for every method and seed.
- Preserve stdout/stderr, resolved arguments, and the best-checkpoint criterion.
- Treat an incomplete completion JSON as a failed job, not a finished run.
- Relaunch through the resumable runner so valid completed jobs are skipped.

## After training

- Evaluate every method with the same target split and attack budgets.
- Average attacker initialisations within each target before computing
  target-level intervals.
- Report failed runs and exclusions with reasons.
- Generate aggregate CSV/JSON files and a SHA-256 manifest.
- Compare manuscript numbers against the machine-readable aggregates.

## Current commands

```bash
python scripts/capture_environment.py
python -m pytest -q
python scripts/run_dual_path_publication_pipeline.py --help
python scripts/run_supervisor_revision_experiments.py --help
```

The final command line should be recorded only after the authors freeze the
architecture and experiment matrix.
