# PhD collaborator handoff

## Start here

1. Read `README.md`, `docs/ARCHITECTURE.md`, and
   `docs/EXPERIMENT_STATUS.md` before running anything.
2. Clone the repository and create the pinned environment.
3. Obtain FaceScrub and the legacy G-Path checkpoints through the private
   handoff. Verify their SHA-256 hashes.
4. Run the test suite and one GPU smoke test.
5. Discuss and freeze the architecture and comparison protocol with Yixuan and
   Prof. Ren before launching the full revision matrix.

## Questions that need agreement

- Should the paper remain about the implemented noised feature-map G-Path, or
  should the code be changed to prototype transmission?
- Is global calibrated fusion sufficient, or is a sample-dependent gate part of
  the intended contribution?
- Which exact official CEM checkpoint, data split, cut layer, and attackers form
  the matched baseline?
- Which two additional datasets can be completed with the available GPU budget?
- What counts as an independent target seed when the spatial foundation is
  frozen?

## Suggested execution order

```bash
conda env create -f environment/conda-linux-64.yml
conda activate publication-cem
python -m pip install -e '.[dev,evaluation,analysis]'
python -m pytest -q
python scripts/smoke_facescrub_mechanism.py --help
python scripts/run_supervisor_revision_experiments.py --help
```

After the design is frozen, first run one complete seed including the strongest
attack. Inspect its logs, metrics, and checkpoint compatibility. Only then
release the remaining seeds and datasets. The resumable runners preserve state,
but they cannot correct a scientifically mismatched protocol.

## Expected outputs

Each formal run should preserve:

- resolved configuration and code commit;
- target and attacker seeds;
- training logs and best-checkpoint hashes;
- per-image or per-batch metric records where permitted;
- aggregate accuracy, MSE, PSNR, SSIM, LPIPS, and identity-similarity metrics;
- parameters, FLOPs, payload bytes, GPU memory, and latency;
- environment capture and dataset manifest;
- failed/interrupted-run records rather than silently omitting them.

Do not put raw faces, private checkpoints, Overleaf review comments, or server
credentials in the public repository.
