# Public result summaries

Only aggregate, non-sensitive result files are committed here. Raw face images,
model checkpoints, server logs, and private paths are excluded.

`facescrub/` contains the current manuscript evidence:

- `formal_summary.json`: aggregate DDP utility and attack intervals;
- `formal_utility_target_level.csv`: five target-adaptation utility rows;
- `formal_component_target_level.csv`: G-only, S-only, and fusion utility;
- `formal_attack_runs.csv`: decoder, GAN, and adaptive attack runs;
- `supplementary_summary.json`: capacity and backbone controls;
- `supplementary_utility_target_level.csv`: control utility rows;
- `supplementary_attack_runs.csv`: control attack rows;
- `efficiency_target_level.csv`: parameter, payload, checkpoint, and latency data;
- `development_operating_point_screen.json`: development-only noise screen;
- `data_manifest.json`: file sizes, row counts, and SHA-256 hashes.

The five target adaptations reuse one frozen spatial foundation. Published CEM
values are not a local matched reproduction. See `docs/EXPERIMENT_STATUS.md`
before citing these records.
