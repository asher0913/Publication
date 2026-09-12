# Experiment status

Status is separated into completed evidence, executable work, and unresolved
design work. This prevents prepared code from being mistaken for a result.

## Completed and available in this repository

- FaceScrub DDP utility for five adaptation seeds: 81.96% mean top-1 accuracy.
- G-only, S-only, pre-adaptation, and post-adaptation utility measurements.
- Decoder, GAN, and wider adaptive joint attacks under training- and
  inference-knowledge settings.
- Capacity-matched one-stage control and a ResNet-18 semantic-backbone control.
- Parameter count, payload, checkpoint size, and GPU latency records.
- Aggregate CSV/JSON files with row counts and SHA-256 hashes.

## Important limits of the completed evidence

- The five DDP targets share one frozen G-Path foundation trained with seed 828;
  they are not five independent end-to-end target seeds.
- Published CEM table values are an external reference. A matched local
  official-CEM reproduction is still required for the main comparison.
- Completed joint attackers consume both released paths, but they do not
  explicitly condition reconstruction on a predicted or oracle identity.
- Current evidence is FaceScrub-only. Cross-dataset generalisation is not yet
  established.
- The current fusion uses global parameters rather than sample-dependent
  gating.

## Implemented runners awaiting GPU execution

- Identity-conditioned joint attack: G-only, unconditioned joint, predicted
  identity posterior, and oracle identity upper bound.
- Noise sensitivity over multiple `(sigma_g, sigma_s)` operating points.
- Fusion comparisons for the currently available trained logits.
- Token dimensions 64, 128, 256, and 384.
- Supervisor-revision aggregation and resumable watchdog support.

The runner is `scripts/run_supervisor_revision_experiments.py`. Its output must
not be cited until the completion artefacts have been inspected and aggregated.

## Work that requires an author decision before execution

- Decide whether the final paper describes the current feature-map release or a
  new prototype-transmission G-Path.
- Decide whether to retain calibrated global fusion or implement a genuinely
  input-dependent gate.
- Freeze the exact official-CEM matched protocol.
- Freeze the independent-seed policy and the additional datasets/backbones.
- Agree on the strongest feasible model-inversion baselines and visual privacy
  metrics.

## Minimum evidence needed before submission

1. A matched local official-CEM baseline under identical data, backbone, cut,
   attack, metric, and training conditions.
2. Independent end-to-end seeds for every headline method, with mean,
   dispersion, and paired statistical comparisons.
3. Identity-conditioned joint attacks, including an oracle upper bound.
4. At least two additional datasets or a documented narrowing of the paper's
   scope to face inference.
5. G-only, S-only, and joint utility **and privacy** ablations.
6. Noise, fusion, token-dimension, runtime, payload, and backbone sensitivity.
7. Qualitative reconstructions selected by a predeclared rule rather than only
   favourable examples.
