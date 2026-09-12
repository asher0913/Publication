# Attribution and release boundary

This project extends the official code accompanying Xia et al., "Theoretical
Insights in Model Inversion Robustness and Conditional Entropy Maximization for
Collaborative Inference Systems" (CVPR 2025). The upstream snapshot is retained
under `archive/upstream_cem/`, together with its original licence and notices.

The dissertation-era implementation in `archive/current_artefact/` also
contains inherited VGG, SSIM, and LCA components. Those files are kept for
provenance and to load the current spatial foundation. Their upstream notices
must remain attached to any redistribution.

The implementation in `src/publication_cem/` and the DDP experiment runners in
`scripts/` are new research code. A project-wide licence has not yet been
approved by all prospective authors. Until that decision is recorded, the
repository is available for inspection and research collaboration but grants
no additional licence for the newly written files.

FaceScrub images and trained checkpoints are not part of this repository. The
dataset must be obtained under its own terms, and checkpoints should be shared
privately with integrity hashes.
