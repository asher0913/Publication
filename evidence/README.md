# Public protocol evidence

This directory contains non-image manifests needed to audit the data split and
evaluation stack. Local dataset roots have been replaced with repository-relative
placeholders; sample paths, class mappings, counts, and recorded hashes are
otherwise preserved. No FaceScrub image is included.

`official_cem_equivalence.json` verifies the ported scalar KMeans/CEM loss path,
not a completed matched training reproduction of the published CEM result.
