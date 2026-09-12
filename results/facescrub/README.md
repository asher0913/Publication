# FaceScrub aggregate evidence

These files are the non-sensitive aggregates used to audit the current DDP
experiments. Target models are the unit of replication. Repeated attacker
initialisations are averaged within each target before target-level intervals
are calculated.

The five target adaptations share one frozen spatial foundation, so they must
not be described as five independent end-to-end runs. Diagnostic, interrupted,
and smoke runs are excluded from these tables. `data_manifest.json` records row
counts, file sizes, and SHA-256 hashes after public path sanitisation.
