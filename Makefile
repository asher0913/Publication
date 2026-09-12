.PHONY: test check smoke matrix attack-matrix manifest dataset-manifest aggregate

test:
	python -m pytest -q

check:
	bash scripts/run_checks.sh

smoke:
	python scripts/smoke_publication_pipeline.py
	python scripts/smoke_attack_suite.py

matrix:
	python scripts/expand_matrix.py --base configs/facescrub_base.json --matrix configs/facescrub_core_matrix.json --output-dir generated_configs/core

attack-matrix:
	python scripts/expand_attack_matrix.py --target-matrix configs/facescrub_core_matrix.json --protocol configs/facescrub_attack_protocol.json --output generated_configs/attack_commands.txt

manifest:
	python scripts/build_manifest.py

dataset-manifest:
	python scripts/build_dataset_manifest.py --data-root ../artefact/data/facescrub

aggregate:
	python scripts/aggregate_results.py --results-root results --output-prefix results/summary
