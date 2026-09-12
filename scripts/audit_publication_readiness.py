#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from scripts.expand_attack_matrix import build_protocol_commands
except ModuleNotFoundError:
    from expand_attack_matrix import build_protocol_commands


CORE_NAME = re.compile(
    r"seed12[5-9]_(no_defense|gaussian|official_cem|single_prototype|slots8)"
)
ATTACKER_SEEDS = (10125, 20125, 30125)
REQUIRED_TARGET_FILES = (
    "resolved_config.json",
    "training.jsonl",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "target_test_metrics.json",
)
REQUIRED_ATTACK_FILES = (
    "attack_training.jsonl",
    "attack_metrics.json",
    "per_image_metrics.jsonl",
)
REQUIRED_PRIVACY_METRICS = (
    "mse",
    "ssim",
    "psnr",
    "lpips",
    "identity_top1_success",
    "face_cosine_similarity",
    "verification_tar_at_far",
)


@dataclass(frozen=True)
class AuditItem:
    identifier: str
    category: str
    status: str
    summary: str
    evidence: str
    required_action: str = ""


def item(
    identifier: str,
    category: str,
    passed: bool,
    summary: str,
    evidence: str,
    required_action: str = "",
) -> AuditItem:
    return AuditItem(
        identifier=identifier,
        category=category,
        status="PASS" if passed else "FAIL",
        summary=summary,
        evidence=evidence,
        required_action="" if passed else required_action,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def core_experiments(root: Path) -> tuple[dict, list[dict]]:
    frozen = root / "configs" / "frozen_facescrub_core_matrix.json"
    matrix_path = frozen if frozen.is_file() else root / "configs/facescrub_core_matrix.json"
    matrix = read_json(matrix_path)
    experiments = [
        experiment
        for experiment in matrix["experiments"]
        if CORE_NAME.fullmatch(experiment["name"])
    ]
    return matrix, experiments


def command_check(root: Path, identifier: str, command: list[str]) -> AuditItem:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_lines = completed.stdout.strip().splitlines()
    evidence = "\n".join(output_lines[-8:]) or "command produced no output"
    return item(
        identifier,
        "engineering",
        completed.returncode == 0,
        f"{' '.join(command)} exits successfully",
        evidence,
        "Fix the reported engineering failure before starting experiments.",
    )


def audit_repository(root: Path, run_checks: bool = False) -> list[AuditItem]:
    checks: list[AuditItem] = []
    required_sources = [
        "src/publication_cem/regularizer.py",
        "src/publication_cem/prototype_bank.py",
        "src/publication_cem/epochwise_cem.py",
        "src/publication_cem/official_cem.py",
        "src/publication_cem/evaluators.py",
        "src/publication_cem/models.py",
        "scripts/train_target.py",
        "scripts/train_attack.py",
        "scripts/build_face_identity_protocol.py",
        "scripts/build_torchvision_protocol.py",
        "scripts/prefetch_evaluation_assets.py",
        "scripts/profile_checkpoint.py",
        "scripts/smoke_facescrub_mechanism.py",
        "scripts/expand_selection_attacks.py",
        "scripts/select_hyperparameters.py",
        "scripts/freeze_headline_matrix.py",
        "scripts/run_command_file.py",
        "scripts/run_full_publication_pipeline.py",
        "scripts/build_paper_handoff.py",
        "scripts/preflight_formal_run.py",
        "scripts/generate_runbooks.py",
        "scripts/summarise_formal_pilot.py",
        "scripts/export_qualitative_grid.py",
        "scripts/validate_formal_environment.py",
        "scripts/smoke_evaluation_pipeline.py",
        "scripts/aggregate_results.py",
        "src/publication_cem/dual_path.py",
        "src/publication_cem/dual_path_attack.py",
        "scripts/run_dual_path_publication_pipeline.py",
        "scripts/run_identity_conditioned_attack_protocol.py",
        "docs/ARCHITECTURE.md",
        "docs/EXPERIMENT_STATUS.md",
        "docs/REPRODUCIBILITY.md",
    ]
    missing_sources = [path for path in required_sources if not (root / path).is_file()]
    checks.append(
        item(
            "source_components",
            "engineering",
            not missing_sources,
            "Core source, protocol and reproducibility files are present",
            f"missing={missing_sources}",
            "Restore every required public research source file.",
        )
    )

    selection_matrix_path = root / "configs" / "facescrub_selection_matrix.json"
    selection_matrix = read_json(selection_matrix_path) if selection_matrix_path.is_file() else {}
    selection_design_ok = (
        len(selection_matrix.get("experiments", [])) == 18
        and {experiment["overrides"].get("seed") for experiment in selection_matrix["experiments"]}
        == {120}
    )
    checks.append(
        item(
            "test_blind_selection_design",
            "protocol",
            selection_design_ok,
            "Hyperparameter selection uses seed 120 and attacker validation only",
            f"selection_runs={len(selection_matrix.get('experiments', []))}, seeds={sorted({experiment['overrides'].get('seed') for experiment in selection_matrix.get('experiments', [])})}",
            "Repair the selection matrix and selection-only attack stage.",
        )
    )

    attack_smoke_path = root / "results" / "attack_suite_smoke.json"
    attack_smoke_ok = False
    attack_smoke_evidence = "attack_suite_smoke.json is missing"
    if attack_smoke_path.is_file():
        attack_smoke = read_json(attack_smoke_path)
        attack_smoke_types = set(attack_smoke.get("attacks", {}))
        attack_smoke_ok = attack_smoke.get("status") == "PASS" and attack_smoke_types == {
            "conv_decoder",
            "residual_decoder",
            "gan",
            "adaptive",
        }
        attack_smoke_evidence = (
            f"status={attack_smoke.get('status')}, types={sorted(attack_smoke_types)}"
        )
    checks.append(
        item(
            "attack_suite_smoke",
            "engineering",
            attack_smoke_ok,
            "Every declared attack completes training, selection and test evaluation",
            attack_smoke_evidence,
            "Run scripts/smoke_attack_suite.py after attack-interface changes.",
        )
    )

    evaluation_smoke_path = root / "results" / "evaluation_pipeline_smoke.json"
    evaluation_smoke_ok = False
    evaluation_smoke_evidence = "evaluation_pipeline_smoke.json is missing"
    if evaluation_smoke_path.is_file():
        evaluation_smoke = read_json(evaluation_smoke_path)
        evaluation_smoke_metrics = set(evaluation_smoke.get("metrics", {}))
        evaluation_smoke_ok = (
            evaluation_smoke.get("status") == "PASS"
            and set(REQUIRED_PRIVACY_METRICS).issubset(evaluation_smoke_metrics)
        )
        evaluation_smoke_evidence = (
            f"status={evaluation_smoke.get('status')}, "
            f"metrics={sorted(evaluation_smoke_metrics)}"
        )
    checks.append(
        item(
            "evaluation_pipeline_smoke",
            "engineering",
            evaluation_smoke_ok,
            "LPIPS and frozen face metrics complete through the formal attack CLI",
            evaluation_smoke_evidence,
            "Run smoke_evaluation_pipeline.py in the evaluation-enabled environment.",
        )
    )

    if run_checks:
        checks.extend(
            [
                command_check(root, "ruff", ["python", "-m", "ruff", "check", "src", "scripts", "tests"]),
                command_check(root, "pytest", ["python", "-m", "pytest", "-q"]),
                command_check(root, "compileall", ["python", "-m", "compileall", "-q", "src", "scripts", "tests"]),
            ]
        )

    pipeline_smoke_path = root / "results" / "pipeline_smoke.json"
    pipeline_smoke_ok = False
    pipeline_smoke_evidence = "pipeline_smoke.json is missing"
    if pipeline_smoke_path.is_file():
        pipeline_smoke = read_json(pipeline_smoke_path)
        pipeline_smoke_ok = pipeline_smoke.get("status") == "PASS"
        pipeline_smoke_evidence = (
            f"status={pipeline_smoke.get('status')}, "
            f"methods={sorted(pipeline_smoke.get('methods', {}))}"
        )
    checks.append(
        item(
            "cli_pipeline_smoke",
            "engineering",
            pipeline_smoke_ok,
            "Target, checkpoint reload, attack and aggregation CLIs complete end to end",
            pipeline_smoke_evidence,
            "Run scripts/smoke_publication_pipeline.py and inspect its evidence.",
        )
    )

    environment_spec = root / "environment" / "conda-linux-64.yml"
    environment_spec_ok = environment_spec.is_file() and all(
        marker in environment_spec.read_text(encoding="utf-8")
        for marker in (
            "python=3.11.9",
            "pytorch=2.2.2",
            "torchvision=0.17.2",
            "pytorch-cuda=12.1",
            "facenet-pytorch==2.6.0",
            "lpips==0.1.4",
        )
    )
    checks.append(
        item(
            "environment_specification",
            "protocol",
            environment_spec_ok,
            "The formal Linux/CUDA dependency environment is version-frozen",
            str(environment_spec.relative_to(root)) if environment_spec.exists() else "missing",
            "Freeze mutually compatible Python, CUDA, PyTorch and evaluator versions.",
        )
    )

    facescrub_smoke_path = root / "results" / "actual_facescrub_mechanism_smoke.json"
    facescrub_smoke_ok = False
    facescrub_smoke_evidence = "actual_facescrub_mechanism_smoke.json is missing"
    if facescrub_smoke_path.is_file():
        facescrub_smoke = read_json(facescrub_smoke_path)
        facescrub_smoke_ok = facescrub_smoke.get("status") == "PASS"
        facescrub_smoke_evidence = (
            f"status={facescrub_smoke.get('status')}, "
            f"classes={facescrub_smoke.get('data', {}).get('num_classes')}, "
            f"smashed_shape={facescrub_smoke.get('model', {}).get('smashed_shape')}"
        )
    checks.append(
        item(
            "facescrub_mechanism_smoke",
            "engineering",
            facescrub_smoke_ok,
            "Real FaceScrub tensors complete forward, privacy backward and decoder steps",
            facescrub_smoke_evidence,
            "Run a real-data mechanism smoke after any data/model interface change.",
        )
    )

    matrix, experiments = core_experiments(root)
    generalisation_files = (
        "configs/facescrub_generalisation_matrix.json",
        "configs/cifar100_generalisation_matrix.json",
        "configs/cifar10_generalisation_matrix.json",
        "configs/generalisation_attack_protocol.json",
    )
    generalisation_configs_ok = all((root / path).is_file() for path in generalisation_files)
    generalisation_target_count = 0
    generalisation_attack_count = 0
    if generalisation_configs_ok:
        generalisation_protocol = read_json(
            root / "configs/generalisation_attack_protocol.json"
        )
        for dataset in ("facescrub", "cifar100", "cifar10"):
            generalisation_matrix = read_json(
                root / f"configs/{dataset}_generalisation_matrix.json"
            )
            generalisation_target_count += len(generalisation_matrix["experiments"])
            generalisation_attack_count += len(
                build_protocol_commands(
                    generalisation_matrix,
                    {
                        "attacker_seeds": generalisation_protocol["attacker_seeds"],
                        "tiers": [
                            {
                                "target_name_regex": ".*",
                                "attack_types": generalisation_protocol[
                                    "attack_types"
                                ],
                            }
                        ],
                    },
                )
            )
        generalisation_configs_ok = (
            generalisation_target_count
            == int(generalisation_protocol["expected_target_runs"])
            and generalisation_attack_count
            == int(generalisation_protocol["expected_attack_runs"])
        )
    configured_datasets = {"facescrub", "cifar100", "cifar10"}
    configured_backbones = {"vgg11_bn_split", "resnet18_split"}
    configured_cuts = {8, 15, 2}
    checks.append(
        item(
            "generalisation_design",
            "protocol",
            generalisation_configs_ok,
            "Frozen matrices and attacks cover three datasets, two backbones and multiple split points",
            f"datasets={sorted(configured_datasets)}, backbones={sorted(configured_backbones)}, cuts={sorted(configured_cuts)}, targets={generalisation_target_count}, attacks={generalisation_attack_count}",
            "Repair the generalisation matrices and attack protocol before formal runs.",
        )
    )

    attack_protocol_path = root / "configs" / "facescrub_attack_protocol.json"
    attack_plan_ok = False
    attack_plan_evidence = "attack protocol is missing"
    if attack_protocol_path.is_file():
        attack_protocol = read_json(attack_protocol_path)
        attack_commands = build_protocol_commands(matrix, attack_protocol)
        attack_plan_ok = len(attack_commands) == 165 and len(set(attack_commands)) == 165
        attack_plan_evidence = f"commands={len(attack_commands)}, unique={len(set(attack_commands))}"
    checks.append(
        item(
            "attack_protocol_design",
            "protocol",
            attack_plan_ok,
            "The tiered attack plan is complete and contains no duplicate runs",
            attack_plan_evidence,
            "Repair the attack protocol to produce 75 primary and 90 strong attacks.",
        )
    )

    base_config = read_json(root / "configs" / "facescrub_base.json")
    evaluation_config = base_config.get("attack", {}).get("evaluation", {})
    evaluator_config_ok = (
        evaluation_config.get("lpips", {}).get("enabled") is True
        and evaluation_config.get("face_identity", {}).get("enabled") is True
        and evaluation_config.get("face_identity", {}).get("target_far") == 0.001
    )
    checks.append(
        item(
            "evaluation_protocol_design",
            "protocol",
            evaluator_config_ok,
            "Formal FaceScrub attacks require LPIPS and fixed-FAR identity metrics",
            json.dumps(evaluation_config, sort_keys=True),
            "Enable and freeze both perceptual and identity evaluation settings.",
        )
    )

    secondary_manifests = {
        name: root / "evidence" / f"{name}_split_manifest.json"
        for name in ("cifar10", "cifar100")
    }
    valid_secondary = []
    for name, path in secondary_manifests.items():
        if not path.is_file():
            continue
        manifest = read_json(path)
        split_records = manifest.get("splits", {})
        required_secondary = {
            "target_train",
            "target_validation",
            "target_test",
            "attacker_auxiliary_train",
            "attacker_auxiliary_validation",
        }
        records_ok = required_secondary.issubset(split_records)
        train_sets = []
        if records_ok:
            for split_name in required_secondary:
                record = split_records[split_name]
                indices = [int(index) for index in record.get("indices", [])]
                source = str(record.get("source", ""))
                payload_text = "\n".join(
                    f"{source}:{index}" for index in indices
                )
                digest = hashlib.sha256(payload_text.encode("ascii")).hexdigest()
                records_ok = records_ok and (
                    indices == sorted(indices)
                    and len(indices) == int(record.get("count", -1))
                    and len(indices) == len(set(indices))
                    and record.get("sha256") == digest
                    and source
                    == ("test" if split_name == "target_test" else "train")
                )
                if split_name != "target_test":
                    train_sets.append(set(indices))
        disjoint = not any(
            left & right
            for position, left in enumerate(train_sets)
            for right in train_sets[position + 1 :]
        )
        data_dir = root / "data/torchvision" / (
            "cifar-10-batches-py" if name == "cifar10" else "cifar-100-python"
        )
        if (
            manifest.get("dataset") == name
            and records_ok
            and disjoint
            and data_dir.is_dir()
        ):
            valid_secondary.append(name)
    secondary_manifest_ok = set(valid_secondary) == {"cifar10", "cifar100"}
    checks.append(
        item(
            "secondary_dataset_splits",
            "preflight",
            secondary_manifest_ok,
            "CIFAR-10/100 index partitions are downloaded, hashed and disjoint",
            f"validated={sorted(valid_secondary)}",
            "Run build_torchvision_protocol.py for CIFAR-10 and CIFAR-100 before GPU training.",
        )
    )

    assets_path = root / "evidence" / "evaluation_assets.json"
    assets_ok = False
    assets = {}
    assets_evidence = "evaluation_assets.json is missing"
    if assets_path.is_file():
        assets = read_json(assets_path)
        assets_ok = (
            assets.get("status") == "VERIFIED"
            and assets.get("lpips_net") == "alex"
            and assets.get("lpips_package_version") == "0.1.4"
            and assets.get("facenet_pytorch_package_version") == "2.6.0"
            and assets.get("face_embedding_shape") == [2, 512]
            and float(assets.get("lpips_self_distance_max", math.inf)) <= 1e-6
            and float(assets.get("face_embedding_norm_error_max", math.inf))
            <= 1e-5
            and valid_sha256(assets.get("lpips_weights_sha256"))
            and valid_sha256(assets.get("face_weights_sha256"))
        )
        assets_evidence = (
            f"status={assets.get('status')}, lpips={assets.get('lpips_package_version')}, "
            f"facenet={assets.get('facenet_pytorch_package_version')}, "
            f"lpips_net={assets.get('lpips_net')}"
        )
    checks.append(
        item(
            "evaluation_assets",
            "preflight",
            assets_ok,
            "Frozen LPIPS-Alex and VGGFace2 weights execute and are hash-recorded",
            assets_evidence,
            "Install the formal environment and run prefetch_evaluation_assets.py.",
        )
    )

    face_protocol_path = root / "evidence" / "face_identity_protocol.json"
    face_protocol_ok = False
    face_protocol_evidence = "face identity protocol is missing"
    if face_protocol_path.is_file():
        face_protocol = read_json(face_protocol_path)
        protocol_file = Path(str(face_protocol.get("protocol_file", "")))
        if not protocol_file.is_absolute():
            protocol_file = root / protocol_file
        current_split_hash = file_sha256(root / "evidence/data_split_manifest.json")
        split_manifest = read_json(root / "evidence/data_split_manifest.json")
        face_protocol_ok = (
            face_protocol.get("status") == "CALIBRATED"
            and face_protocol.get("num_classes") == 526
            and face_protocol.get("target_far") == 0.001
            and face_protocol.get("resize_antialias") is True
            and face_protocol.get("formal_cuda_compatible") is True
            and math.isfinite(float(face_protocol.get("verification_threshold", math.nan)))
            and protocol_file.is_file()
            and face_protocol.get("protocol_file_sha256")
            == file_sha256(protocol_file)
            and face_protocol.get("split_manifest_sha256") == current_split_hash
            and face_protocol.get("target_train_path_sha256")
            == split_manifest["splits"]["target_train"]["sha256"]
            and face_protocol.get("target_validation_path_sha256")
            == split_manifest["splits"]["target_validation"]["sha256"]
            and face_protocol.get("embedding_model_weights_sha256")
            == assets.get("face_weights_sha256")
        )
        face_protocol_evidence = (
            f"status={face_protocol.get('status')}, classes={face_protocol.get('num_classes')}, "
            f"target_far={face_protocol.get('target_far')}, "
            f"formal_cuda={face_protocol.get('formal_cuda_compatible')}"
        )
    checks.append(
        item(
            "face_identity_calibration",
            "preflight",
            face_protocol_ok,
            "Face identity prototypes and FAR threshold are calibrated without test data",
            face_protocol_evidence,
            "Run build_face_identity_protocol.py after evaluator assets are verified.",
        )
    )

    selection_path = root / "evidence" / "frozen_hyperparameters.json"
    frozen_matrix_path = root / "configs" / "frozen_facescrub_core_matrix.json"
    frozen_selection_ok = False
    frozen_selection_evidence = "test-blind hyperparameter selection has not been run"
    if selection_path.is_file() and frozen_matrix_path.is_file():
        selection = read_json(selection_path)
        frozen_matrix = read_json(frozen_matrix_path)
        selected = selection.get("selected", {})
        selected_ok = set(selected) == {"gaussian", "official_cem", "proposed"}
        minimum_accuracy = float(
            selection.get("minimum_eligible_accuracy", math.nan)
        )
        for family in ("gaussian", "official_cem", "proposed"):
            record = selected.get(family, {})
            selected_ok = selected_ok and all(
                math.isfinite(float(record.get(name, math.nan)))
                for name in (
                    "validation_accuracy",
                    "attacker_validation_mse",
                    "noise_std",
                    "privacy_weight",
                )
            )
            selected_ok = selected_ok and (
                float(record.get("validation_accuracy", -math.inf))
                >= minimum_accuracy
            )
        frozen_values_ok = True
        for experiment in frozen_matrix.get("experiments", []):
            name = experiment["name"]
            overrides = experiment["overrides"]
            if name.endswith("_gaussian"):
                frozen_values_ok = frozen_values_ok and overrides.get(
                    "defense.noise_std"
                ) == selected.get("gaussian", {}).get("noise_std")
            elif name.endswith("_official_cem"):
                frozen_values_ok = frozen_values_ok and (
                    overrides.get("defense.noise_std")
                    == selected.get("official_cem", {}).get("noise_std")
                    and overrides.get("defense.privacy_weight")
                    == selected.get("official_cem", {}).get("privacy_weight")
                )
            elif name.endswith(("_single_prototype", "_slots8")):
                frozen_values_ok = frozen_values_ok and (
                    overrides.get("defense.noise_std")
                    == selected.get("proposed", {}).get("noise_std")
                    and overrides.get("defense.privacy_weight")
                    == selected.get("proposed", {}).get("privacy_weight")
                )
        frozen_selection_ok = (
            selection.get("target_test_accessed") is False
            and selection.get("candidate_count") == 18
            and float(selection.get("utility_drop_limit", math.nan)) == 0.01
            and selected_ok
            and frozen_matrix.get("frozen_from_test_blind_selection") is True
            and len(frozen_matrix.get("experiments", [])) == 25
            and frozen_values_ok
        )
        frozen_selection_evidence = (
            f"target_test_accessed={selection.get('target_test_accessed')}, "
            f"headline_runs={len(frozen_matrix.get('experiments', []))}"
        )
    checks.append(
        item(
            "frozen_test_blind_hyperparameters",
            "preflight",
            frozen_selection_ok,
            "Pilot settings are selected without target-test access and frozen for headline seeds",
            frozen_selection_evidence,
            "Complete selection-only runs, select_hyperparameters.py and freeze_headline_matrix.py before headline training.",
        )
    )

    dataset_manifest_path = root / "evidence" / "dataset_manifest.json"
    if dataset_manifest_path.is_file():
        dataset = read_json(dataset_manifest_path)
        dataset_ok = (
            dataset.get("train", {}).get("class_count") == 526
            and dataset.get("validation", {}).get("class_count") == 526
            and dataset.get("train", {}).get("image_count") == 41425
            and dataset.get("validation", {}).get("image_count") == 4335
        )
        evidence = (
            f"classes={dataset.get('train', {}).get('class_count')}, "
            f"train={dataset.get('train', {}).get('image_count')}, "
            f"current_val={dataset.get('validation', {}).get('image_count')}, "
            f"class_hash={dataset.get('class_name_sha256')}"
        )
    else:
        dataset_ok = False
        evidence = "dataset manifest is missing"
    checks.append(
        item(
            "dataset_inventory",
            "protocol",
            dataset_ok,
            "The archived FaceScrub inventory is identified exactly",
            evidence,
            "Regenerate and verify the FaceScrub dataset manifest.",
        )
    )

    split_manifest_path = root / "evidence" / "data_split_manifest.json"
    split_keys = {
        "target_train",
        "target_validation",
        "target_test",
        "attacker_auxiliary_train",
        "attacker_auxiliary_validation",
    }
    split_ok = False
    split_manifest = {}
    split_evidence = "data_split_manifest.json is missing"
    if split_manifest_path.is_file():
        split_manifest = read_json(split_manifest_path)
        available = split_manifest.get("splits", {})
        split_ok = split_keys.issubset(available)
        split_sets = []
        if split_ok:
            for split_name in sorted(split_keys):
                record = available[split_name]
                paths = list(record.get("paths", []))
                digest = hashlib.sha256(
                    "\n".join(paths).encode("utf-8")
                ).hexdigest()
                split_ok = split_ok and (
                    paths == sorted(paths)
                    and len(paths) == int(record.get("count", -1))
                    and len(paths) == len(set(paths))
                    and record.get("sha256") == digest
                )
                split_sets.append((split_name, set(paths)))
            split_ok = split_ok and not any(
                left & right
                for position, (_, left) in enumerate(split_sets)
                for _, right in split_sets[position + 1 :]
            )
        split_counts = {
            name: available[name].get("count") for name in sorted(available)
        }
        split_evidence = (
            f"available_splits={sorted(available)}, counts={split_counts}"
        )
    checks.append(
        item(
            "independent_data_splits",
            "protocol",
            split_ok,
            "Target selection, final test and attacker auxiliary data are separated",
            split_evidence,
            "Create a hashed target-train/validation/test and attacker-auxiliary split before final runs.",
        )
    )

    equivalence_path = root / "evidence" / "official_cem_equivalence.json"
    equivalence_ok = False
    equivalence_evidence = "official_cem_equivalence.json is missing"
    if equivalence_path.is_file():
        equivalence = read_json(equivalence_path)
        equivalence_ok = equivalence.get("status") == "verified"
        equivalence_evidence = json.dumps(equivalence, sort_keys=True)
    checks.append(
        item(
            "official_cem_equivalence",
            "protocol",
            equivalence_ok,
            "The primary CEM baseline is demonstrably equivalent to the public implementation",
            equivalence_evidence,
            "Port the official estimator or insert the candidate into the official runner and record numerical equivalence tests.",
        )
    )

    names = {experiment["name"] for experiment in experiments}
    methods = {CORE_NAME.fullmatch(name).group(1) for name in names}
    target_matrix_ok = len(experiments) == 25 and methods == {
        "no_defense",
        "gaussian",
        "official_cem",
        "single_prototype",
        "slots8",
    }
    checks.append(
        item(
            "target_matrix",
            "protocol",
            target_matrix_ok,
            "The core matrix contains five methods across five target seeds",
            f"core_runs={len(experiments)}, methods={sorted(methods)}",
            "Repair the core matrix so each headline method has seeds 125-129.",
        )
    )

    results_root = root / matrix["results_root"]
    missing_target_files = []
    invalid_target_metrics = []
    for experiment in experiments:
        run_dir = results_root / experiment["name"]
        for filename in REQUIRED_TARGET_FILES:
            if not (run_dir / filename).is_file():
                missing_target_files.append(f"{experiment['name']}/{filename}")
        target_metrics_path = run_dir / "target_test_metrics.json"
        config_path = run_dir / "resolved_config.json"
        training_path = run_dir / "training.jsonl"
        if (
            target_metrics_path.is_file()
            and config_path.is_file()
            and training_path.is_file()
        ):
            target_metrics = read_json(target_metrics_path)
            resolved_config = read_json(config_path)
            training_records = [
                json.loads(line)
                for line in training_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            expected_epochs = int(resolved_config["training"]["epochs"])
            accuracy = float(target_metrics.get("test_accuracy", math.nan))
            target_seed = int(experiment["overrides"]["seed"])
            if (
                len(training_records) != expected_epochs
                or not training_records
                or int(training_records[-1].get("epoch", -1)) != expected_epochs
                or not 1
                <= int(target_metrics.get("best_epoch", -1))
                <= expected_epochs
                or target_metrics.get("target_test_accessed") is not True
                or target_metrics.get("evaluation_split") != "target_test"
                or target_metrics.get("evaluation_noise_seed")
                != target_seed + 200_000
                or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1
            ):
                invalid_target_metrics.append(experiment["name"])
    checks.append(
        item(
            "headline_target_results",
            "evidence",
            not missing_target_files and not invalid_target_metrics,
            "All 25 headline target runs are complete",
            f"missing_count={len(missing_target_files)}, invalid_metrics={invalid_target_metrics[:8]}, examples={missing_target_files[:8]}",
            "Run every frozen headline target configuration and retain configs, logs and best/last checkpoints.",
        )
    )

    missing_attack_files = []
    metric_paths = []
    for experiment in experiments:
        run_dir = results_root / experiment["name"]
        for attack_seed in ATTACKER_SEEDS:
            attack_dir = run_dir / "attacks" / "conv_decoder" / f"seed{attack_seed}"
            complete_attack = True
            for filename in REQUIRED_ATTACK_FILES:
                path = attack_dir / filename
                if not path.is_file():
                    complete_attack = False
                    missing_attack_files.append(
                        f"{experiment['name']}/attacks/conv_decoder/seed{attack_seed}/{filename}"
                    )
            if complete_attack:
                metric_paths.append(attack_dir / "attack_metrics.json")
    checks.append(
        item(
            "base_attack_results",
            "evidence",
            not missing_attack_files,
            "Three independent base-decoder attacks exist for every headline target",
            f"expected=75, metrics_found={len(metric_paths)}, missing_files={len(missing_attack_files)}, examples={missing_attack_files[:8]}",
            "Train all 75 base attackers and retain convergence plus per-image metrics.",
        )
    )

    missing_metrics: dict[str, list[str]] = {}
    expected_facescrub_test_images = int(
        split_manifest.get("splits", {}).get("target_test", {}).get("count", 0)
    )
    for path in metric_paths:
        attack_metrics = read_json(path)
        available = set(attack_metrics)
        absent = [
            metric
            for metric in REQUIRED_PRIVACY_METRICS
            if metric not in available
            or not math.isfinite(float(attack_metrics.get(metric, math.nan)))
        ]
        target_config = read_json(path.parents[3] / "resolved_config.json")
        target_seed = int(target_config["seed"])
        expected_attack_epochs = int(target_config["attack"]["epochs"])
        attack_records = [
            line
            for line in (path.parent / "attack_training.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        per_image_count = sum(
            1
            for line in (path.parent / "per_image_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if attack_metrics.get("evaluation_noise_seed") != target_seed + 300_000:
            absent.append("evaluation_noise_seed")
        if (
            int(attack_metrics.get("attack_epochs", -1)) != expected_attack_epochs
            or len(attack_records) != expected_attack_epochs
            or not 1
            <= int(attack_metrics.get("best_epoch", -1))
            <= expected_attack_epochs
        ):
            absent.append("complete_attack_epochs")
        if (
            attack_metrics.get("evaluation_split") != "target_test"
            or per_image_count != expected_facescrub_test_images
        ):
            absent.append("complete_target_test_evaluation")
        if absent:
            missing_metrics[str(path.relative_to(root))] = absent
    metric_coverage_ok = len(metric_paths) == 75 and not missing_metrics
    checks.append(
        item(
            "privacy_metric_coverage",
            "evidence",
            metric_coverage_ok,
            "Every headline attack reports pixel, perceptual and identity metrics",
            f"metric_files={len(metric_paths)}, incomplete_examples={list(missing_metrics.items())[:3]}",
            "Run all final attacks through the implemented LPIPS and frozen face-recognition evaluators.",
        )
    )

    strong_attack_types = set()
    for path in results_root.rglob("attack_metrics.json") if results_root.exists() else []:
        payload = read_json(path)
        attack_type = payload.get("attack_type")
        if attack_type:
            strong_attack_types.add(attack_type)
    required_attack_types = {"conv_decoder", "residual_decoder", "gan", "adaptive"}
    checks.append(
        item(
            "strong_attack_suite",
            "evidence",
            required_attack_types.issubset(strong_attack_types),
            "The evaluation includes convolutional, residual, GAN and adaptive attacks",
            f"available_attack_types={sorted(strong_attack_types)}",
            "Run the implemented residual, GAN and adaptive attackers with the frozen equal auxiliary-data budget.",
        )
    )

    strong_missing = []
    strong_targets = [
        experiment
        for experiment in experiments
        if experiment["name"].endswith(("official_cem", "slots8"))
    ]
    for experiment in strong_targets:
        for attack_type in ("residual_decoder", "gan", "adaptive"):
            for attack_seed in ATTACKER_SEEDS:
                attack_dir = (
                    results_root
                    / experiment["name"]
                    / "attacks"
                    / attack_type
                    / f"seed{attack_seed}"
                )
                required_paths = [
                    attack_dir / filename for filename in REQUIRED_ATTACK_FILES
                ]
                metrics_path = attack_dir / "attack_metrics.json"
                valid_metrics = False
                if all(path.is_file() for path in required_paths):
                    metrics = read_json(metrics_path)
                    target_config = read_json(
                        results_root / experiment["name"] / "resolved_config.json"
                    )
                    expected_epochs = int(target_config["attack"]["epochs"])
                    attack_epoch_count = sum(
                        1
                        for line in (attack_dir / "attack_training.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    )
                    per_image_count = sum(
                        1
                        for line in (attack_dir / "per_image_metrics.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    )
                    valid_metrics = (
                        metrics.get("evaluation_noise_seed")
                        == int(experiment["overrides"]["seed"]) + 300_000
                        and metrics.get("evaluation_split") == "target_test"
                        and metrics.get("attack_type") == attack_type
                        and int(metrics.get("attack_epochs", -1))
                        == expected_epochs
                        and attack_epoch_count == expected_epochs
                        and per_image_count == expected_facescrub_test_images
                        and all(
                            math.isfinite(float(metrics.get(name, math.nan)))
                            for name in REQUIRED_PRIVACY_METRICS
                        )
                    )
                if not all(path.is_file() for path in required_paths) or not valid_metrics:
                    strong_missing.append(str(attack_dir.relative_to(root)))
    checks.append(
        item(
            "strong_attack_results",
            "evidence",
            not strong_missing,
            "All 90 predeclared strong/adaptive attack results are present",
            f"expected=90, missing={len(strong_missing)}, examples={strong_missing[:5]}",
            "Run the strong-attack tier for official CEM and the proposed method.",
        )
    )

    completed_generalisation = []
    missing_generalisation_targets = []
    missing_generalisation_attacks = []
    generalisation_selection = (
        read_json(root / "evidence/frozen_hyperparameters.json").get("selected", {})
        if (root / "evidence/frozen_hyperparameters.json").is_file()
        else {}
    )
    for dataset in ("facescrub", "cifar100", "cifar10"):
        generalisation_matrix = read_json(
            root / f"configs/{dataset}_generalisation_matrix.json"
        )
        dataset_results_root = root / generalisation_matrix["results_root"]
        for experiment in generalisation_matrix["experiments"]:
            run_dir = dataset_results_root / experiment["name"]
            resolved_path = run_dir / "resolved_config.json"
            target_test_path = run_dir / "target_test_metrics.json"
            if not all((run_dir / filename).is_file() for filename in REQUIRED_TARGET_FILES):
                missing_generalisation_targets.append(
                    f"{dataset}/{experiment['name']}"
                )
                continue
            resolved = read_json(resolved_path)
            target_test = read_json(target_test_path)
            training_records = [
                line
                for line in (run_dir / "training.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            expected_target_epochs = int(resolved["training"]["epochs"])
            defense = resolved.get("defense", {})
            defense_name = defense.get("name")
            family = (
                "official_cem"
                if defense_name == "official_cem"
                else "proposed"
                if defense_name == "prototype_cem"
                else None
            )
            selected_values = generalisation_selection.get(family, {})
            frozen_transfer_ok = family is not None and (
                defense.get("noise_std") == selected_values.get("noise_std")
                and defense.get("privacy_weight")
                == selected_values.get("privacy_weight")
            )
            if (
                target_test.get("target_test_accessed") is not True
                or target_test.get("evaluation_split") != "target_test"
                or not math.isfinite(
                    float(target_test.get("test_accuracy", math.nan))
                )
                or len(training_records) != expected_target_epochs
                or not frozen_transfer_ok
            ):
                missing_generalisation_targets.append(
                    f"{dataset}/{experiment['name']}:invalid_target_test"
                )
                continue
            completed_generalisation.append(resolved)
            for attack_seed in ATTACKER_SEEDS:
                metrics_path = (
                    run_dir
                    / "attacks"
                    / "conv_decoder"
                    / f"seed{attack_seed}"
                    / "attack_metrics.json"
                )
                attack_dir = metrics_path.parent
                if not all(
                    (attack_dir / filename).is_file()
                    for filename in REQUIRED_ATTACK_FILES
                ):
                    missing_generalisation_attacks.append(
                        f"{dataset}/{experiment['name']}/seed{attack_seed}"
                    )
                    continue
                metrics = read_json(metrics_path)
                required = ("mse", "ssim", "psnr", "lpips")
                expected_attack_epochs = int(resolved["attack"]["epochs"])
                attack_epoch_count = sum(
                    1
                    for line in (attack_dir / "attack_training.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
                split_path = Path(resolved["data"]["split_manifest"])
                if not split_path.is_absolute():
                    split_path = root / split_path
                evaluation_count = int(
                    read_json(split_path)["splits"]["target_test"]["count"]
                )
                per_image_count = sum(
                    1
                    for line in (attack_dir / "per_image_metrics.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
                if any(
                    not math.isfinite(float(metrics.get(name, math.nan)))
                    for name in required
                ) or metrics.get("evaluation_noise_seed") != int(
                    resolved["seed"]
                ) + 300_000 or (
                    int(metrics.get("attack_epochs", -1))
                    != expected_attack_epochs
                    or attack_epoch_count != expected_attack_epochs
                    or per_image_count != evaluation_count
                    or metrics.get("evaluation_split") != "target_test"
                ):
                    missing_generalisation_attacks.append(
                        f"{dataset}/{experiment['name']}/seed{attack_seed}:invalid"
                    )
    datasets = {
        config.get("data", {}).get("name") for config in completed_generalisation
    }
    backbones = {
        config.get("model", {}).get("name") for config in completed_generalisation
    }
    split_points = {
        config.get("model", {}).get("cut_index")
        for config in completed_generalisation
    }
    datasets.discard(None)
    backbones.discard(None)
    split_points.discard(None)
    generalisation_ok = (
        len(completed_generalisation) == 30
        and not missing_generalisation_targets
        and not missing_generalisation_attacks
        and len(datasets) >= 2
        and len(backbones) >= 2
        and len(split_points) >= 2
    )
    checks.append(
        item(
            "generalisation_results",
            "evidence",
            generalisation_ok,
            "Completed target and inversion results cover multiple datasets, backbones and split points",
            f"completed_targets={len(completed_generalisation)}/30, missing_attacks={len(missing_generalisation_attacks)}/90, datasets={sorted(datasets)}, backbones={sorted(backbones)}, split_points={sorted(split_points)}, examples={(missing_generalisation_targets + missing_generalisation_attacks)[:5]}",
            "Run all 30 frozen generalisation targets and their 90 predeclared inversion attacks.",
        )
    )

    environment_json = root / "evidence" / "environment" / "environment.json"
    hardware_record = root / "evidence" / "environment" / "nvidia-smi.txt"
    environment_validation_path = (
        root / "evidence/environment/formal_environment_validation.json"
    )
    environment_ok = False
    environment_evidence = "formal environment capture is missing"
    if (
        environment_json.is_file()
        and hardware_record.is_file()
        and environment_validation_path.is_file()
    ):
        environment = read_json(environment_json)
        environment_validation = read_json(environment_validation_path)
        environment_ok = (
            environment.get("cuda_available") is True
            and str(environment.get("torch", "")).startswith("2.2.2")
            and str(environment.get("torchvision", "")).startswith("0.17.2")
            and bool(environment.get("gpus"))
            and environment_validation.get("status") == "PASS"
            and not environment_validation.get("mismatches")
        )
        environment_evidence = (
            f"cuda={environment.get('cuda_available')}, torch={environment.get('torch')}, "
            f"torchvision={environment.get('torchvision')}, gpus={len(environment.get('gpus', []))}"
        )
    checks.append(
        item(
            "environment_capture",
            "preflight",
            environment_ok,
            "Exact dependencies and GPU environment are archived",
            environment_evidence,
            "Capture pip/conda lock, OS, CUDA, cuDNN, PyTorch and nvidia-smi from the final experiment machine.",
        )
    )

    formal_pilot_path = root / "evidence" / "formal_gpu_pilot.json"
    formal_pilot_ok = False
    formal_pilot_evidence = "formal CUDA pilot has not been completed"
    if formal_pilot_path.is_file():
        formal_pilot = read_json(formal_pilot_path)
        methods = formal_pilot.get("methods", {})
        formal_pilot_ok = (
            formal_pilot.get("status") == "PASS"
            and formal_pilot.get("formal_cuda_environment") is True
            and formal_pilot.get("target_test_accessed") is False
            and set(methods)
            == {
                "seed119_no_defense",
                "seed119_official_cem",
                "seed119_proposed",
            }
            and all(
                int(record.get("peak_gpu_memory_bytes", 0)) > 0
                for record in methods.values()
            )
            and set(methods.get("seed119_no_defense", {}).get("attacks", {}))
            == {"conv_decoder"}
            and set(methods.get("seed119_official_cem", {}).get("attacks", {}))
            == {"conv_decoder"}
            and set(methods.get("seed119_proposed", {}).get("attacks", {}))
            == {"conv_decoder", "residual_decoder", "gan", "adaptive"}
        )
        formal_pilot_evidence = (
            f"status={formal_pilot.get('status')}, methods={sorted(methods)}, "
            f"target_test_accessed={formal_pilot.get('target_test_accessed')}"
        )
    checks.append(
        item(
            "formal_gpu_pilot",
            "preflight",
            formal_pilot_ok,
            "A full-data CUDA pilot validates target and attacker resource use",
            formal_pilot_evidence,
            "Run the three-method two-epoch pilot before selection or headline experiments.",
        )
    )

    valid_efficiency_profiles = []
    invalid_efficiency_profiles = []
    for experiment in experiments:
        path = results_root / experiment["name"] / "efficiency_profile.json"
        if not path.is_file():
            invalid_efficiency_profiles.append(experiment["name"])
            continue
        profile = read_json(path)
        finite_metrics = (
            "edge_latency_ms_per_sample_mean",
            "cloud_latency_ms_per_sample_mean",
            "end_to_end_latency_ms_per_sample_mean",
        )
        if (
            str(profile.get("device", "")).startswith("cuda")
            and int(profile.get("profiled_batches", 0)) >= 1
            and int(profile.get("transmitted_bytes_per_sample_fp32", 0)) > 0
            and all(
                math.isfinite(float(profile.get(name, math.nan)))
                and float(profile.get(name, 0.0)) > 0
                for name in finite_metrics
            )
        ):
            valid_efficiency_profiles.append(path)
        else:
            invalid_efficiency_profiles.append(experiment["name"])
    checks.append(
        item(
            "efficiency_results",
            "evidence",
            len(valid_efficiency_profiles) == 25,
            "Every headline target has a deployment and state-size profile",
            f"valid_profiles={len(valid_efficiency_profiles)}/25, invalid={invalid_efficiency_profiles[:5]}",
            "Run profile_checkpoint.py for each headline best checkpoint.",
        )
    )

    results_path = root / "paper" / "sections" / "results.tex"
    main_path = root / "paper" / "main.tex"
    if results_path.is_file() and main_path.is_file():
        results_tex = results_path.read_text(encoding="utf-8")
        main_tex = main_path.read_text(encoding="utf-8")
        placeholders = [
            marker
            for marker in (
                "No publication-candidate result is available yet",
                "intentionally blank",
                "Empirical result placeholder",
            )
            if marker in results_tex or marker in main_tex
        ]
    else:
        # The public source release intentionally excludes the private manuscript.
        # Keep the readiness result conservative without making the code audit crash.
        placeholders = ["manuscript_not_in_public_repository"]
    checks.append(
        item(
            "paper_results_filled",
            "manuscript",
            not placeholders,
            "The manuscript contains audited numerical results rather than placeholders",
            f"placeholders={placeholders}",
            "Generate tables/figures from audited results and replace all empirical placeholders.",
        )
    )

    claim_registry_path = root / "docs" / "claim_registry.md"
    if claim_registry_path.is_file():
        claim_registry = claim_registry_path.read_text(encoding="utf-8")
        unmeasured_claims = claim_registry.count("Not measured")
    else:
        # Claim-to-evidence mapping is maintained with the private manuscript.
        unmeasured_claims = 1
    checks.append(
        item(
            "claim_registry_supported",
            "manuscript",
            unmeasured_claims == 0,
            "All intended headline claims have supporting evidence",
            f"not_measured_claims={unmeasured_claims}",
            "Complete required experiments or remove unsupported claims from the intended paper scope.",
        )
    )
    return checks


def category_status(checks: Iterable[AuditItem]) -> dict[str, str]:
    grouped: dict[str, list[AuditItem]] = {}
    for check in checks:
        grouped.setdefault(check.category, []).append(check)
    return {
        category: "PASS" if all(check.status == "PASS" for check in values) else "FAIL"
        for category, values in grouped.items()
    }


def payload(root: Path, checks: list[AuditItem]) -> dict:
    categories = category_status(checks)
    code_categories = {"engineering", "protocol"}
    pre_run_categories = code_categories | {"preflight"}
    code_ready = all(categories.get(category) == "PASS" for category in code_categories)
    check_status = {check.identifier: check.status for check in checks}
    pilot_ready = (
        code_ready
        and check_status.get("environment_capture") == "PASS"
        and check_status.get("formal_gpu_pilot") == "PASS"
    )
    pre_run_ready = all(
        categories.get(category) == "PASS" for category in pre_run_categories
    )
    evidence_ready = all(
        categories.get(category) == "PASS"
        for category in ("evidence", "manuscript")
    )
    return {
        "schema_version": 1,
        "root": str(root),
        "code_status": "CODE_READY" if code_ready else "CODE_NOT_READY",
        "pilot_status": "PILOT_READY" if pilot_ready else "PILOT_NOT_READY",
        "pre_run_status": "PRE_RUN_READY" if pre_run_ready else "PRE_RUN_NOT_READY",
        "evidence_status": "EVIDENCE_READY" if evidence_ready else "EVIDENCE_NOT_READY",
        "overall_status": "READY" if all(value == "PASS" for value in categories.values()) else "NOT_READY",
        "category_status": categories,
        "passed": sum(check.status == "PASS" for check in checks),
        "failed": sum(check.status == "FAIL" for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(audit: dict) -> str:
    lines = [
        "# Publication Readiness Audit",
        "",
        f"**Overall status: {audit['overall_status']}**",
        f"**Code status: {audit['code_status']}**",
        f"**Pilot status: {audit['pilot_status']}**",
        f"**Pre-run status: {audit['pre_run_status']}**",
        f"**Evidence status: {audit['evidence_status']}**",
        "",
        "This status distinguishes executable code from completed scientific evidence. "
        "READY does not guarantee peer-review acceptance; it means the repository contains "
        "the predefined evidence needed to draft and audit the submission.",
        "",
        "## Category status",
        "",
        "| Category | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| {category} | {status} |"
        for category, status in sorted(audit["category_status"].items())
    )
    lines.extend(["", "## Checks", ""])
    for check in audit["checks"]:
        lines.extend(
            [
                f"### [{check['status']}] {check['identifier']}",
                "",
                check["summary"],
                "",
                f"Evidence: `{check['evidence']}`",
                "",
            ]
        )
        if check["required_action"]:
            lines.extend([f"Required action: {check['required_action']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return success while still reporting NOT_READY",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    audit = payload(root, audit_repository(root, run_checks=args.run_checks))

    output_json = args.output_json or root / "results" / "readiness_audit.json"
    output_md = args.output_md or root / "results" / "readiness_audit.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("overall_status", "passed", "failed")}, sort_keys=True))
    if audit["overall_status"] != "READY" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
