#!/usr/bin/env python3
"""Run a configured batch of MapAnything metric-distance reconstructions.

Edit ``TESTS`` below and run this script. Each entry identifies a dataset,
scene, and scenario. Images are discovered automatically in
``../dataset/<dataset>/<scene>/images/<scenario>`` and executed in a separate
process through ``reconstruct_cli_metric.py``.
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

# Source directory and output metadata are derived from these three fields:
# ../dataset/<dataset>/<scene>/images/<scenario>
# outputs/metric_<scene>_<scenario>
# subset_label=<scenario>
TESTS = [
    # {"dataset": "custom", "scene": "kettle", "scenario": "scen_1"},
    # {"dataset": "custom", "scene": "kettle", "scenario": "scen_2"},
    # {"dataset": "custom", "scene": "kettle", "scenario": "scen_3"},
    # {"dataset": "custom", "scene": "kettle", "scenario": "scen_4"},
    # {"dataset": "custom", "scene": "spqr", "scenario": "scen_1"},
    # {"dataset": "custom", "scene": "spqr", "scenario": "scen_2"},
    # {"dataset": "custom", "scene": "spqr", "scenario": "scen_3"},
    # {"dataset": "custom", "scene": "spqr", "scenario": "scen_4"},
    # {"dataset": "custom", "scene": "room_1", "scenario": "scen_1"},
    # {"dataset": "custom", "scene": "room_1", "scenario": "scen_2"},
    # {"dataset": "custom", "scene": "room_1", "scenario": "scen_3"},
    # {"dataset": "custom", "scene": "room_1", "scenario": "scen_4"},
    # {"dataset": "custom", "scene": "miscellaneous", "scenario": "scen_1"},
    # {"dataset": "custom", "scene": "miscellaneous", "scenario": "scen_2"},
    # {"dataset": "custom", "scene": "miscellaneous", "scenario": "scen_3"},
    # {"dataset": "custom", "scene": "miscellaneous", "scenario": "scen_4"},
    {"dataset": "custom", "scene": "room_2", "scenario": "scen_1"},
    {"dataset": "custom", "scene": "room_2", "scenario": "scen_2"},
    {"dataset": "custom", "scene": "room_2", "scenario": "scen_3"},
    {"dataset": "custom", "scene": "room_2", "scenario": "scen_4"},
]

# Per-test values override these defaults.
RUNS_PER_TEST = 1
UNCOMPRESSED_NPZ = False
SLEEP_BETWEEN_TESTS_SECONDS = 0
STOP_ON_FAILURE = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run TESTS configured in this file through reconstruct_cli_metric.py."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print commands without executing them.",
    )
    return parser.parse_args()


def normalize_path_component(value, field_name, index):
    component = str(value).strip()
    if not component or Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"TESTS[{index}] has an invalid {field_name}: {component!r}")
    return component


def discover_scenario_images(dataset_root, dataset, scene, scenario, index):
    source_dir = (dataset_root / dataset / scene / "images" / scenario).resolve()
    if not source_dir.is_dir():
        raise ValueError(
            f"TESTS[{index}] source directory does not exist: {source_dir}"
        )

    image_paths = sorted(
        (
            path.resolve()
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not image_paths:
        raise ValueError(
            f"TESTS[{index}] contains no supported images in: {source_dir}"
        )
    return source_dir, image_paths


def validate_test(test, index, dataset_root=None):
    required = {"dataset", "scene", "scenario"}
    missing = sorted(required.difference(test))
    if missing:
        raise ValueError(
            f"TESTS[{index}] is missing required fields: {', '.join(missing)}"
        )

    dataset = normalize_path_component(test["dataset"], "dataset", index)
    scene = normalize_path_component(test["scene"], "scene", index)
    scenario = normalize_path_component(test["scenario"], "scenario", index)
    source_dir, image_paths = discover_scenario_images(
        Path(dataset_root or DATASET_ROOT).expanduser().resolve(),
        dataset,
        scene,
        scenario,
        index,
    )
    test_name = f"metric_{scene}_{scenario}"

    runs = test.get("runs", RUNS_PER_TEST)
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        raise ValueError(f"TESTS[{index}] runs must be a positive integer")
    sleep_seconds = test.get("sleep_after", SLEEP_BETWEEN_TESTS_SECONDS)
    if not isinstance(sleep_seconds, (int, float)) or sleep_seconds < 0:
        raise ValueError(f"TESTS[{index}] sleep_after must be non-negative")

    return {
        **test,
        "dataset": dataset,
        "scene": scene,
        "scenario": scenario,
        "test_name": test_name,
        "scene_name": scene,
        "subset_label": scenario,
        "source_dir": str(source_dir),
        "src": [str(path) for path in image_paths],
        "runs": runs,
        "sleep_after": float(sleep_seconds),
        "uncompressed_npz": bool(test.get("uncompressed_npz", UNCOMPRESSED_NPZ)),
    }


def validate_tests(tests, dataset_root=None):
    normalized = [
        validate_test(test, index, dataset_root=dataset_root)
        for index, test in enumerate(tests)
    ]
    test_names = [test["test_name"] for test in normalized]
    duplicates = sorted({name for name in test_names if test_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate test_name values: {', '.join(duplicates)}")
    return normalized


def build_command(test, script_path, python_executable=None):
    executable = python_executable or sys.executable
    command = [
        str(executable),
        script_path.name,
        "--test-name",
        test["test_name"],
        "--scene-name",
        test["scene_name"],
        "--subset-label",
        test["subset_label"],
        "--src",
        *test["src"],
        "--runs",
        str(test["runs"]),
    ]
    if test["uncompressed_npz"]:
        command.append("--uncompressed-npz")
    return command


def print_summary(results, total_seconds):
    print("\nBatch summary")
    print(f"Total time: {total_seconds:.1f}s")
    for result in results:
        duration = (
            "not run" if result["seconds"] is None else f"{result['seconds']:.1f}s"
        )
        print(f"- {result['test_name']}: {result['status']} ({duration})")


def main():
    args = parse_args()
    script_path = Path(__file__).resolve().with_name("reconstruct_cli_metric.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Metric reconstruction CLI not found: {script_path}")
    if not TESTS:
        print("No metric tests configured. Edit TESTS in this file and run again.")
        return

    tests = validate_tests(TESTS)
    results = []
    batch_start = time.perf_counter()
    exit_code = 0

    for index, test in enumerate(tests, start=1):
        command = build_command(test, script_path)
        print(f"[{index}/{len(tests)}] {shlex.join(command)}")
        if args.dry_run:
            results.append(
                {"test_name": test["test_name"], "status": "DRY RUN", "seconds": None}
            )
            continue

        test_start = time.perf_counter()
        completed = subprocess.run(command, cwd=script_path.parent, check=False)
        test_seconds = time.perf_counter() - test_start
        if completed.returncode == 0:
            status = "OK"
        else:
            status = f"FAILED (exit {completed.returncode})"
            exit_code = completed.returncode
        results.append(
            {
                "test_name": test["test_name"],
                "status": status,
                "seconds": test_seconds,
            }
        )

        if completed.returncode != 0 and STOP_ON_FAILURE:
            break
        if index < len(tests) and test["sleep_after"] > 0:
            print(f"Sleeping {test['sleep_after']:g}s before the next reconstruction")
            time.sleep(test["sleep_after"])

    print_summary(results, time.perf_counter() - batch_start)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
