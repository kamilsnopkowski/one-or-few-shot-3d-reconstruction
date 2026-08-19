#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Run MapAnything and export a self-contained metric measurement package.

This is a separate entrypoint for the physical-distance experiment. It leaves
``reconstruct_cli_eval.py`` unchanged and reuses its inference implementation.
In addition to the regular visualization and point cloud, it stores the exact
processed RGB images alongside the per-pixel 3D points. Consequently, clicks
in ``scripts/measure_metric_distance.py`` address the same pixel grid as the
predicted geometry without remapping coordinates from the original image.
"""

import argparse
import statistics
import time
from pathlib import Path

import reconstruct_cli_eval as base

EXPERIMENT_VIEW_COUNTS = {1, 2, 4, 8}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run image-only MapAnything reconstruction and export a "
            "self-contained package for physical distance measurements."
        )
    )
    parser.add_argument(
        "--test-name",
        required=True,
        help="Name used for the output folder under outputs/.",
    )
    parser.add_argument(
        "--scene-name",
        help="Stable scene identifier stored in the experiment archive.",
    )
    parser.add_argument(
        "--subset-label",
        help="Optional label such as n1, n2, n4, or n8.",
    )
    parser.add_argument(
        "--src",
        nargs="+",
        required=True,
        help="Input image files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--runs",
        type=base.positive_int,
        default=1,
        help="Number of inference repetitions. One is recommended for this experiment.",
    )
    parser.add_argument(
        "--uncompressed-npz",
        action="store_true",
        help="Write metric_predictions.npz without compression for faster export.",
    )
    return parser.parse_args()


def processed_images_to_uint8(images):
    return base.np.stack([base.image_to_uint8(image) for image in images])


def write_processed_images(output_dir, image_names, processed_images):
    from PIL import Image

    image_dir = output_dir / "processed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_names = []
    for index, (image_name, image) in enumerate(zip(image_names, processed_images)):
        safe_stem = Path(str(image_name)).stem.replace(" ", "_")
        output_name = f"{index:02d}_{safe_stem}.png"
        Image.fromarray(image).save(image_dir / output_name)
        output_names.append(output_name)
    return image_dir, output_names


def export_metric_artifacts(
    output_dir,
    image_paths,
    export_predictions,
    evaluation_predictions,
    *,
    scene_name,
    subset_label,
    compress_npz,
):
    ply_path = output_dir / "reconstruction_raw.ply"
    npz_path = output_dir / "metric_predictions.npz"
    selected_images_path = output_dir / "selected_images.txt"
    manifest_path = output_dir / "metric_experiment_manifest.json"

    point_count = base.write_binary_point_cloud_ply(
        ply_path,
        export_predictions["world_points"],
        export_predictions["images"],
        export_predictions["final_masks"],
    )

    processed_images = processed_images_to_uint8(export_predictions["images"])
    image_names = [path.name for path in image_paths]
    processed_image_dir, processed_image_names = write_processed_images(
        output_dir,
        image_names,
        processed_images,
    )

    npz_payload = dict(evaluation_predictions)
    npz_payload.update(
        {
            "processed_images_uint8": processed_images,
            "image_paths": base.np.asarray([str(path) for path in image_paths]),
            "image_names": base.np.asarray(image_names),
            "original_image_sizes_wh": base.read_original_image_sizes(image_paths),
            "camera_pose_convention": base.np.asarray(
                "OpenCV_cam2world_x_right_y_down_z_forward"
            ),
            "world_points_definition": base.np.asarray(
                "depth_z unprojected with predicted intrinsics and camera_poses"
            ),
            "distance_units": base.np.asarray("meters"),
            "metric_scale_already_applied": base.np.asarray(True),
            "experiment_scene_name": base.np.asarray(scene_name),
            "experiment_subset_label": base.np.asarray(subset_label),
            "experiment_view_count": base.np.asarray(
                len(image_paths),
                dtype=base.np.int32,
            ),
        }
    )
    save_npz = base.np.savez_compressed if compress_npz else base.np.savez
    save_npz(npz_path, **npz_payload)
    base.write_selected_images(selected_images_path, image_paths)

    manifest = {
        "experiment": {
            "type": "physical_metric_distance",
            "scene_name": scene_name,
            "subset_label": subset_label,
            "view_count": len(image_paths),
            "recommended_view_counts": sorted(EXPERIMENT_VIEW_COUNTS),
            "distance_units": "meters",
            "scale_policy": (
                "Use raw world_points directly. The model metric scale is already "
                "applied; do not perform Sim(3) alignment or multiply by "
                "metric_scaling_factor again."
            ),
        },
        "model": base.MODEL_NAME,
        "image_count": len(image_paths),
        "point_count": point_count,
        "coordinate_system": {
            "camera_poses": "OpenCV camera-to-world (+X right, +Y down, +Z forward)",
            "world_points": "MapAnything predicted world frame without alignment",
            "units": "meters (predicted metric scale)",
        },
        "preprocessing": base.PREPROCESSING,
        "files": {
            "visualization_glb": "reconstruction.glb",
            "raw_point_cloud_ply": ply_path.name,
            "prediction_archive": npz_path.name,
            "processed_images_directory": processed_image_dir.name,
            "processed_images": processed_image_names,
            "selected_images": selected_images_path.name,
            "runtime_metrics": "metrics.json",
            "measurement_results": "metric_measurements.csv",
        },
        "npz_arrays": base.describe_npz_payload(npz_payload),
    }
    base.write_json(manifest_path, manifest)
    return {
        "ply": str(ply_path),
        "npz": str(npz_path),
        "processed_images": str(processed_image_dir),
        "selected_images": str(selected_images_path),
        "manifest": str(manifest_path),
        "point_count": point_count,
    }


def main():
    process_start = time.perf_counter()
    args = parse_args()
    base.validate_test_name(args.test_name)

    image_paths = base.resolve_sources(args.src)
    view_count = len(image_paths)
    scene_name = args.scene_name or args.test_name
    subset_label = args.subset_label or f"n{view_count}"
    if view_count not in EXPERIMENT_VIEW_COUNTS:
        print(
            f"Warning: the planned experiment uses 1, 2, 4, or 8 images; "
            f"received {view_count}."
        )

    output_dir = base.OUTPUT_ROOT / args.test_name
    output_path = output_dir / "reconstruction.glb"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_import_start = time.perf_counter()
    base.import_runtime_dependencies()
    runtime_imports = time.perf_counter() - runtime_import_start
    if not base.torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    setup_timings = {}
    model_sampler = base.MemorySampler()
    base.torch.cuda.empty_cache()
    base.torch.cuda.reset_peak_memory_stats()
    model_sampler.start()
    try:
        print(f"Loading MapAnything model: {base.MODEL_NAME}")
        with base.timed_stage(setup_timings, "model_load"):
            model = base.MapAnything.from_pretrained(base.MODEL_NAME).to(base.DEVICE).eval()
    finally:
        base.torch.cuda.synchronize()
        model_sampler.stop()

    model_load_memory = model_sampler.as_dict()
    model_load_cuda = base.get_cuda_memory()

    benchmark_start = time.perf_counter()
    runs = []
    final_evaluation_data = None
    for run_number in range(1, args.runs + 1):
        run, evaluation_data = base.run_reconstruction(
            model,
            image_paths,
            output_path,
            run_number,
            args.runs,
            capture_evaluation_data=run_number == args.runs,
        )
        runs.append(run)
        if evaluation_data is not None:
            final_evaluation_data = evaluation_data
    total_benchmark = time.perf_counter() - benchmark_start

    if final_evaluation_data is None:
        raise RuntimeError("Final inference did not return evaluation data")

    print(f"Exporting metric experiment artifacts to: {output_dir}")
    artifact_export_start = time.perf_counter()
    export_predictions, evaluation_predictions = final_evaluation_data
    artifact_outputs = export_metric_artifacts(
        output_dir,
        image_paths,
        export_predictions,
        evaluation_predictions,
        scene_name=scene_name,
        subset_label=subset_label,
        compress_npz=not args.uncompressed_npz,
    )
    artifact_export = time.perf_counter() - artifact_export_start

    timing_summary = base.summarize_metric_dicts([run["timings"] for run in runs])
    memory_summary = base.summarize_metric_dicts([run["memory"] for run in runs])
    cuda_summary = base.summarize_metric_dicts([run["cuda"] for run in runs])
    timings = {
        "runtime_imports": runtime_imports,
        "model_load": setup_timings["model_load"],
        "pipeline_total_median": statistics.median(
            [run["timings"]["pipeline_total"] for run in runs]
        ),
        "total_benchmark": total_benchmark,
        "metric_artifact_export": artifact_export,
        "total_process": time.perf_counter() - process_start,
    }
    metrics = {
        "inputs": [str(path) for path in image_paths],
        "outputs": {"glb": str(output_path), **artifact_outputs},
        "experiment": {
            "scene_name": scene_name,
            "subset_label": subset_label,
            "view_count": view_count,
        },
        "device": base.DEVICE,
        "model": base.MODEL_NAME,
        "benchmark": {
            "mode": "metric_distance_experiment_export",
            "runs": args.runs,
        },
        "timings": timings,
        "timing_summary": timing_summary,
        "memory": {
            "model_load": model_load_memory,
            "runs_summary": memory_summary,
            "process": base.get_process_memory(),
        },
        "cuda": {
            "model_load": model_load_cuda,
            "runs_summary": cuda_summary,
        },
        "runs": runs,
    }
    base.write_json(metrics_path, metrics)

    print(f"Saved visualization: {output_path}")
    print(f"Saved point cloud: {artifact_outputs['ply']}")
    print(f"Saved metric predictions: {artifact_outputs['npz']}")
    print(f"Saved processed images: {artifact_outputs['processed_images']}")
    print(f"Saved experiment manifest: {artifact_outputs['manifest']}")
    print(f"Saved runtime metrics: {metrics_path}")


if __name__ == "__main__":
    main()
