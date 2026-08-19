#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Run MapAnything and export artifacts suitable for geometric evaluation.

In addition to the visualization GLB and performance metrics, this entrypoint
stores a PCL-compatible binary PLY point cloud and an NPZ archive containing
the predicted camera poses, intrinsics, depths, masks, confidence values, and
per-pixel world points. Camera poses follow MapAnything's OpenCV cam2world
convention: +X right, +Y down, +Z forward.
"""

import argparse
import glob
import json
import os
import resource
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "facebook/map-anything"
DEVICE = "cuda"
OUTPUT_ROOT = Path("outputs")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MEMORY_SAMPLE_INTERVAL = 0.05
PREPROCESSING = {
    "resize_mode": "fixed_mapping",
    "size": None,
    "norm_type": "dinov2",
    "patch_size": 14,
    "resolution_set": 518,
    "stride": 1,
}

# Runtime dependencies stay lazy so --help and syntax checks work without the
# MapAnything environment being active.
np = None
torch = None
MapAnything = None
depthmap_to_world_frame = None
load_images = None
predictions_to_glb = None


def import_runtime_dependencies():
    global np
    global torch
    global MapAnything
    global depthmap_to_world_frame
    global load_images
    global predictions_to_glb

    import numpy as np_module
    import torch as torch_module

    from mapanything.models import MapAnything as mapanything_model
    from mapanything.utils.geometry import depthmap_to_world_frame as depth_to_world
    from mapanything.utils.image import load_images as load_image_views
    from mapanything.utils.viz import predictions_to_glb as export_glb

    np = np_module
    torch = torch_module
    MapAnything = mapanything_model
    depthmap_to_world_frame = depth_to_world
    load_images = load_image_views
    predictions_to_glb = export_glb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run image-only MapAnything reconstruction and export GLB, binary "
            "PLY, camera/depth NPZ, a manifest, and benchmark metrics."
        )
    )
    parser.add_argument(
        "--test-name",
        required=True,
        help="Name used for the output folder under outputs/.",
    )
    parser.add_argument(
        "--src",
        nargs="+",
        required=True,
        help="Input image files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=3,
        help="Number of benchmark repetitions after the model is loaded.",
    )
    parser.add_argument(
        "--uncompressed-npz",
        action="store_true",
        help="Write predictions.npz without compression for faster export.",
    )
    return parser.parse_args()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--runs must be at least 1")
    return parsed


def validate_test_name(test_name):
    if not test_name or Path(test_name).name != test_name or test_name in {".", ".."}:
        raise ValueError("--test-name must be a single non-empty folder name")


def resolve_sources(source_values):
    image_paths = []
    missing = []

    for value in source_values:
        expanded = os.path.expanduser(value)
        matches = glob.glob(expanded, recursive=True) if glob.has_magic(expanded) else []
        candidates = matches if matches else [expanded]

        for candidate in candidates:
            path = Path(candidate)
            if not path.exists():
                missing.append(str(path))
                continue

            if path.is_dir():
                image_paths.extend(
                    child.resolve()
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path.resolve())

    unique_paths = sorted(set(image_paths), key=str)
    if not unique_paths:
        detail = f" Missing inputs: {missing}" if missing else ""
        raise ValueError(f"No supported image files found in --src.{detail}")
    return unique_paths


def read_rss_kb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def kb_to_mb(value):
    return None if value is None else value / 1024.0


class MemorySampler:
    def __init__(self):
        self.start_rss_kb = read_rss_kb()
        self.end_rss_kb = None
        self.peak_rss_kb = self.start_rss_kb
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._sample()
        self.end_rss_kb = read_rss_kb()

    def _run(self):
        while not self._stop_event.wait(MEMORY_SAMPLE_INTERVAL):
            self._sample()

    def _sample(self):
        current = read_rss_kb()
        if current is not None and (
            self.peak_rss_kb is None or current > self.peak_rss_kb
        ):
            self.peak_rss_kb = current

    def as_dict(self):
        return {
            "rss_start_mb": kb_to_mb(self.start_rss_kb),
            "rss_end_mb": kb_to_mb(self.end_rss_kb),
            "rss_peak_mb": kb_to_mb(self.peak_rss_kb),
        }


@contextmanager
def timed_stage(timings, name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start


def get_cuda_memory():
    return {
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def tensor_to_numpy(value, *, batch_index=0, squeeze_last=False, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if batch_index is not None and value.ndim > 0:
        value = value[batch_index]
    if squeeze_last and value.ndim > 0 and value.shape[-1] == 1:
        value = value[..., 0]
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def prepare_predictions(outputs, views):
    world_points = []
    images = []
    masks = []
    depths = []
    intrinsics = []
    camera_poses = []
    confidences = []
    metric_scaling_factors = []
    processed_sizes_hw = []
    input_true_shapes_hw = []

    for prediction, view in zip(outputs, views):
        depth_tensor = prediction["depth_z"][0].squeeze(-1)
        intrinsics_tensor = prediction["intrinsics"][0]
        camera_pose_tensor = prediction["camera_poses"][0]
        points_tensor, valid_mask_tensor = depthmap_to_world_frame(
            depth_tensor,
            intrinsics_tensor,
            camera_pose_tensor,
        )

        depth = depth_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        points = points_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        valid_mask = valid_mask_tensor.detach().cpu().numpy().astype(bool, copy=False)

        raw_mask = tensor_to_numpy(
            prediction.get("mask"),
            squeeze_last=True,
        )
        if raw_mask is None:
            mask = np.ones_like(depth, dtype=bool)
        else:
            mask = raw_mask.astype(bool, copy=False)
        mask &= valid_mask
        mask &= np.isfinite(points).all(axis=-1)
        mask &= np.isfinite(depth)
        mask &= depth > 0

        confidence = tensor_to_numpy(
            prediction.get("conf"),
            squeeze_last=True,
            dtype=np.float32,
        )
        if confidence is None:
            confidence = np.full_like(depth, np.nan, dtype=np.float32)

        scaling_factor = tensor_to_numpy(
            prediction.get("metric_scaling_factor"),
            dtype=np.float32,
        )
        if scaling_factor is None:
            scaling_factor = np.array(np.nan, dtype=np.float32)
        scaling_factor = np.asarray(scaling_factor, dtype=np.float32).reshape(-1)[0]

        image = tensor_to_numpy(
            prediction["img_no_norm"],
            dtype=np.float32,
        )
        intrinsic = intrinsics_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        camera_pose = camera_pose_tensor.detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )

        true_shape = np.asarray(view.get("true_shape", [[depth.shape[0], depth.shape[1]]]))
        true_shape = true_shape.reshape(-1, 2)[0].astype(np.int32, copy=False)

        world_points.append(points)
        images.append(image)
        masks.append(mask)
        depths.append(depth)
        intrinsics.append(intrinsic)
        camera_poses.append(camera_pose)
        confidences.append(confidence)
        metric_scaling_factors.append(scaling_factor)
        processed_sizes_hw.append(np.asarray(depth.shape, dtype=np.int32))
        input_true_shapes_hw.append(true_shape)

    export_predictions = {
        "world_points": np.stack(world_points),
        "images": np.stack(images),
        "final_masks": np.stack(masks),
    }
    evaluation_predictions = {
        "world_points": export_predictions["world_points"],
        "final_masks": export_predictions["final_masks"],
        "depth_z": np.stack(depths),
        "intrinsics": np.stack(intrinsics),
        "camera_poses": np.stack(camera_poses),
        "confidence": np.stack(confidences),
        "metric_scaling_factor": np.asarray(
            metric_scaling_factors,
            dtype=np.float32,
        ),
        "processed_image_sizes_hw": np.stack(processed_sizes_hw),
        "input_true_shapes_hw": np.stack(input_true_shapes_hw),
    }
    return export_predictions, evaluation_predictions


def read_original_image_sizes(image_paths):
    from PIL import Image, ImageOps

    sizes = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                sizes.append(image.size)
        except OSError:
            sizes.append((-1, -1))
    return np.asarray(sizes, dtype=np.int32)


def image_to_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    finite = image[np.isfinite(image)]
    if finite.size and finite.max() <= 1.5:
        image = image * 255.0
    return np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0).clip(
        0,
        255,
    ).astype(np.uint8)


def write_binary_point_cloud_ply(path, world_points, images, masks):
    valid_masks = []
    point_count = 0
    for points, mask in zip(world_points, masks):
        valid = np.asarray(mask, dtype=bool) & np.isfinite(points).all(axis=-1)
        valid_masks.append(valid)
        point_count += int(valid.sum())

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated by reconstruct_cli_eval.py\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as ply_file:
        ply_file.write(header)
        for points, image, valid in zip(world_points, images, valid_masks):
            selected_points = np.asarray(points[valid], dtype="<f4")
            selected_colors = image_to_uint8(image)[valid, :3]
            vertices = np.empty(len(selected_points), dtype=vertex_dtype)
            vertices["x"] = selected_points[:, 0]
            vertices["y"] = selected_points[:, 1]
            vertices["z"] = selected_points[:, 2]
            vertices["red"] = selected_colors[:, 0]
            vertices["green"] = selected_colors[:, 1]
            vertices["blue"] = selected_colors[:, 2]
            vertices.tofile(ply_file)

    return point_count


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


def write_selected_images(path, image_paths):
    with open(path, "w", encoding="utf-8") as output_file:
        for image_path in image_paths:
            output_file.write(f"{image_path.name}\n")


def describe_npz_payload(payload):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in payload.items()
        if isinstance(value, np.ndarray)
    }


def export_evaluation_artifacts(
    output_dir,
    image_paths,
    export_predictions,
    evaluation_predictions,
    compress_npz,
):
    ply_path = output_dir / "reconstruction_raw.ply"
    npz_path = output_dir / "predictions.npz"
    selected_images_path = output_dir / "selected_images.txt"
    manifest_path = output_dir / "evaluation_manifest.json"

    point_count = write_binary_point_cloud_ply(
        ply_path,
        export_predictions["world_points"],
        export_predictions["images"],
        export_predictions["final_masks"],
    )

    npz_payload = dict(evaluation_predictions)
    npz_payload.update(
        {
            "image_paths": np.asarray([str(path) for path in image_paths]),
            "image_names": np.asarray([path.name for path in image_paths]),
            "original_image_sizes_wh": read_original_image_sizes(image_paths),
            "camera_pose_convention": np.asarray(
                "OpenCV_cam2world_x_right_y_down_z_forward"
            ),
            "world_points_definition": np.asarray(
                "depth_z unprojected with predicted intrinsics and camera_poses"
            ),
        }
    )
    save_npz = np.savez_compressed if compress_npz else np.savez
    save_npz(npz_path, **npz_payload)
    write_selected_images(selected_images_path, image_paths)

    manifest = {
        "model": MODEL_NAME,
        "image_count": len(image_paths),
        "point_count": point_count,
        "coordinate_system": {
            "camera_poses": "OpenCV camera-to-world (+X right, +Y down, +Z forward)",
            "world_points": "MapAnything predicted world frame before any evaluation alignment",
            "units": "metric prediction; estimate and report Sim(3) scale against ETH3D",
        },
        "preprocessing": PREPROCESSING,
        "files": {
            "visualization_glb": "reconstruction.glb",
            "raw_point_cloud_ply": ply_path.name,
            "prediction_archive": npz_path.name,
            "selected_images": selected_images_path.name,
            "benchmark_metrics": "metrics.json",
        },
        "npz_arrays": describe_npz_payload(npz_payload),
    }
    write_json(manifest_path, manifest)
    return {
        "ply": str(ply_path),
        "npz": str(npz_path),
        "selected_images": str(selected_images_path),
        "manifest": str(manifest_path),
        "point_count": point_count,
    }


def get_process_memory():
    return {
        "rss_peak_mb": kb_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    }


def summarize_values(values):
    available = [value for value in values if value is not None]
    if not available:
        return {"median": None, "min": None, "max": None, "values": values}
    return {
        "median": statistics.median(available),
        "min": min(available),
        "max": max(available),
        "values": values,
    }


def summarize_metric_dicts(metric_dicts):
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    return {
        key: summarize_values([metric_dict.get(key) for metric_dict in metric_dicts])
        for key in keys
    }


def run_reconstruction(
    model,
    image_paths,
    output_path,
    run_number,
    total_runs,
    capture_evaluation_data,
):
    timings = {}
    sampler = MemorySampler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler.start()
    pipeline_start = time.perf_counter()
    evaluation_data = None

    try:
        print(f"Run {run_number}/{total_runs}: loading {len(image_paths)} image(s)")
        with timed_stage(timings, "image_load"):
            views = load_images(
                [str(path) for path in image_paths],
                **PREPROCESSING,
                verbose=False,
            )

        print(f"Run {run_number}/{total_runs}: running inference")
        with timed_stage(timings, "inference"):
            with torch.no_grad():
                outputs = model.infer(
                    views,
                    memory_efficient_inference=True,
                    minibatch_size=1,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                )

        print(f"Run {run_number}/{total_runs}: postprocessing predictions")
        with timed_stage(timings, "postprocess"):
            export_predictions, evaluation_predictions = prepare_predictions(
                outputs,
                views,
            )

        print(f"Run {run_number}/{total_runs}: exporting visualization: {output_path}")
        with timed_stage(timings, "export"):
            scene = predictions_to_glb(export_predictions, as_mesh=False)
            scene.export(str(output_path))

        if capture_evaluation_data:
            evaluation_data = (export_predictions, evaluation_predictions)
    finally:
        torch.cuda.synchronize()
        timings["pipeline_total"] = time.perf_counter() - pipeline_start
        sampler.stop()

    result = {
        "run": run_number,
        "timings": timings,
        "memory": sampler.as_dict(),
        "cuda": get_cuda_memory(),
    }
    return result, evaluation_data


def main():
    process_start = time.perf_counter()
    args = parse_args()
    validate_test_name(args.test_name)

    image_paths = resolve_sources(args.src)
    output_dir = OUTPUT_ROOT / args.test_name
    output_path = output_dir / "reconstruction.glb"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_import_start = time.perf_counter()
    import_runtime_dependencies()
    runtime_imports = time.perf_counter() - runtime_import_start
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    setup_timings = {}
    model_sampler = MemorySampler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_sampler.start()
    try:
        print(f"Loading MapAnything model: {MODEL_NAME}")
        with timed_stage(setup_timings, "model_load"):
            model = MapAnything.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    finally:
        torch.cuda.synchronize()
        model_sampler.stop()

    model_load_memory = model_sampler.as_dict()
    model_load_cuda = get_cuda_memory()

    benchmark_start = time.perf_counter()
    runs = []
    final_evaluation_data = None
    for run_number in range(1, args.runs + 1):
        run, evaluation_data = run_reconstruction(
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

    print(f"Exporting evaluation artifacts to: {output_dir}")
    artifact_export_start = time.perf_counter()
    export_predictions, evaluation_predictions = final_evaluation_data
    artifact_outputs = export_evaluation_artifacts(
        output_dir,
        image_paths,
        export_predictions,
        evaluation_predictions,
        compress_npz=not args.uncompressed_npz,
    )
    artifact_export = time.perf_counter() - artifact_export_start

    timing_summary = summarize_metric_dicts([run["timings"] for run in runs])
    memory_summary = summarize_metric_dicts([run["memory"] for run in runs])
    cuda_summary = summarize_metric_dicts([run["cuda"] for run in runs])

    timings = {
        "runtime_imports": runtime_imports,
        "model_load": setup_timings["model_load"],
        "pipeline_total_median": timing_summary["pipeline_total"]["median"],
        "total_benchmark": total_benchmark,
        "evaluation_artifact_export": artifact_export,
        "total_process": time.perf_counter() - process_start,
    }
    metrics = {
        "inputs": [str(path) for path in image_paths],
        "outputs": {
            "glb": str(output_path),
            **artifact_outputs,
        },
        "device": DEVICE,
        "model": MODEL_NAME,
        "benchmark": {
            "mode": "evaluation_export",
            "runs": args.runs,
            "timing_definitions": {
                "runtime_imports": "Lazy runtime dependency imports.",
                "model_load": "Model construction, weight loading, device transfer, and eval setup.",
                "pipeline_total": "One reconstruction run: image_load + inference + postprocess + GLB export.",
                "total_benchmark": "All repeated reconstruction runs, excluding runtime_imports, model_load, and evaluation artifact export.",
                "evaluation_artifact_export": "Final binary PLY, NPZ, image list, and manifest export.",
                "total_process": "CLI main execution through metrics payload creation.",
            },
        },
        "timings": timings,
        "timing_summary": timing_summary,
        "memory": {
            "model_load": model_load_memory,
            "runs_summary": memory_summary,
            "process": get_process_memory(),
        },
        "cuda": {
            "model_load": model_load_cuda,
            "runs_summary": cuda_summary,
        },
        "runs": runs,
    }
    write_json(metrics_path, metrics)

    print(f"Saved visualization: {output_path}")
    print(f"Saved point cloud: {artifact_outputs['ply']}")
    print(f"Saved predictions: {artifact_outputs['npz']}")
    print(f"Saved manifest: {artifact_outputs['manifest']}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
