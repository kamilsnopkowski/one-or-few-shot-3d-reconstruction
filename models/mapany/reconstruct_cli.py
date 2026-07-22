#!/usr/bin/env python3

"""Run image-only MapAnything reconstruction from the command line."""

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

# Keep runtime imports lazy so argument parsing does not require the ML environment.
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
        description="Run MapAnything image-only 3D reconstruction without Gradio."
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
    return parser.parse_args()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--runs must be at least 1")
    return parsed


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


def prepare_export_predictions(outputs):
    world_points = []
    images = []
    masks = []

    for prediction in outputs:
        depth = prediction["depth_z"][0].squeeze(-1)
        intrinsics = prediction["intrinsics"][0]
        camera_pose = prediction["camera_poses"][0]
        points, valid_mask = depthmap_to_world_frame(depth, intrinsics, camera_pose)

        if prediction.get("mask") is None:
            mask = np.ones_like(depth.detach().cpu().numpy(), dtype=bool)
        else:
            mask = (
                prediction["mask"][0]
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
        mask &= valid_mask.detach().cpu().numpy()

        world_points.append(points.detach().cpu().numpy())
        images.append(prediction["img_no_norm"][0].detach().cpu().numpy())
        masks.append(mask)

    return {
        "world_points": np.stack(world_points),
        "images": np.stack(images),
        "final_masks": np.stack(masks),
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)
        metrics_file.write("\n")


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


def run_reconstruction(model, image_paths, output_path, run_number, total_runs):
    timings = {}
    sampler = MemorySampler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler.start()
    pipeline_start = time.perf_counter()

    try:
        print(f"Run {run_number}/{total_runs}: loading {len(image_paths)} image(s)")
        with timed_stage(timings, "image_load"):
            views = load_images(
                [str(path) for path in image_paths],
                resize_mode="fixed_mapping",
                size=None,
                norm_type="dinov2",
                patch_size=14,
                verbose=False,
                resolution_set=518,
                stride=1,
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
            export_predictions = prepare_export_predictions(outputs)

        print(f"Run {run_number}/{total_runs}: exporting reconstruction: {output_path}")
        with timed_stage(timings, "export"):
            scene = predictions_to_glb(export_predictions, as_mesh=False)
            scene.export(str(output_path))
    finally:
        torch.cuda.synchronize()
        timings["pipeline_total"] = time.perf_counter() - pipeline_start
        sampler.stop()

    return {
        "run": run_number,
        "timings": timings,
        "memory": sampler.as_dict(),
        "cuda": get_cuda_memory(),
    }


def main():
    process_start = time.perf_counter()
    args = parse_args()

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
    runs = [
        run_reconstruction(model, image_paths, output_path, run_number, args.runs)
        for run_number in range(1, args.runs + 1)
    ]
    total_benchmark = time.perf_counter() - benchmark_start

    timing_summary = summarize_metric_dicts([run["timings"] for run in runs])
    memory_summary = summarize_metric_dicts([run["memory"] for run in runs])
    cuda_summary = summarize_metric_dicts([run["cuda"] for run in runs])

    timings = {
        "runtime_imports": runtime_imports,
        "model_load": setup_timings["model_load"],
        "pipeline_total_median": timing_summary["pipeline_total"]["median"],
        "total_benchmark": total_benchmark,
        "total_process": time.perf_counter() - process_start,
    }
    metrics = {
        "inputs": [str(path) for path in image_paths],
        "output": str(output_path),
        "device": DEVICE,
        "model": MODEL_NAME,
        "benchmark": {
            "mode": "default",
            "runs": args.runs,
            "timing_definitions": {
                "runtime_imports": "Lazy runtime dependency imports.",
                "model_load": "Model construction, weight loading, device transfer, and eval setup.",
                "pipeline_total": "One reconstruction run: image_load + inference + postprocess + export.",
                "total_benchmark": "All repeated reconstruction runs, excluding runtime_imports and model_load.",
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

    print(f"Saved reconstruction: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()