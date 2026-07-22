#!/usr/bin/env python3
"""Run VGGT image reconstruction from the command line."""

import argparse
import glob
import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from pathlib import Path


MODEL_NAME = "facebook/VGGT-1B"
MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
DEVICE = "cuda"
OUTPUT_ROOT = Path("outputs")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MEMORY_SAMPLE_INTERVAL = 0.05

CONFIDENCE_THRESHOLD = 50.0
FRAME_FILTER = "All"
MASK_BLACK_BACKGROUND = False
MASK_WHITE_BACKGROUND = False
SHOW_CAMERAS = True
MASK_SKY = False
PREDICTION_MODE = "Depthmap and Camera Branch"

# Keep runtime imports lazy so argument parsing does not load the ML stack.
torch = None
VGGT = None
load_and_preprocess_images = None
pose_encoding_to_extri_intri = None
unproject_depth_map_to_point_map = None
predictions_to_glb = None


def import_runtime_dependencies():
    global torch
    global VGGT
    global load_and_preprocess_images
    global pose_encoding_to_extri_intri
    global unproject_depth_map_to_point_map
    global predictions_to_glb

    import torch as torch_module

    from visual_util import predictions_to_glb as export_glb
    from vggt.models.vggt import VGGT as vggt_model
    from vggt.utils.geometry import unproject_depth_map_to_point_map as unproject_depth
    from vggt.utils.load_fn import load_and_preprocess_images as load_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode_pose

    torch = torch_module
    VGGT = vggt_model
    load_and_preprocess_images = load_images
    pose_encoding_to_extri_intri = decode_pose
    unproject_depth_map_to_point_map = unproject_depth
    predictions_to_glb = export_glb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT image-based 3D reconstruction without the Gradio UI."
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
    return parser.parse_args()


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


def postprocess_predictions(predictions, image_shape):
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_shape
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            predictions[key] = value.detach().cpu().numpy().squeeze(0)

    predictions["pose_enc_list"] = None
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    return predictions


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)
        metrics_file.write("\n")


def main():
    args = parse_args()
    overall_start = time.perf_counter()

    validate_test_name(args.test_name)
    image_paths = resolve_sources(args.src)
    output_dir = OUTPUT_ROOT / args.test_name
    output_path = output_dir / "reconstruction.glb"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    import_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    timings = {}
    sampler = MemorySampler()
    sampler.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        print(f"Loading VGGT model: {MODEL_NAME}")
        with timed_stage(timings, "model_load"):
            model = VGGT()
            model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URL))
            model = model.to(DEVICE).eval()

        print(f"Loading {len(image_paths)} image(s)")
        with timed_stage(timings, "image_load"):
            images = load_and_preprocess_images(
                [str(path) for path in image_paths]
            ).to(DEVICE)

        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        print("Running inference")
        with timed_stage(timings, "inference"):
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)

        print("Postprocessing predictions")
        with timed_stage(timings, "postprocess"):
            predictions = postprocess_predictions(predictions, images.shape[-2:])

        print(f"Exporting reconstruction: {output_path}")
        with timed_stage(timings, "export"):
            scene = predictions_to_glb(
                predictions,
                conf_thres=CONFIDENCE_THRESHOLD,
                filter_by_frames=FRAME_FILTER,
                mask_black_bg=MASK_BLACK_BACKGROUND,
                mask_white_bg=MASK_WHITE_BACKGROUND,
                show_cam=SHOW_CAMERAS,
                mask_sky=MASK_SKY,
                target_dir=None,
                prediction_mode=PREDICTION_MODE,
            )
            scene.export(file_obj=str(output_path))
    finally:
        torch.cuda.synchronize()
        sampler.stop()

    timings["total"] = time.perf_counter() - overall_start

    ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_peak_kb = max(
        value for value in (sampler.peak_rss_kb, ru_maxrss_kb) if value is not None
    )
    metrics = {
        "inputs": [str(path) for path in image_paths],
        "output": str(output_path),
        "device": DEVICE,
        "model": MODEL_NAME,
        "timings": timings,
        "memory": {
            "rss_start_mb": kb_to_mb(sampler.start_rss_kb),
            "rss_end_mb": kb_to_mb(sampler.end_rss_kb),
            "rss_peak_mb": kb_to_mb(rss_peak_kb),
        },
        "cuda": get_cuda_memory(),
    }
    write_json(metrics_path, metrics)

    print(f"Saved reconstruction: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
