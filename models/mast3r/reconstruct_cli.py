# --------------------------------------------------------
# CLI reconstruction entrypoint for MASt3R
# --------------------------------------------------------
import argparse
import contextlib
import glob
import importlib.util
import json
import resource
import threading
import time
from pathlib import Path


MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
WEIGHTS_PATH = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEVICE = "cuda"
OUTPUT_ROOT = Path("outputs")


def supported_image_extensions():
    extensions = {".jpeg", ".jpg", ".png"}
    if importlib.util.find_spec("pillow_heif") is not None:
        extensions.update({".heic", ".heif"})
    return extensions


def _rss_bytes_procfs():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _ru_maxrss_bytes():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (AttributeError, OSError):
        return None


class MemorySampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.start_rss_bytes = None
        self.end_rss_bytes = None
        self.peak_rss_bytes = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.start_rss_bytes = _rss_bytes_procfs()
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.end_rss_bytes = _rss_bytes_procfs()
        peaks = [
            value
            for value in (self.peak_rss_bytes, self.end_rss_bytes, _ru_maxrss_bytes())
            if value is not None
        ]
        self.peak_rss_bytes = max(peaks, default=None)

    def _sample(self):
        while not self._stop.is_set():
            rss = _rss_bytes_procfs()
            if rss is not None:
                self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            self._stop.wait(self.interval)

    def as_dict(self):
        return {
            "rss_start_bytes": self.start_rss_bytes,
            "rss_end_bytes": self.end_rss_bytes,
            "rss_peak_bytes": self.peak_rss_bytes,
        }


class Timer:
    def __init__(self, synchronize_cuda):
        self.synchronize_cuda = synchronize_cuda
        self.sections = {}

    @contextlib.contextmanager
    def section(self, name, uses_cuda=False):
        if uses_cuda:
            self.synchronize_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            if uses_cuda:
                self.synchronize_cuda()
            self.sections[name] = time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(
        prog="reconstruct_cli",
        description="Run MASt3R reconstruction from image paths without Gradio.",
    )
    parser.add_argument("--test-name", required=True, help="Name for this run under outputs/.")
    parser.add_argument(
        "--src",
        nargs="+",
        required=True,
        help="Image files, directories, or glob patterns.",
    )
    return parser.parse_args()


def resolve_sources(src_values):
    extensions = supported_image_extensions()
    paths = []

    for value in src_values:
        pattern = str(Path(value).expanduser())
        matches = sorted(glob.glob(pattern, recursive=True))
        candidates = matches if matches else [pattern]

        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                children = sorted(path.rglob("*"), key=lambda item: item.as_posix())
                paths.extend(
                    child
                    for child in children
                    if child.is_file() and child.suffix.lower() in extensions
                )
            elif path.is_file() and path.suffix.lower() in extensions:
                paths.append(path)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    if not unique_paths:
        supported = ", ".join(sorted(extensions))
        raise SystemExit(f"No supported images found from --src. Supported extensions: {supported}")
    return unique_paths


def validate_test_name(test_name):
    if test_name in {"", ".", ".."} or Path(test_name).name != test_name:
        raise SystemExit("--test-name must be a single directory name.")


def mb_or_none(value):
    return None if value is None else round(value / (1024**2), 2)


def main():
    total_start = time.perf_counter()
    args = parse_args()
    validate_test_name(args.test_name)
    image_paths = resolve_sources(args.src)

    run_dir = (OUTPUT_ROOT / args.test_name).resolve()
    output_path = run_dir / "reconstruction.glb"
    metrics_path = run_dir / "metrics.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch

    import mast3r.utils.path_to_dust3r  # noqa
    from mast3r.model import AsymmetricMASt3R
    from mast3r.reconstruction import (
        export_scene_to_glb,
        load_reconstruction_images,
        prepare_scene_for_export,
        reconstruct_scene,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for MASt3R reconstruction, but no CUDA device is available.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timer = Timer(lambda: torch.cuda.synchronize(DEVICE))

    with MemorySampler() as memory:
        with timer.section("model_load", uses_cuda=True):
            model = AsymmetricMASt3R.from_pretrained(WEIGHTS_PATH).to(DEVICE)

        with timer.section("image_load"):
            imgs, inference_paths = load_reconstruction_images(
                image_paths,
                image_size=512,
                silent=False,
            )

        with timer.section("inference", uses_cuda=True):
            scene_state = reconstruct_scene(
                outdir=str(run_dir),
                gradio_delete_cache=1,
                model=model,
                retrieval_model=None,
                device=DEVICE,
                current_scene_state=None,
                filelist=inference_paths,
                imgs=imgs,
                optim_level="refine+depth",
                lr1=0.07,
                niter1=300,
                lr2=0.01,
                niter2=300,
                matching_conf_thr=0.0,
                scenegraph_type="complete",
                winsize=1,
                win_cyclic=False,
                refid=0,
                shared_intrinsics=False,
            )

        with timer.section("postprocess", uses_cuda=True):
            prepared_scene = prepare_scene_for_export(
                scene_state,
                min_conf_thr=1.5,
                clean_depth=True,
                TSDF_thresh=0.0,
            )
            if prepared_scene is None:
                raise RuntimeError("Reconstruction did not produce a scene.")

        with timer.section("export"):
            export_scene_to_glb(
                str(output_path),
                *prepared_scene,
                cam_size=0.2,
                as_pointcloud=True,
                transparent_cams=False,
                silent=False,
            )

        scene_state.cleanup()

    timings = dict(timer.sections)
    memory_metrics = {
        key.replace("_bytes", "_mb"): mb_or_none(value)
        for key, value in memory.as_dict().items()
    }
    cuda_metrics = {
        "max_allocated_mb": mb_or_none(torch.cuda.max_memory_allocated()),
        "max_reserved_mb": mb_or_none(torch.cuda.max_memory_reserved()),
    }
    metrics = {
        "inputs": image_paths,
        "output": str(output_path),
        "device": DEVICE,
        "model": MODEL_ID,
        "timings": timings,
        "memory": memory_metrics,
        "cuda": cuda_metrics,
    }
    metrics["timings"]["total"] = time.perf_counter() - total_start

    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print("Wrote", output_path)
    print("Wrote metrics", metrics_path)


if __name__ == "__main__":
    main()
