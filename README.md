# Supplementary Materials for the Master's Thesis

This repository contains supplementary materials for a master's thesis on one-shot and few-shot 3D reconstruction. It brings together the experiment scripts, dataset documentation, collected measurements, and result spreadsheets used to evaluate and compare the selected reconstruction methods.

The repository does not provide complete runtime environments, model implementations, or model weights. The included scripts should be copied into the appropriate model repositories or working directories and executed in environments that meet the requirements specified by the model authors.

## Contents

The materials cover three analyzed 3D reconstruction methods: VGGT, MapAnything, and MASt3R. For each method, the repository provides scripts for individual reconstructions and batch experiments on predefined scenes.

It also includes method-specific result spreadsheets, a combined performance comparison spreadsheet, the results of the MapAnything point-to-point distance experiment, and [documentation of the photo subsets](./DATASET_PHOTO_SETS.md) used in every experiment scenario. The document lists the exact input photographs for each dataset, scene, and scenario, covering all 38 scenes and the one-, two-, four-, and eight-image configurations used in the experiments.

## Performance Analysis

The experiment scripts and result spreadsheets document the runtime and resource-use analysis conducted for the thesis.

### Usage

1. Prepare the environment for the selected model according to the documentation provided by its authors.
2. Copy the appropriate `reconstruct_cli.py` file and, optionally, `reconstruct_cli_batch.py` into the working directory of that model.
3. Make sure the active environment has the required packages, model weights, CUDA, and input data available.
4. Run the script in the target model environment.

Example single-test run:

```bash
python reconstruct_cli.py \
  --test-name test_name \
  --src path/to/images
```

The batch scripts use test lists defined directly in the `reconstruct_cli_batch.py` files. They assume that the required data is available in the parent `dataset` directory, for example under paths such as `../dataset/custom/...`, `../dataset/eth3d/...`, and `../dataset/nerf_llff_data/...`. Before running them, provide this directory structure or adjust the `TESTS` variable to match the local dataset layout.

### Results

The scripts create an `outputs/<test_name>/` directory containing:

- `reconstruction.glb` - the generated 3D reconstruction,
- `metrics.json` - runtime metrics, including stage timings, RAM usage, and CUDA memory usage.

The following Excel files contain the collected performance measurements:

- `performance_data.xlsx` - combined performance comparison spreadsheet,
- `models/vggt/vggt_results.xlsx` - VGGT results,
- `models/mapany/mapany_results.xlsx` - MapAnything results,
- `models/mast3r/mast3r_results.xlsx` - MASt3R results.

## Geometric Evaluation and Point-to-Point Distance Experiment

The three scripts in the repository root are MapAnything entrypoints used to prepare reconstructions for geometric evaluation and a focused physical-distance experiment. This experiment is not a complete metric analysis of the reconstruction. Its scope is limited to selecting two points in an input image, retrieving their corresponding predicted 3D points, calculating the distance between them, and comparing that result with a real-world reference measurement.

- [`reconstruct_cli_eval.py`](./reconstruct_cli_eval.py) runs image-only reconstruction and exports the visualization, a raw PLY point cloud, predicted camera parameters and geometry, an evaluation manifest, the selected-image list, and runtime metrics.
- [`reconstruct_cli_metric.py`](./reconstruct_cli_metric.py) reuses `reconstruct_cli_eval.py` and creates a self-contained package for the point-to-point distance experiment. It saves the exact processed images and their corresponding per-pixel 3D points so that the selected image coordinates can be mapped directly to the predicted geometry.
- [`reconstruct_cli_metric_batch.py`](./reconstruct_cli_metric_batch.py) runs a configured list of point-to-point distance experiments in separate processes. Edit its `TESTS` list before use; images are discovered under `../dataset/<dataset>/<scene>/images/<scenario>/` and outputs are named `outputs/metric_<scene>_<scenario>/`.

Copy all three files into the root of a MapAnything working directory. The distance-experiment script imports the evaluation script, and the batch script expects the two reconstruction entrypoints to remain beside it.

Example evaluation export:

```bash
python reconstruct_cli_eval.py \
  --test-name evaluation_name \
  --src path/to/images \
  --runs 3
```

Example reconstruction prepared for point-to-point distance measurement:

```bash
python reconstruct_cli_metric.py \
  --test-name metric_room_2_scen_1 \
  --scene-name room_2 \
  --subset-label scen_1 \
  --src path/to/images \
  --runs 1
```

The distance-experiment reconstruction creates `outputs/<test_name>/` with:

- `reconstruction.glb` and `reconstruction_raw.ply` - visualization and raw point cloud,
- `metric_predictions.npz` - camera, depth, confidence, mask, processed-image, and per-pixel 3D data,
- `processed_images/` and `selected_images.txt` - the exact image grid used by the predictions and the source-image list,
- `metric_experiment_manifest.json` - experiment metadata and array descriptions,
- `metrics.json` - runtime and resource-use metrics.

To validate the configured batch without starting reconstruction, run `python reconstruct_cli_metric_batch.py --dry-run`; remove `--dry-run` to execute it.

The [`metric_experiment_data.xlsx`](./metric_experiment_data.xlsx) spreadsheet contains the collected point-to-point distance results for five custom scenes and the one-, two-, four-, and eight-image scenarios. Each predicted distance is confronted with a corresponding real-world measurement. The workbook includes overall and per-scene summaries, a MAPE comparison, predicted and reference distances, absolute, signed, and percentage errors, RMSE, confidence and depth values, metric scaling factors, selected pixel coordinates, and the underlying raw measurement records. These results characterize this specific distance-measurement task and should not be interpreted as a comprehensive metric evaluation of the reconstructed scenes.

## Qualitative Visual Analysis

An interactive [app used for the visual analysis](https://one-or-few-shot-3d-reconstruction.vercel.app/) is available online.

The [`quality_data.xlsx`](./quality_data.xlsx) spreadsheet contains details of the visual analysis of the reconstruction results conducted as part of the thesis.

The [`quality_extras_data.xlsx`](./quality_extras_data.xlsx) spreadsheet contains a complementary comparison of selected scene subsets reconstructed from additional variants of the input image sets.

## External Materials

- Experiment results and custom dataset: [Google Drive](https://drive.google.com/drive/folders/1aigl2HDYsFkaPHm-5PpmWyUgdT2znGYT?usp=sharing)
