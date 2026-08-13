# Supplementary Materials for the Master's Thesis

This repository contains supplementary materials for a master's thesis on one-shot and few-shot 3D reconstruction. It brings together the experiment scripts, dataset documentation, collected measurements, and result spreadsheets used to evaluate and compare the selected reconstruction methods.

The repository does not provide complete runtime environments, model implementations, or model weights. The included scripts should be copied into the appropriate model repositories or working directories and executed in environments that meet the requirements specified by the model authors.

## Contents

The materials cover three analyzed 3D reconstruction methods: VGGT, MapAnything, and MASt3R. For each method, the repository provides scripts for individual reconstructions and batch experiments on predefined scenes.

It also includes method-specific result spreadsheets, a combined performance comparison spreadsheet, and [documentation of the photo subsets](./DATASET_PHOTO_SETS.md) used in every experiment scenario. The document lists the exact input photographs for each dataset, scene, and scenario, covering all 38 scenes and the one-, two-, four-, and eight-image configurations used in the experiments.

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

## Qualitative Visual Analysis

The [`quality_data.xlsx`](./quality_data.xlsx) spreadsheet contains details of the visual analysis of the reconstruction results conducted as part of the thesis.

The [`quality_extras_data.xlsx`](./quality_extras_data.xlsx) spreadsheet contains a complementary comparison of selected scene subsets reconstructed from additional variants of the input image sets.

## External Materials

- Experiment results and custom dataset: [Google Drive](https://drive.google.com/drive/folders/1aigl2HDYsFkaPHm-5PpmWyUgdT2znGYT?usp=sharing)
