# Auxiliary Scripts for the Master's Thesis

This repository is an appendix to a master's thesis on one-shot and few-shot 3D reconstruction. It contains auxiliary scripts developed for running selected models, collecting metrics, and comparing results.

The repository does not provide complete runtime environments for the models. The scripts should be treated as supporting material: they should be copied into the appropriate model repositories or working directories and then executed in environments that meet the requirements specified by the authors of those models.

## Contents

The repository contains helper scripts prepared for three analyzed 3D reconstruction methods: VGGT, MapAnything, and MASt3R. For each method, a script for running a single reconstruction and a batch script for running experiments on predefined scenes were prepared.

In addition to the scripts, the repository includes spreadsheets with experiment results and a combined comparison spreadsheet used during result analysis.

## Usage

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

## Results

The scripts create an `outputs/<test_name>/` directory containing:

- `reconstruction.glb` - the generated 3D reconstruction,
- `metrics.json` - runtime metrics, including stage timings, RAM usage, and CUDA memory usage.

The Excel files contain the collected experiment results:

- `data.xlsx` - combined comparison spreadsheet,
- `models/vggt/vggt_results.xlsx` - VGGT results,
- `models/mapany/mapany_results.xlsx` - MapAnything results,
- `models/mast3r/mast3r_results.xlsx` - MASt3R results.

## External Materials

- Experiment results: [Google Drive](https://drive.google.com/drive/folders/1aigl2HDYsFkaPHm-5PpmWyUgdT2znGYT?usp=sharing)
