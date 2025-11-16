# Sample Data Layout and Full Data Instructions

This repository includes a minimal set of sample `.binetflow` files so you can verify the end‑to‑end pipeline quickly. The same folder layout is required for running with full (non‑sampled) datasets.

## Expected Folder Structure

Place files under `Datasets/` with the following layout:

- CTU-13
  - `1/capture20110810.binetflow`
  - `2/capture20110811.binetflow`
  - `5/capture20110815-2.binetflow`
  - `9/capture20110817.binetflow`
  - `13/capture20110815-3.binetflow`
- NCC
  - `scenario_dataset_1/dataset_result.binetflow`
  - `scenario_dataset_2/dataset_result.binetflow`
  - `scenario_dataset_5/dataset_result.binetflow`
  - `scenario_dataset_9/dataset_result.binetflow`
  - `scenario_dataset_13/dataset_result.binetflow`
- NCC-2
  - `sensor1/sensor1.binetflow`
  - `sensor2/sensor2.binetflow`
  - `sensor3/sensor3.binetflow`

The sample files shipped in this repo follow this structure, but may be limited subsets intended only for quick checks.

## Using Full (Non‑Sampled) Data

- Replace each sample file with the full corresponding `.binetflow` file in the same relative path shown above.
- Do not change file names or directory names; the scripts look for those exact names.
- For Code Ocean, upload the full data into the `Datasets/` directories or connect a data asset and map it to the same paths.

## Outputs

Each dataset maker writes encoded CSVs into its `final_dataset/` folder, for example:

- `Datasets/CTU-13/final_dataset/train.csv`
- `Datasets/NCC/final_dataset/train.csv`
- `Datasets/NCC-2/final_dataset/train.csv`

If all three exist, the combiner generates `Datasets/combined_train.csv`.

## Tips

- Run a layout check first: `python scripts/run_all.py --check`
- You can run just one dataset (e.g., only CTU-13) by passing `--datasets ctu13`.
- Combining requires all three train CSVs to exist; otherwise the combiner is skipped automatically.
