# SB-Net-raw: Code Ocean Guide

This capsule prepares encoded CSV datasets from CTU-13, NCC, and NCC-2 `.binetflow` sources and (optionally) combines them for downstream experiments.

## What’s Included

- `co_run.sh`: Entry-point script (used by Code Ocean Run) that installs requirements and invokes the orchestrator.
- `scripts/run_all.py`: Orchestrates dataset checks, runs dataset makers, and conditionally runs the combiner.
- `Datasets/*/data_maker_*.py`: Dataset makers for CTU-13, NCC, NCC-2.
- `Datasets/train_combiner.py`: Combines all three encoded training CSVs when available.
- Sample data structure and guidance: `Datasets/_SAMPLE_DATA_README.md`.

## Data Layout (Sample and Full)

Use the exact folder/file names below under `Datasets/` (sample files follow this layout and full data must replace them in-place):

- CTU-13: `1/capture20110810.binetflow`, `2/capture20110811.binetflow`, `5/capture20110815-2.binetflow`, `9/capture20110817.binetflow`, `13/capture20110815-3.binetflow`
- NCC: `scenario_dataset_1/dataset_result.binetflow`, `scenario_dataset_2/dataset_result.binetflow`, `scenario_dataset_5/dataset_result.binetflow`, `scenario_dataset_9/dataset_result.binetflow`, `scenario_dataset_13/dataset_result.binetflow`
- NCC-2: `sensor1/sensor1.binetflow`, `sensor2/sensor2.binetflow`, `sensor3/sensor3.binetflow`

See `Datasets/_SAMPLE_DATA_README.md` for details and tips on swapping to full (non-sampled) data.

## Code Ocean Run

Set the capsule’s Run to execute:

```
bash co_run.sh
```

Optionally choose which datasets to process via an environment variable:

- `DATASETS_ARG=all` (default)
- `DATASETS_ARG=ctu13`
- `DATASETS_ARG=ncc`
- `DATASETS_ARG=ncc2`

In Code Ocean, configure `DATASETS_ARG` under Environment Variables if you want to run a subset. The script will:

1. `pip install -r requirements.txt`
2. Run `python scripts/run_all.py --datasets "$DATASETS_ARG"`

## Local Run (outside Code Ocean)

- Python: 3.10+
- Install: `pip install -r requirements.txt`

- Validate folder layout only (no processing):
  - `python scripts/run_all.py --check`
- Run all available datasets:
  - `python scripts/run_all.py --datasets all`
- Run a specific dataset only:
  - `python scripts/run_all.py --datasets ctu13`
  - `python scripts/run_all.py --datasets ncc`
  - `python scripts/run_all.py --datasets ncc2`

## Outputs

- Per-dataset encoded CSVs written to:
  - `Datasets/CTU-13/final_dataset/train.csv` and `test_*.csv`
  - `Datasets/NCC/final_dataset/train.csv` and `test_*.csv`
  - `Datasets/NCC-2/final_dataset/train.csv` and `test_*.csv`
- Combined training CSV (only when all three train CSVs exist):
  - `Datasets/combined_train.csv`

The orchestrator now safely skips combining if any of the three train CSVs are missing.

## Notes & Recommendations

- Reproducibility: dataset splits use `random_state=42`.
- Memory/Time: full (non-sampled) `.binetflow` files can be large; allocate sufficient RAM/CPU in Code Ocean.
- Relative Paths: runners set working directories so data makers can resolve relative file paths correctly.

## Troubleshooting

- "missing folders" during `--check`: ensure your files are under the exact paths listed above.
- Combiner skipped: one or more of the following are missing:
  `Datasets/CTU-13/final_dataset/train.csv`, `Datasets/NCC/final_dataset/train.csv`, `Datasets/NCC-2/final_dataset/train.csv`.
- Pandas dtype warnings: benign during coercion; scripts explicitly coerce and fill numeric types.
