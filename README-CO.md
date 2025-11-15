# SB-Net-raw: Code Ocean Quick Start

This capsule runs the dataset makers and creates encoded CSVs for downstream experiments.

## Prerequisites

- Python 3.10+
- Install deps: `pip install -r requirements.txt`
- Place sample data as described in `Datasets/_SAMPLE_DATA_README.md`

## Run

- Validate expected input folders exist (no execution):
  - `python scripts/run_all.py --check`
- Run all dataset makers and the combiner:
  - `python scripts/run_all.py --datasets all`
- Run a specific dataset maker only:
  - `python scripts/run_all.py --datasets ctu13`
  - `python scripts/run_all.py --datasets ncc`
  - `python scripts/run_all.py --datasets ncc2`

## Outputs

- Each dataset maker writes to `<dataset>/final_dataset/`.
- The combiner writes `Datasets/combined_train.csv`.

## Notes

- The makers rely on relative paths; the runner sets correct working directories per dataset.
- If you only provide a subset of sample data, use `--datasets` to run what you have.
