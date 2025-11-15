# Sample Data Layout

Expected folder layout for running the dataset makers. Place your .binetflow files in these directories relative to each dataset folder.

CTU-13 (place under `Datasets/CTU-13/`):

- `1/capture20110810.binetflow`
- `2/capture20110811.binetflow`
- `5/capture20110815-2.binetflow`
- `9/capture20110817.binetflow`
- `13/capture20110815-3.binetflow`

NCC (place under `Datasets/NCC/`):

- `scenario_dataset_1/dataset_result.binetflow`
- `scenario_dataset_2/dataset_result.binetflow`
- `scenario_dataset_5/dataset_result.binetflow`
- `scenario_dataset_9/dataset_result.binetflow`
- `scenario_dataset_13/dataset_result.binetflow`

NCC-2 (place under `Datasets/NCC-2/`):

- `sensor1/sensor1.binetflow`
- `sensor2/sensor2.binetflow`
- `sensor3/sensor3.binetflow`

Outputs:

- Each maker writes encoded CSVs into `<dataset>/final_dataset/` as `train.csv` and `test_<key>.csv`.
- The combiner `Datasets/train_combiner.py` produces `Datasets/combined_train.csv`.
