import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "Datasets"


def check_ctu13(layout_root: Path) -> List[str]:
    required = ["1", "2", "5", "9", "13"]
    missing = [d for d in required if not (layout_root / d).exists()]
    return missing


def check_ncc(layout_root: Path) -> List[str]:
    required = [
        "scenario_dataset_1",
        "scenario_dataset_2",
        "scenario_dataset_5",
        "scenario_dataset_9",
        "scenario_dataset_13",
    ]
    missing = [d for d in required if not (layout_root / d).exists()]
    return missing


def check_ncc2(layout_root: Path) -> List[str]:
    required = ["sensor1", "sensor2", "sensor3"]
    missing = [d for d in required if not (layout_root / d).exists()]
    return missing


def run_script(script: Path, cwd: Path) -> None:
    print(f"\n=== Running {script.name} in {cwd} ===")
    try:
        subprocess.run([sys.executable, str(script.name)], check=True, cwd=str(cwd))
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script {script} failed with exit code {e.returncode}.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run dataset makers and combiner")
    parser.add_argument(
        "--datasets",
        choices=["ctu13", "ncc", "ncc2", "all"],
        default="all",
        help="Which dataset makers to run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate expected input folder layout without running",
    )
    args = parser.parse_args()

    to_run = []
    if args.datasets in ("ctu13", "all"):
        ctu_dir = DATASETS_DIR / "CTU-13"
        missing = check_ctu13(ctu_dir)
        if missing:
            print(
                "CTU-13: missing folders: "
                + ", ".join(missing)
                + f" under {ctu_dir}"
            )
        else:
            to_run.append((ctu_dir / "data_maker_CTU_13.py", ctu_dir))

    if args.datasets in ("ncc", "all"):
        ncc_dir = DATASETS_DIR / "NCC"
        missing = check_ncc(ncc_dir)
        if missing:
            print(
                "NCC: missing folders: "
                + ", ".join(missing)
                + f" under {ncc_dir}"
            )
        else:
            to_run.append((ncc_dir / "data_maker_NCC.py", ncc_dir))

    if args.datasets in ("ncc2", "all"):
        ncc2_dir = DATASETS_DIR / "NCC-2"
        missing = check_ncc2(ncc2_dir)
        if missing:
            print(
                "NCC-2: missing folders: "
                + ", ".join(missing)
                + f" under {ncc2_dir}"
            )
        else:
            to_run.append((ncc2_dir / "data_maker_NCC_2.py", ncc2_dir))

    if args.check:
        print("\nCheck complete. See messages above for any missing folders.")
        return

    # Run available makers
    for script, cwd in to_run:
        run_script(script, cwd)

    # Try to combine only when all three encoded train files exist
    combiner = DATASETS_DIR / "train_combiner.py"
    expected_trains = [
        DATASETS_DIR / "CTU-13" / "final_dataset" / "train.csv",
        DATASETS_DIR / "NCC" / "final_dataset" / "train.csv",
        DATASETS_DIR / "NCC-2" / "final_dataset" / "train.csv",
    ]
    if combiner.exists():
        if all(p.exists() for p in expected_trains):
            print("\n=== Running train_combiner.py (all train.csv found) ===")
            subprocess.run([sys.executable, str(combiner)], check=True, cwd=str(DATASETS_DIR))
        else:
            missing = [str(p) for p in expected_trains if not p.exists()]
            print("\nSkipping train_combiner.py. Missing expected files:")
            for m in missing:
                print(f"  - {m}")
            print("Provide all three datasets (CTU-13, NCC, NCC-2) to combine.")
    else:
        print("Combiner not found, skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
