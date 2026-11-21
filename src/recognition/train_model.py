
import csv
from pathlib import Path
from collections import Counter

# SignCadence dataset validation

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "handpose_features.csv"
)

EXPECTED_FEATURES = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


def check_dataset():

    print("\nSIGNCADENCE MODEL READINESS CHECK")
    print("=" * 40)

    if not DATASET_FILE.exists():
        print("ERROR: Processed dataset not found.")
        return

    with open(DATASET_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = (
            ["capture_id", "sign_label"]
            + EXPECTED_FEATURES
        )

        if not all(
            column in reader.fieldnames
            for column in required_columns
        ):
            print("ERROR: Dataset columns are incomplete.")
            return

        rows = list(reader)

    if not rows:
        print("ERROR: Dataset is empty.")
        return

    labels = Counter(
        row["sign_label"].strip()
        for row in rows
    )

    capture_ids = [
        row["capture_id"]
        for row in rows
    ]

    print("Processed captures:", len(rows))
    print("Unique capture IDs:", len(set(capture_ids)))
    print("Sign labels:", len(labels))

    print("\nCAPTURES PER SIGN")

    for label, count in sorted(labels.items()):
        print(f"{label}: {count}")

    if len(set(capture_ids)) != len(capture_ids):
        print("\nWARNING: Duplicate capture IDs detected.")

    if "" in labels:
        print("\nWARNING: Some captures have empty labels.")

    if len(labels) < 2:
        print("\nMODEL NOT READY.")
        print("At least two different sign labels are")
        print("needed for a basic classification model.")
        print("Please collect and verify more sign concepts.")

    else:
        print("\nMultiple sign labels detected.")
        print("Further dataset quality checks are needed")
        print("before model training.")

    print("\nThis check does not train an AI model.")
    print("=" * 40)


if __name__ == "__main__":
    check_dataset()
