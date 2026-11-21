
import csv
import math
from pathlib import Path
from collections import defaultdict

# SignCadence dataset quality control

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "landmarks"
    / "signcadence_landmarks.csv"
)

EXPECTED_LANDMARKS = set(range(21))


def validate_dataset():

    print("\nSIGNCADENCE DATASET QUALITY CHECK")
    print("=" * 45)

    if not DATASET_FILE.exists():
        print("ERROR: Landmark dataset not found.")
        return

    captures = defaultdict(list)
    errors = []

    with open(DATASET_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = [
            "capture_id",
            "sign_label",
            "hand_index",
            "landmark_index",
            "x",
            "y",
            "z"
        ]

        if not reader.fieldnames or not all(
            column in reader.fieldnames
            for column in required_columns
        ):
            print("ERROR: Required CSV columns are missing.")
            return

        for row_number, row in enumerate(reader, start=2):
            capture_id = row["capture_id"].strip()

            if not capture_id:
                errors.append(
                    f"Row {row_number}: Missing capture ID."
                )
                continue

            captures[capture_id].append(row)

    if not captures:
        print("ERROR: No valid captures found.")
        return

    print("Total captures:", len(captures))

    for capture_id, rows in captures.items():

        labels = {
            row["sign_label"].strip()
            for row in rows
        }

        if "" in labels:
            errors.append(
                f"{capture_id}: Empty sign label."
            )

        if len(labels) > 1:
            errors.append(
                f"{capture_id}: Conflicting sign labels."
            )

        hand_indices = {
            row["hand_index"].strip()
            for row in rows
        }

        if len(hand_indices) != 1:
            errors.append(
                f"{capture_id}: Unexpected hand count."
            )

        landmark_indices = []

        for row in rows:
            try:
                landmark_index = int(row["landmark_index"])
                landmark_indices.append(landmark_index)

                if landmark_index not in EXPECTED_LANDMARKS:
                    errors.append(
                        f"{capture_id}: Invalid landmark index "
                        f"{landmark_index}."
                    )

                for axis in ("x", "y", "z"):
                    value = float(row[axis])

                    if not math.isfinite(value):
                        errors.append(
                            f"{capture_id}: Invalid {axis} coordinate."
                        )

            except (ValueError, TypeError, KeyError):
                errors.append(
                    f"{capture_id}: Invalid landmark data."
                )

        if (
            len(landmark_indices) != 21
            or set(landmark_indices) != EXPECTED_LANDMARKS
        ):
            errors.append(
                f"{capture_id}: Incomplete or duplicate landmarks."
            )

    print("Quality errors:", len(errors))

    if errors:
        print("\nISSUES FOUND:")

        for error in errors:
            print("-", error)

        print("\nDATASET NEEDS ATTENTION.")

    else:
        print("\nAll captures passed the current checks.")
        print("DATASET STRUCTURE LOOKS GOOD.")

    print("=" * 45)


if __name__ == "__main__":
    validate_dataset()
