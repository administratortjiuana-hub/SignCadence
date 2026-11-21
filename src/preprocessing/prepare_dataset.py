
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import csv
from pathlib import Path
from collections import defaultdict

# -----------------------------------------
# SIGNCADENCE - DATASET PREPROCESSING
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "landmarks"
    / "signcadence_landmarks.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_FOLDER / "handpose_features.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Landmark dataset not found: {INPUT_FILE}"
    )

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Group landmark rows by capture ID
captures = defaultdict(list)

with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "capture_id",
        "sign_label",
        "signer_id",
        "hand_index",
        "landmark_index",
        "x",
        "y",
        "z"
    }

    available_columns = set(reader.fieldnames or [])

    if not required_columns.issubset(available_columns):
        missing = required_columns - available_columns
        raise ValueError(
            f"The input CSV is missing columns: {sorted(missing)}"
        )

    for row in reader:
        captures[row["capture_id"]].append(row)

# Prepare 63 numerical features:
# 21 landmarks multiplied by x, y and z.
feature_names = []

for landmark in range(21):
    feature_names.extend([
        f"x{landmark}",
        f"y{landmark}",
        f"z{landmark}"
    ])

output_rows = []
skipped = 0

for capture_id, rows in captures.items():

    # This prototype currently supports one hand per capture.
    hand_ids = {row["hand_index"] for row in rows}

    if len(hand_ids) != 1:
        print(f"Skipping {capture_id}: expected one hand.")
        skipped += 1
        continue

    # Check signer identity.
    signer_ids = {
        row["signer_id"].strip() for row in rows
    }

    if len(signer_ids) != 1 or "" in signer_ids:
        print(f"Skipping {capture_id}: invalid signer ID.")
        skipped += 1
        continue

    # Check sign label consistency.
    labels = {
        row["sign_label"].strip() for row in rows
    }

    if len(labels) != 1 or "" in labels:
        print(f"Skipping {capture_id}: inconsistent labels.")
        skipped += 1
        continue

    # Sort landmarks into their correct order.
    try:
        rows.sort(
            key=lambda row: int(row["landmark_index"])
        )
        indices = [
            int(row["landmark_index"]) for row in rows
        ]
    except (ValueError, TypeError):
        print(f"Skipping {capture_id}: invalid landmark index.")
        skipped += 1
        continue

    if indices != list(range(21)):
        print(f"Skipping {capture_id}: invalid landmark data.")
        skipped += 1
        continue

    # Extract numerical coordinates.
    try:
        features = []

        for row in rows:
            features.extend([
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ])

    except (ValueError, TypeError):
        print(f"Skipping {capture_id}: invalid coordinates.")
        skipped += 1
        continue

    output_rows.append({
        "capture_id": capture_id,
        "sign_label": labels.pop(),
        "signer_id": signer_ids.pop(),
        "features": features
    })

# Save processed dataset.
with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "capture_id",
        "sign_label",
        "signer_id",
        *feature_names
    ])

    for item in output_rows:
        writer.writerow([
            item["capture_id"],
            item["sign_label"],
            item["signer_id"],
            *item["features"]
        ])

print("\nSIGNCADENCE DATASET PREPARATION")
print("=" * 45)
print("Original captures:", len(captures))
print("Valid captures:", len(output_rows))
print("Skipped captures:", skipped)
print("Features per capture:", len(feature_names))
print("Distinct signers:", len({
    item["signer_id"] for item in output_rows
}))
print("Labels:", sorted({
    item["sign_label"] for item in output_rows
}))
print("Output file:", OUTPUT_FILE)
print("=" * 45)

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "handpose_features.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "splits"


def split_dataset():

    print("\nSIGNCADENCE DATASET SPLITTING")
    print("=" * 45)

    if not INPUT_FILE.exists():
        print("ERROR: Processed dataset not found.")
        return

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("ERROR: Dataset is empty.")
        return

    required_columns = {"signer_id", "sign_label", "capture_id"}

    if not required_columns.issubset(fieldnames):
        print("SPLITTING STOPPED.")
        print("Required columns are missing:", required_columns - set(fieldnames))
        return

    if any(not row["signer_id"].strip() for row in rows):
        print("SPLITTING STOPPED.")
        print("Some captures have missing signer IDs.")
        return

    if any(not row["sign_label"].strip() for row in rows):
        print("SPLITTING STOPPED.")
        print("Some captures have missing sign labels.")
        return

    signers = {
        row["signer_id"].strip()
        for row in rows
    }

    labels = Counter(
        row["sign_label"].strip()
        for row in rows
    )

    print("Total captures:", len(rows))
    print("Distinct signers:", len(signers))
    print("Signers:", sorted(signers))
    print("Labels:", dict(labels))

    print("\nSIGNER-INDEPENDENT SPLIT CHECK")
    print("-" * 45)

    if len(signers) < 3:
        print("STATUS: NOT READY")
        print("At least three distinct signers are needed")
        print("for basic train, validation and test groups.")
        print("Current signers:", len(signers))
        print("Additional signers needed:", 3 - len(signers))
        print("\nNo files have been created or modified.")
        return

    # More than three signers does not automatically guarantee
    # a valid split. Every label must be represented across
    # the signer groups selected for each partition.
    print("STATUS: BASIC SIGNER CHECK PASSED")
    print("A class-aware split must still be designed.")
    print("No files have been created or modified.")

    print("=" * 45)


if __name__ == "__main__":
    split_dataset()

import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "handpose_features.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "splits"


def split_dataset():

    print("\nSIGNCADENCE DATASET SPLITTING")
    print("=" * 45)

    if not INPUT_FILE.exists():
        print("ERROR: Processed dataset not found.")
        return

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("ERROR: Dataset is empty.")
        return

    required_columns = {"signer_id", "sign_label", "capture_id"}

    if not required_columns.issubset(fieldnames):
        print("SPLITTING STOPPED.")
        print("Required columns are missing:", required_columns - set(fieldnames))
        return

    if any(not row["signer_id"].strip() for row in rows):
        print("SPLITTING STOPPED.")
        print("Some captures have missing signer IDs.")
        return

    if any(not row["sign_label"].strip() for row in rows):
        print("SPLITTING STOPPED.")
        print("Some captures have missing sign labels.")
        return

    signers = {
        row["signer_id"].strip()
        for row in rows
    }

    labels = Counter(
        row["sign_label"].strip()
        for row in rows
    )

    print("Total captures:", len(rows))
    print("Distinct signers:", len(signers))
    print("Signers:", sorted(signers))
    print("Labels:", dict(labels))

    print("\nSIGNER-INDEPENDENT SPLIT CHECK")
    print("-" * 45)

    if len(signers) < 3:
        print("STATUS: NOT READY")
        print("At least three distinct signers are needed")
        print("for basic train, validation and test groups.")
        print("Current signers:", len(signers))
        print("Additional signers needed:", 3 - len(signers))
        print("\nNo files have been created or modified.")
        return

    # More than three signers does not automatically guarantee
    # a valid split. Every label must be represented across
    # the signer groups selected for each partition.
    print("STATUS: BASIC SIGNER CHECK PASSED")
    print("A class-aware split must still be designed.")
    print("No files have been created or modified.")

    print("=" * 45)


if __name__ == "__main__":
    split_dataset()
