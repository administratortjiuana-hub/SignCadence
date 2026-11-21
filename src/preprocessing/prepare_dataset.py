
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

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Landmark dataset not found: {INPUT_FILE}"
    )

# Group landmark rows by capture ID
captures = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "capture_id",
        "sign_label",
        "hand_index",
        "landmark_index",
        "x",
        "y",
        "z"
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError("The input CSV is missing required columns.")

    for row in reader:
        captures[row["capture_id"]].append(row)

# Prepare machine-learning features
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

    # This first prototype supports one hand per capture.
    hand_ids = {row["hand_index"] for row in rows}

    if len(hand_ids) != 1:
        print(f"Skipping {capture_id}: expected one hand.")
        skipped += 1
        continue

    # Sort landmarks into their correct order
    rows.sort(key=lambda row: int(row["landmark_index"]))

    indices = [int(row["landmark_index"]) for row in rows]

    if indices != list(range(21)):
        print(f"Skipping {capture_id}: invalid landmark data.")
        skipped += 1
        continue

    labels = {row["sign_label"] for row in rows}

    if len(labels) != 1:
        print(f"Skipping {capture_id}: inconsistent labels.")
        skipped += 1
        continue

    # Extract 63 numerical features:
    # 21 landmarks multiplied by x, y and z.
    features = []

    for row in rows:
        features.extend([
            float(row["x"]),
            float(row["y"]),
            float(row["z"])
        ])

    output_rows.append({
        "capture_id": capture_id,
        "sign_label": labels.pop(),
        "features": features
    })

# Save processed dataset
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "capture_id",
        "sign_label",
        *feature_names
    ])

    for item in output_rows:
        writer.writerow([
            item["capture_id"],
            item["sign_label"],
            *item["features"]
        ])

print("\nSIGNCADENCE DATASET PREPARATION")
print("=" * 40)
print("Original captures:", len(captures))
print("Valid captures:", len(output_rows))
print("Skipped captures:", skipped)
print("Features per capture:", len(feature_names))
print("Labels:", sorted({
    item["sign_label"] for item in output_rows
}))
print("Output file:", OUTPUT_FILE)
print("=" * 40)
