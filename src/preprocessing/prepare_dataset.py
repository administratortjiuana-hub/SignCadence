
import csv
import math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "landmarks"
    / "signcadence_landmarks.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_FOLDER / "handpose_features.csv"

REQUIRED_COLUMNS = {
    "capture_id",
    "sign_label",
    "signer_id",
    "hand_index",
    "landmark_index",
    "x",
    "y",
    "z",
}


def prepare_dataset(input_file=INPUT_FILE, output_file=OUTPUT_FILE):

    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Landmark dataset not found: {input_file}"
        )

    captures = defaultdict(list)

    with open(input_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        available_columns = set(reader.fieldnames or [])

        if not REQUIRED_COLUMNS.issubset(available_columns):
            missing = REQUIRED_COLUMNS - available_columns
            raise ValueError(
                f"Input CSV is missing columns: {sorted(missing)}"
            )

        for row in reader:
            captures[row["capture_id"].strip()].append(row)

    feature_names = []

    for landmark in range(21):
        feature_names.extend([
            f"x{landmark}",
            f"y{landmark}",
            f"z{landmark}",
        ])

    output_rows = []
    skipped = 0

    for capture_id, rows in captures.items():

        if not capture_id:
            skipped += 1
            continue

        hand_ids = {row["hand_index"].strip() for row in rows}

        if len(hand_ids) != 1:
            skipped += 1
            continue

        signer_ids = {
            row["signer_id"].strip() for row in rows
        }

        if len(signer_ids) != 1 or "" in signer_ids:
            skipped += 1
            continue

        labels = {
            row["sign_label"].strip() for row in rows
        }

        if len(labels) != 1 or "" in labels:
            skipped += 1
            continue

        try:
            ordered_rows = sorted(
                rows,
                key=lambda row: int(row["landmark_index"])
            )

            indices = [
                int(row["landmark_index"])
                for row in ordered_rows
            ]

            if indices != list(range(21)):
                skipped += 1
                continue

            features = []

            for row in ordered_rows:
                coordinates = [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ]

                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError("Invalid coordinate")

                features.extend(coordinates)

        except (ValueError, TypeError, KeyError):
            skipped += 1
            continue

        output_rows.append({
            "capture_id": capture_id,
            "sign_label": labels.pop(),
            "signer_id": signer_ids.pop(),
            "features": features,
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "capture_id",
            "sign_label",
            "signer_id",
            *feature_names,
        ])

        for item in output_rows:
            writer.writerow([
                item["capture_id"],
                item["sign_label"],
                item["signer_id"],
                *item["features"],
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
    print("Output file:", output_file)
    print("=" * 45)

    return {
        "original": len(captures),
        "valid": len(output_rows),
        "skipped": skipped,
        "features": len(feature_names),
    }


if __name__ == "__main__":
    prepare_dataset()
