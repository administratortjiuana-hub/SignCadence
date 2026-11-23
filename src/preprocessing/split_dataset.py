
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


def split_dataset(input_file=INPUT_FILE):

    input_file = Path(input_file)

    print("\nSIGNCADENCE DATASET SPLITTING")
    print("=" * 45)

    if not input_file.exists():
        print("ERROR: Processed dataset not found.")
        return

    with open(input_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("ERROR: Dataset is empty.")
        return

    required_columns = {"signer_id", "sign_label", "capture_id"}

    if not required_columns.issubset(fieldnames):
        print("SPLITTING STOPPED.")
        print("Required columns are missing:",
              required_columns - set(fieldnames))
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

    print("STATUS: BASIC SIGNER CHECK PASSED")
    print("A class-aware split must still be designed.")
    print("No files have been created or modified.")
    print("=" * 45)


if __name__ == "__main__":
    split_dataset()
