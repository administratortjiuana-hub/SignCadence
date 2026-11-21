
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

    with open(INPUT_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("ERROR: Dataset is empty.")
        return

    if "signer_id" not in fieldnames:
        print("SPLITTING STOPPED.")
        print("The dataset has no signer_id column.")
        print("We must identify each consenting signer")
        print("before creating signer-independent splits.")
        return

    if "sign_label" not in fieldnames:
        print("ERROR: sign_label column is missing.")
        return

    signers = {
        row["signer_id"].strip()
        for row in rows
        if row["signer_id"].strip()
    }

    if len(signers) < 3:
        print("SPLITTING STOPPED.")
        print("At least three distinct signers are needed")
        print("for basic train, validation and test groups.")
        return

    if any(not row["signer_id"].strip() for row in rows):
        print("SPLITTING STOPPED.")
        print("Some captures have missing signer IDs.")
        return

    print("Distinct signers:", len(signers))
    print("Captures:", len(rows))
    print("Labels:", dict(Counter(
        row["sign_label"] for row in rows
    )))

    print("\nThe dataset has passed basic signer checks.")
    print("A carefully designed, class-aware split")
    print("must be planned before files are created.")

    print("\nNo files have been split or modified.")
    print("=" * 45)


if __name__ == "__main__":
    split_dataset()
