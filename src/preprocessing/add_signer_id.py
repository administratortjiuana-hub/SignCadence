
import csv
import shutil
from pathlib import Path
from datetime import datetime

dataset_file = Path("dataset/landmarks/signcadence_landmarks.csv")

if not dataset_file.exists():
    raise FileNotFoundError(f"Dataset not found: {dataset_file}")

# Read the dataset
with dataset_file.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames:
        raise ValueError("The CSV file has no column headers.")

    fieldnames = list(reader.fieldnames)
    rows = list(reader)

if "capture_id" not in fieldnames:
    raise ValueError("The dataset must contain a capture_id column.")

# Ask for the signer ID
signer_id = input("Enter the signer ID (e.g. TEST_SIGNER_001): ").strip()

if not signer_id:
    raise ValueError("Signer ID cannot be empty.")

# Group rows by capture
captures = {}

for row in rows:
    capture_id = row["capture_id"].strip()

    if not capture_id:
        raise ValueError("A row has an empty capture_id.")

    captures.setdefault(capture_id, []).append(row)

# Check existing signer IDs
if "signer_id" not in fieldnames:
    fieldnames.append("signer_id")

for capture_id, capture_rows in captures.items():
    existing_ids = {
        row.get("signer_id", "").strip()
        for row in capture_rows
        if row.get("signer_id", "").strip()
    }

    if len(existing_ids) > 1:
        raise ValueError(
            f"Capture {capture_id} has conflicting signer IDs: {existing_ids}"
        )

    if existing_ids:
        existing_id = next(iter(existing_ids))

        for row in capture_rows:
            if not row.get("signer_id", "").strip():
                row["signer_id"] = existing_id
    else:
        for row in capture_rows:
            row["signer_id"] = signer_id

# Check whether changes are needed
if all(row.get("signer_id", "").strip() for row in rows):
    print("All rows already have signer IDs.")
    print("No changes were made.")
    raise SystemExit

# Confirm before writing
print(f"Total captures: {len(captures)}")
print(f"Total landmark rows: {len(rows)}")
print(f"New signer ID for unassigned captures: {signer_id}")

confirmation = input(
    "Confirm that all unassigned captures belong to this signer? Type YES: "
).strip()

if confirmation != "YES":
    print("Cancelled. No changes were made.")
    raise SystemExit

# Create a backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file = dataset_file.with_name(
    f"{dataset_file.stem}_backup_{timestamp}{dataset_file.suffix}"
)

shutil.copy2(dataset_file, backup_file)

# Save updated dataset
with dataset_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Signer IDs updated successfully.")
print(f"Total captures: {len(captures)}")
print(f"Total landmark rows: {len(rows)}")
print(f"Backup created: {backup_file}")
