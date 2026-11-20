
import csv
from pathlib import Path

# Locate the vocabulary dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "signcadence_nsl_provisional_vocabulary.csv"
)

def display_vocabulary():

    if not CSV_FILE.exists():
        print("Vocabulary file not found!")
        return

    with open(CSV_FILE, "r", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        print("\nSIGNCADENCE - NSL VOCABULARY")
        print("=" * 45)

        for item in reader:
            print(
                f"{item['concept_id']} | "
                f"{item['english_concept']} | "
                f"{item['community_verification_status']}"
            )

        print("=" * 45)


if __name__ == "__main__":
    display_vocabulary()
