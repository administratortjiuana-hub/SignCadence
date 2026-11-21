
import csv
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT / "dataset" / "processed" / "handpose_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "signcadence_model.joblib"

FEATURE_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

RANDOM_STATE = 42


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATASET_FILE}"
        )

    with DATASET_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = (
            ["capture_id", "sign_label", "signer_id"]
            + FEATURE_COLUMNS
        )

        missing = [
            column for column in required
            if column not in (reader.fieldnames or [])
        ]

        if missing:
            raise ValueError(
                f"Missing dataset columns: {missing}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("The processed dataset is empty.")

    return rows


def train_model():
    print("\nSIGNCADENCE AI MODEL TRAINING")
    print("=" * 45)

    rows = load_dataset()

    features = []
    labels = []
    signers = []
    capture_ids = []

    for row_number, row in enumerate(rows, start=2):
        label = row["sign_label"].strip()
        signer = row["signer_id"].strip()
        capture = row["capture_id"].strip()

        if not label or not signer or not capture:
            raise ValueError(
                f"Empty label, signer or capture ID at CSV row "
                f"{row_number}."
            )

        try:
            values = [
                float(row[column])
                for column in FEATURE_COLUMNS
            ]
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid landmark values at CSV row {row_number}."
            )

        if not all(
            value == value
            and abs(value) != float("inf")
            for value in values
        ):
            raise ValueError(
                f"Non-finite landmark values at CSV row {row_number}."
            )

        features.append(values)
        labels.append(label)
        signers.append(signer)
        capture_ids.append(capture)

    if len(set(capture_ids)) != len(capture_ids):
        raise ValueError("Duplicate capture IDs detected.")

    label_counts = Counter(labels)
    signers_per_label = defaultdict(set)

    for label, signer in zip(labels, signers):
        signers_per_label[label].add(signer)

    print("Processed captures:", len(rows))
    print("Different signs:", len(label_counts))
    print("Different signers:", len(set(signers)))

    print("\nCAPTURES PER SIGN")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

    if len(label_counts) < 2:
        print("\nTRAINING STOPPED.")
        print("At least two different sign labels are required.")
        return

    for label, count in label_counts.items():
        if count < 4:
            print(f"\nTRAINING STOPPED: {label} has only {count} captures.")
            print("Collect at least 4 captures per sign to proceed.")
            return

        if len(signers_per_label[label]) < 2:
            print(f"\nTRAINING STOPPED: {label} has fewer than 2 signers.")
            print("Each sign must have examples from at least 2 signers.")
            return

    # Keep each signer's captures together during testing.
    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    all_labels = set(label_counts)
    selected_split = None

    for train_indices, test_indices in splitter.split(
        features, labels, groups=signers
    ):
        train_labels = {labels[i] for i in train_indices}
        test_labels = {labels[i] for i in test_indices}

        if train_labels == all_labels and test_labels == all_labels:
            selected_split = (train_indices, test_indices)
            break

    if selected_split is None:
        print("\nTRAINING STOPPED.")
        print("Could not create a signer-independent split")
        print("containing every sign in both training and testing.")
        print("Collect more examples from additional signers.")
        return

    train_indices, test_indices = selected_split

    X_train = [features[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print("Training captures:", len(X_train))
    print("Testing captures:", len(X_test))
    print(f"Test accuracy: {accuracy:.2%}")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(all_labels),
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "labels": sorted(all_labels),
            "sklearn_version": sklearn.__version__,
        },
        MODEL_FILE,
    )

    print("Model saved to:", MODEL_FILE)
    print("\nTraining completed.")
    print("=" * 45)


if __name__ == "__main__":
    try:
        train_model()
    except (FileNotFoundError, ValueError) as error:
        print("\nERROR:", error)
