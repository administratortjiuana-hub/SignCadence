
import csv
import tempfile
import unittest
from pathlib import Path

from src.preprocessing.prepare_dataset import prepare_dataset


class TestPrepareDataset(unittest.TestCase):

    def create_dataset(self, file_path, landmark_count=21):
        columns = [
            "capture_id",
            "sign_label",
            "signer_id",
            "hand_index",
            "landmark_index",
            "x",
            "y",
            "z",
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            for index in range(landmark_count):
                writer.writerow({
                    "capture_id": "TEST_CAPTURE",
                    "sign_label": "HELLO",
                    "signer_id": "TEST_SIGNER_001",
                    "hand_index": 0,
                    "landmark_index": index,
                    "x": 0.1,
                    "y": 0.2,
                    "z": 0.3,
                })

    def test_valid_capture_produces_63_features(self):
        with tempfile.TemporaryDirectory() as folder:
            input_file = Path(folder) / "input.csv"
            output_file = Path(folder) / "output.csv"

            self.create_dataset(input_file)

            result = prepare_dataset(input_file, output_file)

            self.assertEqual(result["valid"], 1)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["features"], 63)

            with open(output_file, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sign_label"], "HELLO")
            self.assertEqual(rows[0]["signer_id"], "TEST_SIGNER_001")
            self.assertEqual(len(rows[0]), 66)

    def test_incomplete_capture_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            input_file = Path(folder) / "input.csv"
            output_file = Path(folder) / "output.csv"

            self.create_dataset(input_file, landmark_count=20)

            result = prepare_dataset(input_file, output_file)

            self.assertEqual(result["valid"], 0)
            self.assertEqual(result["skipped"], 1)

            with open(output_file, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
