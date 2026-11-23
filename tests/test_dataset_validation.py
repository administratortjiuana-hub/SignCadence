
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.preprocessing import validate_dataset


class TestDatasetValidation(unittest.TestCase):

    def create_valid_dataset(self, file_path):
        columns = [
            "capture_id",
            "sign_label",
            "hand_index",
            "landmark_index",
            "x",
            "y",
            "z",
        ]

        with open(file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            for index in range(21):
                writer.writerow({
                    "capture_id": "TEST_CAPTURE",
                    "sign_label": "HELLO",
                    "hand_index": 0,
                    "landmark_index": index,
                    "x": 0.1,
                    "y": 0.2,
                    "z": 0.3,
                })

    def get_output(self, dataset_file):
        with patch.object(
            validate_dataset, "DATASET_FILE", dataset_file
        ):
            with patch("builtins.print") as mock_print:
                validate_dataset.validate_dataset()

        return " ".join(
            " ".join(map(str, call.args))
            for call in mock_print.call_args_list
        )

    def modify_first_row(self, dataset_file, changes):
        with open(dataset_file, "r", newline="") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames
            rows = list(reader)

        rows[0].update(changes)

        with open(dataset_file, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_capture_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "test.csv"
            self.create_valid_dataset(dataset_file)

            output = self.get_output(dataset_file)

            self.assertIn("Quality errors: 0", output)
            self.assertIn(
                "DATASET STRUCTURE LOOKS GOOD", output
            )

    def test_incomplete_capture_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "test.csv"
            self.create_valid_dataset(dataset_file)

            with open(dataset_file, "r", newline="") as file:
                rows = list(csv.DictReader(file))

            with open(dataset_file, "w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=rows[0].keys()
                )
                writer.writeheader()
                writer.writerows(rows[:-1])

            output = self.get_output(dataset_file)

            self.assertIn(
                "Incomplete or duplicate landmarks", output
            )

    def test_invalid_landmark_index_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "test.csv"
            self.create_valid_dataset(dataset_file)

            self.modify_first_row(
                dataset_file, {"landmark_index": "25"}
            )

            output = self.get_output(dataset_file)

            self.assertIn("Invalid landmark index 25", output)

    def test_non_numeric_coordinate_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "test.csv"
            self.create_valid_dataset(dataset_file)

            self.modify_first_row(
                dataset_file, {"x": "not_a_number"}
            )

            output = self.get_output(dataset_file)

            self.assertIn("Invalid landmark data.", output)

    def test_infinite_coordinate_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "test.csv"
            self.create_valid_dataset(dataset_file)

            self.modify_first_row(
                dataset_file, {"x": "inf"}
            )

            output = self.get_output(dataset_file)

            self.assertIn("Invalid x coordinate.", output)


if __name__ == "__main__":
    unittest.main()
