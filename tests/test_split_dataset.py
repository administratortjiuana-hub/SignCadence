
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.preprocessing.split_dataset import split_dataset


class TestSplitDataset(unittest.TestCase):

    def create_dataset(self, file_path, signer_ids):
        columns = ["capture_id", "sign_label", "signer_id"]

        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            for index, signer_id in enumerate(signer_ids):
                writer.writerow({
                    "capture_id": f"CAPTURE_{index}",
                    "sign_label": "HELLO",
                    "signer_id": signer_id,
                })

    def test_fewer_than_three_signers_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "input.csv"
            self.create_dataset(
                dataset_file,
                ["SIGNER_001", "SIGNER_001", "SIGNER_002"]
            )

            with patch("builtins.print") as mock_print:
                split_dataset(dataset_file)

            printed_text = " ".join(
                " ".join(map(str, call.args))
                for call in mock_print.call_args_list
            )

            self.assertIn("STATUS: NOT READY", printed_text)
            self.assertIn("Additional signers needed: 1", printed_text)

    def test_three_signers_pass_basic_check(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "input.csv"
            self.create_dataset(
                dataset_file,
                ["SIGNER_001", "SIGNER_002", "SIGNER_003"]
            )

            with patch("builtins.print") as mock_print:
                split_dataset(dataset_file)

            printed_text = " ".join(
                " ".join(map(str, call.args))
                for call in mock_print.call_args_list
            )

            self.assertIn(
                "STATUS: BASIC SIGNER CHECK PASSED",
                printed_text
            )

    def test_missing_signer_id_stops_check(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_file = Path(folder) / "input.csv"
            self.create_dataset(
                dataset_file,
                ["SIGNER_001", "", "SIGNER_003"]
            )

            with patch("builtins.print") as mock_print:
                split_dataset(dataset_file)

            printed_text = " ".join(
                " ".join(map(str, call.args))
                for call in mock_print.call_args_list
            )

            self.assertIn(
                "Some captures have missing signer IDs.",
                printed_text
            )


if __name__ == "__main__":
    unittest.main()
