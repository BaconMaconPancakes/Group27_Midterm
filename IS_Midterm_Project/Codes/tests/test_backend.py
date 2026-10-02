"""Focused checks for the shared CSV contract and persisted prediction API."""

import unittest
import hashlib

from src.data_validation import DataValidationError, load_and_validate_dataset, validate_dataset
from src.predict import (
    InputValidationError, get_available_options, get_model_info, load_model,
    predict_event, predict_event_proba, validate_event_input,
)
from src.train import DATASET


class BackendTests(unittest.TestCase):
    def test_dataset_contract_and_leakage_rejection(self):
        frame, summary = load_and_validate_dataset(DATASET)
        self.assertEqual(summary["records"], 72)
        self.assertEqual(set(summary["class_distribution"].values()), {18})
        self.assertEqual(summary["duplicate_rows"], 0)
        altered = frame.copy()
        altered["category_hint"] = altered["event_type"]
        with self.assertRaises(DataValidationError) as caught:
            validate_dataset(altered)
        self.assertIn("target-leakage", str(caught.exception))

    def test_invalid_input_is_rejected_before_prediction(self):
        result = validate_event_input(0, -2, "Unknown Hall", "Unknown Gear")
        self.assertFalse(result["valid"])
        self.assertEqual(set(result["errors"]), {"participants", "duration", "venue", "equipment_needs"})
        with self.assertRaises(InputValidationError):
            predict_event(0, 2, "Auditorium", "Projector")

    def test_corrupt_dataset_rows_are_rejected(self):
        original, _ = load_and_validate_dataset(DATASET)
        changes = {
            "missing": lambda frame: frame.__setitem__("duration", frame["duration"].mask(frame.index == 0)),
            "nonpositive": lambda frame: frame.loc.__setitem__((0, "participants"), 0),
            "invalid_venue": lambda frame: frame.loc.__setitem__((0, "venue"), "Unknown Hall"),
            "invalid_target": lambda frame: frame.loc.__setitem__((0, "event_type"), "Other"),
            "duplicate": lambda frame: frame.loc.__setitem__(1, frame.loc[0]),
        }
        for label, change in changes.items():
            with self.subTest(issue=label):
                altered = original.copy()
                change(altered)
                with self.assertRaises(DataValidationError):
                    validate_dataset(altered)

    def test_saved_pipeline_and_four_representative_predictions(self):
        before = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        model = load_model()
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertEqual(get_model_info()["dataset_size"], 72)
        options = get_available_options()
        self.assertIn("Auditorium", options["venue"])
        examples = (
            (120, 3, "Auditorium", "Projector"),
            (150, 3, "Open Field", "Sports Gear"),
            (170, 4, "Auditorium", "Microphones"),
            (65, 2, "Conference Room", "Projector"),
        )
        for event in examples:
            with self.subTest(event=event):
                self.assertTrue(validate_event_input(*event)["valid"])
                predicted = predict_event(*event)["event_type"]
                scored = predict_event_proba(*event)
                self.assertEqual(predicted, scored["event_type"])
                self.assertEqual(set(scored["probabilities"]), set(get_model_info()["target_classes"]))
                self.assertAlmostEqual(sum(scored["probabilities"].values()), 1.0)
        self.assertEqual(hashlib.sha256(DATASET.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
