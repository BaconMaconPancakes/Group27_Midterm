"""Small, reusable prediction API for Streamlit or other callers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .data_validation import FEATURE_COLUMNS, validate_event_values

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"


class InputValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("Invalid event input: " + "; ".join(f"{key}: {value}" for key, value in errors.items()))


@lru_cache(maxsize=1)
def _bundle() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model is missing: {MODEL_PATH}. Run python -m src.train first.")
    bundle = joblib.load(MODEL_PATH)

    # scikit-learn 1.9 models may omit this attribute, but older runtimes
    # still access it while calculating predict_proba().
    classifier = bundle["pipeline"].steps[-1][1]
    if classifier.__class__.__name__ == "LogisticRegression" and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    return bundle


def load_model():
    """Load and cache the persisted Logistic Regression pipeline."""
    return _bundle()["pipeline"]


def get_available_options() -> dict[str, list[str]]:
    """Get categorical choices from training metadata in the saved model."""
    options = _bundle()["metadata"]["available_options"]
    return {key: list(values) for key, values in options.items()}


def validate_event_input(participants, duration, venue, equipment_needs) -> dict:
    """Return {valid, errors, values}; values are normalized when valid."""
    options = get_available_options()
    return validate_event_values(
        participants, duration, venue, equipment_needs,
        venues=tuple(options["venue"]), equipment=tuple(options["equipment_needs"]),
    )


def _event_frame(participants, duration, venue, equipment_needs) -> pd.DataFrame:
    result = validate_event_input(participants, duration, venue, equipment_needs)
    if not result["valid"]:
        raise InputValidationError(result["errors"])
    return pd.DataFrame([result["values"]], columns=list(FEATURE_COLUMNS))


def predict_event(participants, duration, venue, equipment_needs) -> dict[str, str]:
    """Predict one event with the saved pipeline; never retrain here."""
    frame = _event_frame(participants, duration, venue, equipment_needs)
    return {"event_type": str(load_model().predict(frame)[0])}


def predict_event_proba(participants, duration, venue, equipment_needs) -> dict:
    """Return the predicted class and measured model probabilities for one event."""
    frame = _event_frame(participants, duration, venue, equipment_needs)
    model = load_model()
    values = model.predict_proba(frame)[0]
    probabilities = {str(label): float(value) for label, value in zip(model.classes_, values)}
    return {"event_type": str(model.classes_[values.argmax()]), "probabilities": probabilities}


def get_model_info() -> dict:
    """Return model identity, features, classes, dataset size, and measured metrics."""
    metadata = _bundle()["metadata"]
    return {key: metadata[key] for key in (
        "primary_model", "comparison_model", "feature_names", "target_classes",
        "dataset_size", "class_distribution", "metrics", "metric_averaging",
    )}
