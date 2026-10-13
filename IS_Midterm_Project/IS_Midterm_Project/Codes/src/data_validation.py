"""Shared schema and validation for the independent campus-event CSV."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ("participants", "duration", "venue", "equipment_needs")
TARGET_COLUMN = "event_type"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
TARGET_CLASSES = ("Academic", "Sports", "Cultural", "Organizational")
VALID_VENUES = (
    "Auditorium", "Lecture Hall", "Conference Room",
    "Multipurpose Hall", "Gymnasium", "Open Field",
)
VALID_EQUIPMENT = (
    "Projector", "Computers", "Microphones",
    "Sound System", "Sports Gear", "No Equipment",
)
IDENTIFIER_PATTERN = re.compile(r"(^id$|(^|_)id$|identifier|student|person|name|code$)", re.I)
LEAKAGE_PATTERN = re.compile(r"target|label|category|event.?type|class", re.I)


class DataValidationError(ValueError):
    """Dataset schema or content is unsuitable for model training."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Dataset validation failed:\n- " + "\n- ".join(errors))


def validate_event_values(
    participants: object,
    duration: object,
    venue: object,
    equipment_needs: object,
    *,
    venues: tuple[str, ...] = VALID_VENUES,
    equipment: tuple[str, ...] = VALID_EQUIPMENT,
) -> dict:
    """Return UI-friendly validation and normalized inputs for a single event."""
    errors: dict[str, str] = {}
    clean: dict[str, int | float | str] = {}

    if isinstance(participants, bool) or not isinstance(participants, (int, float)):
        errors["participants"] = "Enter a positive whole number of participants."
    elif not math.isfinite(participants) or participants <= 0 or not float(participants).is_integer():
        errors["participants"] = "Participants must be a positive whole number."
    else:
        clean["participants"] = int(participants)

    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        errors["duration"] = "Enter a positive duration in hours."
    elif not math.isfinite(duration) or duration <= 0:
        errors["duration"] = "Duration must be a positive finite number of hours."
    else:
        clean["duration"] = float(duration)

    if not isinstance(venue, str) or venue.strip() not in venues:
        errors["venue"] = f"Choose a venue from: {', '.join(venues)}."
    else:
        clean["venue"] = venue.strip()

    if not isinstance(equipment_needs, str) or equipment_needs.strip() not in equipment:
        errors["equipment_needs"] = f"Choose equipment from: {', '.join(equipment)}."
    else:
        clean["equipment_needs"] = equipment_needs.strip()

    return {"valid": not errors, "errors": errors, "values": clean if not errors else None}


def validate_dataset(frame: pd.DataFrame) -> dict:
    """Validate all rows, then return a summary or raise with every detected issue."""
    errors: list[str] = []
    columns = list(frame.columns)
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(REQUIRED_COLUMNS))
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if unexpected:
        errors.append(f"Unexpected columns: {unexpected}")
        identifiers = [name for name in unexpected if IDENTIFIER_PATTERN.search(name)]
        leakage = [name for name in unexpected if LEAKAGE_PATTERN.search(name)]
        if identifiers:
            errors.append(f"Possible identifier columns: {identifiers}")
        if leakage:
            errors.append(f"Possible target-leakage columns: {leakage}")
    if frame.empty:
        errors.append("Dataset is empty")
    elif not 30 <= len(frame) <= 100:
        errors.append(f"Dataset should contain 30 to 100 records; found {len(frame)}")
    if frame.isna().any().any():
        counts = frame.isna().sum()
        errors.append(f"Missing values: {counts[counts > 0].to_dict()}")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        errors.append(f"Duplicate rows: {duplicates}")

    if not missing:
        for name in ("participants", "duration"):
            if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
                errors.append(f"{name} must have a numeric data type")
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.isna().any() or not values.map(lambda value: math.isfinite(value)).all():
                errors.append(f"{name} contains invalid or non-finite numerical values")
            elif (values <= 0).any():
                errors.append(f"{name} must be greater than zero in every row")
            elif name == "participants" and not (values % 1 == 0).all():
                errors.append("participants must be whole numbers")

        for name, valid in (("venue", VALID_VENUES), ("equipment_needs", VALID_EQUIPMENT)):
            if not pd.api.types.is_string_dtype(frame[name]):
                errors.append(f"{name} must have a text data type")
            invalid = sorted({str(value) for value in frame[name].dropna() if value not in valid})
            if invalid:
                errors.append(f"Invalid {name} values: {invalid}")

        if not pd.api.types.is_string_dtype(frame[TARGET_COLUMN]):
            errors.append("event_type must have a text data type")
        invalid_targets = sorted({str(value) for value in frame[TARGET_COLUMN].dropna() if value not in TARGET_CLASSES})
        if invalid_targets:
            errors.append(f"Invalid event_type values: {invalid_targets}")
        distribution = {name: int((frame[TARGET_COLUMN] == name).sum()) for name in TARGET_CLASSES}
        if any(count == 0 for count in distribution.values()):
            errors.append(f"All four target classes are required: {distribution}")
        elif min(distribution.values()) < 2:
            errors.append(f"Each target class needs at least two records for stratification: {distribution}")
        elif max(distribution.values()) > 2 * min(distribution.values()):
            errors.append(f"Target classes are too imbalanced for this small dataset: {distribution}")
    else:
        distribution = {}

    if errors:
        raise DataValidationError(errors)
    return {
        "records": int(len(frame)),
        "input_features": list(FEATURE_COLUMNS),
        "class_distribution": distribution,
        "duplicate_rows": duplicates,
        "missing_values": 0,
        "unexpected_columns": [],
        "identifier_columns": [],
        "possible_target_leakage_columns": [],
    }


def load_and_validate_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a CSV from disk without changing it and validate its contents."""
    frame = pd.read_csv(path)
    return frame, validate_dataset(frame)
