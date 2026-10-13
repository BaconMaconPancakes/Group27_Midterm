"""Run with `python -m src.train` to train and save both model pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_validation import FEATURE_COLUMNS, TARGET_CLASSES, TARGET_COLUMN, load_and_validate_dataset
from .evaluate import evaluate_predictions, save_evaluation
from .preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "campus_events.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessing", build_preprocessor()), ("classifier", estimator)])


def train() -> pd.DataFrame:
    frame, validation = load_and_validate_dataset(DATASET)
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]

    #added for knime continuity
    train_frame = pd.read_csv(ROOT / "data" / "knime_train.csv")
    test_frame = pd.read_csv(ROOT / "data" / "knime_test.csv")

    train_x = train_frame.loc[:, list(FEATURE_COLUMNS)]
    train_y = train_frame[TARGET_COLUMN]

    test_x = test_frame.loc[:, list(FEATURE_COLUMNS)]
    test_y = test_frame[TARGET_COLUMN]
    
    models = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows: list[dict] = []
    result_rows: list[dict] = []
    matrices: dict[str, pd.DataFrame] = {}
    fitted: dict[str, Pipeline] = {}
    for name, make_estimator in models.items():
        # cross_validate clones the complete pipeline; each fold fits its own preprocessing.
        scores = cross_validate(
            build_pipeline(make_estimator()), train_x, train_y, cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, error_score="raise"
        )
        for fold, (accuracy, f1) in enumerate(zip(scores["test_accuracy"], scores["test_f1_macro"]), start=1):
            cv_rows.append({"model": name, "fold": fold, "accuracy": float(accuracy), "f1_macro": float(f1)})

        pipeline = build_pipeline(make_estimator())
        pipeline.fit(train_x, train_y)
        row, matrix = evaluate_predictions(name, test_y, pipeline.predict(test_x), list(TARGET_CLASSES))
        result_rows.append(row)
        matrices[name] = matrix
        fitted[name] = pipeline

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / "cross_validation.csv", index=False)
    save_evaluation(result_rows, matrices, RESULTS_DIR)
    metadata = {
        "primary_model": "logistic_regression",
        "comparison_model": "decision_tree",
        "feature_names": list(FEATURE_COLUMNS),
        "target_classes": list(TARGET_CLASSES),
        "dataset_size": validation["records"],
        "class_distribution": validation["class_distribution"],
        "available_options": {
            "venue": sorted(frame["venue"].unique().tolist()),
            "equipment_needs": sorted(frame["equipment_needs"].unique().tolist()),
        },
        "metrics": {row["model"]: {key: row[key] for key in ("accuracy", "precision", "recall", "f1_score")} for row in result_rows},
        "metric_averaging": "macro",
        "test_size": len(test_x),
        "training_size": len(train_x),
        "random_state": RANDOM_STATE,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODELS_DIR / f"{name}.joblib")
    return pd.DataFrame(result_rows)


if __name__ == "__main__":
    print(train().to_string(index=False))
