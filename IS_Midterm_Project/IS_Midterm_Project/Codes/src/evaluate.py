"""Held-out multiclass evaluation and portable result files."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(model_name: str, actual, predicted, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    # Macro averaging gives each of the four classes equal weight.
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    row = {
        "model": model_name,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }
    matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels), index=labels, columns=labels)
    matrix.index.name = "actual\\predicted"
    return row, matrix


def save_evaluation(rows: list[dict], matrices: dict[str, pd.DataFrame], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1_score"]).to_csv(
        results_dir / "model_comparison.csv", index=False
    )
    for model_name, matrix in matrices.items():
        matrix.to_csv(results_dir / f"confusion_matrix_{model_name}.csv")
