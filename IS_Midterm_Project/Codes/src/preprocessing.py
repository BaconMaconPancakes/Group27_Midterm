"""Unfitted sklearn preprocessing shared by both classifiers."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ("participants", "duration")
CATEGORICAL_FEATURES = ("venue", "equipment_needs")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
