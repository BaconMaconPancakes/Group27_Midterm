# Campus Event Type Classification

Group 27's midterm backend classifies proposed campus events as **Academic**, **Sports**, **Cultural**, or **Organizational**. The runnable Python project is in [`Codes/`](Codes/). The dataset is **project-created synthetic example data**, not measured university statistics. Its small size and constructed origin limit conclusions about real campus events.

## Layout and shared data

| Path | Purpose |
|---|---|
| `Codes/data/campus_events.csv` | Independent CSV used by Python and KNIME |
| `Codes/data/data_dictionary.md` | Column types, valid values, roles, and relevance |
| `Codes/src/data_validation.py` | Dataset and single-event validation |
| `Codes/src/preprocessing.py` | Unfitted sklearn preprocessing factory |
| `Codes/src/train.py` | Stratified training, cross-validation, and persistence |
| `Codes/src/evaluate.py` | Held-out metrics and confusion matrices |
| `Codes/src/predict.py` | Streamlit-facing prediction API |
| `Codes/models/` | Trained sklearn pipelines produced by training |
| `Codes/results/` | Measured evaluation CSVs produced by training |

The CSV contains **72 rows**, four inputs, and one target. `participants` is expected attendance (positive whole number); `duration` is hours (positive number); `venue` is the planned campus space; `equipment_needs` is the primary equipment requirement. See the [data dictionary](Codes/data/data_dictionary.md) for all valid categories. Each target class has 18 examples. Rows have no personal information or identifiers. Several venues, equipment types, participant counts, and durations overlap across classes, so no single input maps directly to a label.

KNIME can independently load this exact CSV with a CSV Reader. Set `participants` to integer, `duration` to number, and the remaining columns to strings. The Python backend reads the file without rewriting it.

## Setup and training

From `Codes/`, create a Python environment and install `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Training first validates required columns, data types, missing values, duplicate rows, numeric ranges, valid categories and classes, target distribution, unexpected columns, and possible identifier or target-leakage columns. Invalid data raises `DataValidationError` with specific messages. The CSV is never generated or changed by training or prediction.

A reproducible 75%/25% stratified split uses random state 42. Five-fold stratified cross-validation runs on the training partition only. Each estimator has an sklearn `Pipeline` containing a `ColumnTransformer`: numeric inputs are standardized and categorical inputs are one-hot encoded. All preprocessing is fitted within each training fold or on the training partition; the held-out test partition stays unseen until evaluation. The primary estimator is `LogisticRegression`; the comparison is `DecisionTreeClassifier` with limited depth and leaf size. Both use the same four features and preprocessing.

The held-out comparison in `Codes/results/model_comparison.csv` reports accuracy and **macro-averaged** precision, recall, and F1. Macro averaging gives each class equal weight; zero-division scores are set to zero. `Codes/results/confusion_matrix_logistic_regression.csv` and `Codes/results/confusion_matrix_decision_tree.csv` use actual labels as rows and predicted labels as columns. `Codes/results/cross_validation.csv` records each training-fold accuracy and macro F1. Results are calculated by the code, never entered by hand. Small holdout and synthetic data make differences between algorithms unstable; neither model is universally superior.

The current run measured these scores on 18 held-out rows:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.389 | 0.488 | 0.400 | 0.425 |
| Decision Tree | 0.444 | 0.521 | 0.438 | 0.390 |

These results are weak and should be presented as a demonstration of the workflow, not as validated operational performance. The four logistical inputs overlap heavily, especially for Academic and Organizational events. A larger, real, independently labelled dataset would be needed before deployment.

Training writes complete preprocessing-plus-classifier pipelines to `Codes/models/logistic_regression.joblib` and `Codes/models/decision_tree.joblib`. Prediction uses the saved primary pipeline and does not retrain.

## Backend API and Streamlit integration

Run Streamlit from `Codes/` so `src` can be imported. The UI only collects inputs and displays messages; validation, category choices, encoding, model loading, and prediction stay in the backend.

```python
import streamlit as st
from src.predict import (
    InputValidationError, get_available_options, get_model_info,
    predict_event, predict_event_proba, validate_event_input,
)

options = get_available_options()
info = get_model_info()
st.caption(f"Primary model: {info['primary_model']} | Training examples: {info['dataset_size']}")
participants = st.number_input("Participants", min_value=1, value=120, step=1)
duration = st.number_input("Duration (hours)", min_value=0.5, value=3.0, step=0.5)
venue = st.selectbox("Venue", options["venue"])
equipment_needs = st.selectbox("Primary equipment", options["equipment_needs"])

if st.button("Classify"):
    checked = validate_event_input(participants, duration, venue, equipment_needs)
    if not checked["valid"]:
        for field, message in checked["errors"].items():
            st.error(f"{field}: {message}")
    else:
        try:
            label = predict_event(**checked["values"])
            scored = predict_event_proba(**checked["values"])
            st.success(label["event_type"])
            st.write(scored["probabilities"])
        except InputValidationError as exc:
            for field, message in exc.errors.items():
                st.error(f"{field}: {message}")
```

`load_model()` returns the cached trained primary pipeline. `validate_event_input(...)` returns `{"valid": bool, "errors": dict, "values": dict | None}`. `predict_event(...)` returns `{"event_type": str}`; `predict_event_proba(...)` adds a `probabilities` mapping. Both raise `InputValidationError` for invalid input. `get_available_options()` returns categorical dropdown values stored with the model. `get_model_info()` returns model names, features, target classes, dataset size, class distribution, and actual held-out metrics. Train first so the saved artifact exists.

## Remaining team work

- **KNIME teammate:** Read `Codes/data/campus_events.csv` directly, build the independent visual workflow using Logistic Regression and Decision Tree, and document its own split, preprocessing, and measured results. Different implementations or splits can yield different results.
- **Streamlit teammate:** Build the interface using the API above, display validation messages and model output, and avoid fitting models or encoding categories in UI code.
