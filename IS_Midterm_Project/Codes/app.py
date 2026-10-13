"""Streamlit interface for the campus-event classifier.

Run from the Codes directory with:
    streamlit run app.py
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.data_validation import FEATURE_COLUMNS
from src.predict import (
    InputValidationError,
    get_available_options,
    get_model_info,
    predict_event_proba,
    validate_event_input,
)


CLASS_EXPLANATIONS = {
    "Academic": "Likely focused on learning, instruction, research, or an academic presentation.",
    "Sports": "Likely focused on physical activity, athletics, games, or sports training.",
    "Cultural": "Likely focused on arts, culture, performances, or community activities.",
    "Organizational": "Likely focused on meetings, coordination, planning, or organizational work.",
}


def _show_errors(errors: dict[str, str]) -> None:
    for field, message in errors.items():
        st.error(f"{field}: {message}")


def _predict_and_display(participants, duration, venue, equipment_needs) -> None:
    checked = validate_event_input(participants, duration, venue, equipment_needs)
    if not checked["valid"]:
        _show_errors(checked["errors"])
        return

    try:
        result = predict_event_proba(**checked["values"])
    except InputValidationError as exc:
        _show_errors(exc.errors)
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first with: python -m src.train")
        return

    label = result["event_type"]
    probabilities = result["probabilities"]
    st.success(f"Predicted event type: {label}")
    st.write(CLASS_EXPLANATIONS[label])

    st.subheader("Class probabilities")
    for class_name in ("Academic", "Sports", "Cultural", "Organizational"):
        probability = float(probabilities.get(class_name, 0.0))
        st.write(f"**{class_name}**: {probability:.1%}")
        st.caption(CLASS_EXPLANATIONS[class_name])
        st.progress(probability)


def _parse_number(value: str, field: str):
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        st.error(f"{field}: enter a number.")
        return None
    if not math.isfinite(number):
        st.error(f"{field}: enter a finite number.")
        return None
    return int(number) if number.is_integer() else number


st.set_page_config(page_title="Campus Event Classifier", page_icon="🎓", layout="centered")
st.title("Campus Event Type Classifier")
st.caption("Classifies a proposed event using the saved Logistic Regression model.")
st.warning("This demonstration uses a small, project-created synthetic dataset; its prediction is not validated operational performance.")

try:
    options = get_available_options()
    model_info = get_model_info()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("From the Codes folder, run: python -m src.train")
    st.stop()

with st.expander("Model information"):
    st.write(f"Primary model: {model_info['primary_model']}")
    st.write(f"Training examples: {model_info['dataset_size']}")
    st.write(f"Features: {', '.join(model_info['feature_names'])}")

tab_form, tab_upload = st.tabs(["Enter an event", "Upload one event"])

with tab_form:
    with st.form("event_form"):
        participants_text = st.text_input("Participants", placeholder="e.g. 120")
        duration_text = st.text_input("Duration (hours)", placeholder="e.g. 3")
        venue_choice = st.selectbox("Venue", ["Select a venue", *options["venue"]])
        equipment_choice = st.selectbox(
            "Primary equipment", ["Select equipment", *options["equipment_needs"]]
        )
        submitted = st.form_submit_button("Classify event")

    if submitted:
        participants = _parse_number(participants_text, "participants")
        duration = _parse_number(duration_text, "duration")
        venue = None if venue_choice == "Select a venue" else venue_choice
        equipment_needs = None if equipment_choice == "Select equipment" else equipment_choice
        _predict_and_display(participants, duration, venue, equipment_needs)

with tab_upload:
    st.write("Upload a CSV containing exactly one event row with these input columns:")
    st.code(", ".join(FEATURE_COLUMNS))
    uploaded = st.file_uploader("Choose an event CSV", type=["csv"])

    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            missing = [column for column in FEATURE_COLUMNS if column not in uploaded_frame.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif len(uploaded_frame) != 1:
                st.error("The uploaded CSV must contain exactly one event row.")
            else:
                row = uploaded_frame.iloc[0]
                st.dataframe(uploaded_frame, use_container_width=True, hide_index=True)
                if st.button("Classify uploaded event"):
                    _predict_and_display(
                        row["participants"],
                        row["duration"],
                        row["venue"],
                        row["equipment_needs"],
                    )
        except Exception as exc:  # Streamlit should show a friendly file error.
            st.error(f"Could not read the uploaded CSV: {exc}")
