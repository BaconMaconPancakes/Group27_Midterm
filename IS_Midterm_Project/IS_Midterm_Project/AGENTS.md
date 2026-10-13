# Project instructions

- Keep one project root: this directory. Place Python source, dataset, artifacts, and dependency specification in `Codes/`.
- Preserve `Codes/data/campus_events.csv` as an independent, portable CSV shared by Python and KNIME. Training and prediction must never regenerate or modify it.
- Use Logistic Regression as the primary classifier and Decision Tree as the comparison classifier. Fit preprocessing within each training fold or split only.
- Keep Streamlit integration in `Codes/src/predict.py`; UI code should not duplicate validation or encoding.
- Label the dataset as synthetic and report only measured evaluation results.
