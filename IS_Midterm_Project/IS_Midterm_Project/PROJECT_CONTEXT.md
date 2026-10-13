# Campus Event Type Classification

Group 27 midterm project. Proposed campus events are classified as Academic, Sports, Cultural, or Organizational from four inputs: participants, duration (hours), venue, and equipment_needs.

The runnable Python backend is in `Codes/`. Its independent CSV at `Codes/data/campus_events.csv` is also the input for the KNIME teammate. The Streamlit teammate should use `Codes/src/predict.py`; the UI is outside this backend scope.

The dataset is project-created synthetic example data, not observed university statistics. Model metrics measure performance on a small holdout of these examples and do not establish real-world accuracy.
