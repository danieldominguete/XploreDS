"""
Xplore DS :: Drift Data Analysis
"""

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import pandas as pd


def get_drift_analysis(expected_data: pd, actual_data: pd, column_selected: str):

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(stattest="psi", stattest_threshold="0.3"),
        ]
    )
    data_drift_report.run(
        reference_data=expected_data[[column_selected]],
        current_data=actual_data[[column_selected]],
    )
    report = data_drift_report.as_dict()
    drift_detected = report["metrics"][1]["result"]["drift_by_columns"][
        column_selected
    ]["drift_detected"]

    drift_score = report["metrics"][1]["result"]["drift_by_columns"][column_selected][
        "drift_score"
    ]

    return drift_score
