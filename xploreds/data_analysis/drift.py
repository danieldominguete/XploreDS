"""
Xplore DS :: Drift Data Analysis
"""

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.options.data_drift import DataDriftOptions
import pandas as pd


def calculate_psi_score(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Calculate Population Stability Index (PSI) score between two dataframe columns with custom binning

    Args:
        expected_df: Reference/expected dataframe
        actual_df: Current/actual dataframe
        column_name: Column name to compare
        num_bins: Number of bins to use (default: 10)
        bin_type: Type of binning strategy. Options:
                 - 'auto': Automatically determine bin edges
                 - 'uniform': Uniform bin sizes
                 - 'quantile': Equal number of samples in each bin

    Returns:
        float: PSI drift score

    Note:
        PSI < 0.1: No significant distribution change
        0.1 <= PSI < 0.2: Moderate distribution change
        PSI >= 0.2: Significant distribution change
    """

    # Create drift report with PSI test and binning options
    drift_report = Report(
        metrics=[
            DataDriftPreset(stattest="psi", stattest_threshold="0.3"),
        ]
    )

    # Run analysis
    drift_report.run(
        reference_data=expected_df[[column_name]], current_data=actual_df[[column_name]]
    )

    # Extract PSI score from report
    report_dict = drift_report.as_dict()
    psi_score = report_dict["metrics"][1]["result"]["drift_by_columns"][column_name][
        "drift_score"
    ]

    return psi_score
