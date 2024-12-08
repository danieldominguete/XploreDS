"""
Xplore DS :: Drift Data Analysis
"""

from evidently.calculations.stattests import psi_stat_test
import pandas as pd


def calculate_psi_score(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_name: str,
    feature_type: str = "num",  # "num", "cat", "text", "datetime", "date", "id", "unknown"
    threshold: float = 0.2,  # PSI threshold for drift detection
):
    """
        Calculate PSI score using EvidendlyAI's psi_stat_test function with all available parameters.

        Args:
            reference_data: Reference/baseline distribution
            current_data: Current/production distribution
            feature_type: Type of feature being analyzed:
                - "num": Numerical
                - "cat": Categorical
                - "text": Text data
                - "datetime": Datetime
                - "date": Date
                - "id": Identifier
                - "unknown": Unknown type
            threshold: PSI threshold for drift detection (default: 0.2)
            num_bins: Number of bins for numerical data (default: 10)
            bin_strategy: Binning strategy for numerical data:
                - "quantile": Equal-frequency binning
                - "uniform": Equal-width binning
                - "auto": Automatic binning
            min_samples: Minimum number of samples required (default: 50)
            aggregation: Aggregation method for datetime/date features (default: "mean")
            datetime_aggregation_window: Time window for datetime aggregation (default: "D")
            confidence: Confidence level for statistical test (default: 0.95)
            cat_top_k: Number of top categories to consider for categorical features

        Returns:
            StatTest object with drift results

        Note:
    #         PSI < 0.1: No significant distribution change
    #         0.1 <= PSI < 0.2: Moderate distribution change
    #         PSI >= 0.2: Significant distribution change
    """

    # Extract columns
    reference_data = reference_data[column_name]
    current_data = current_data[column_name]

    result = psi_stat_test(
        reference_data=reference_data,
        current_data=current_data,
        feature_type=feature_type,
        threshold=threshold,
    )

    return result.drift_score, bool(result.drifted)
