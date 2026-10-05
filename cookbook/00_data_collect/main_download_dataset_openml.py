"""
Xplore DS :: OpenML Dataset Downloader

Download datasets from OpenML.

Author: daniel.dominguete@gmail.com
"""

from pathlib import Path

from dotenv import load_dotenv
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd

from xploreds.log.xds_log import XploreDSLogging, create_logger
from xploreds.utils.xds_environment import resolve_project_root
from xploreds.io.xds_file import save_dataframe_to_parquet


def main() -> None:
    """Execute the cookbook script pipeline."""
    script_path = Path(__file__).resolve()
    project_root = resolve_project_root(script_path.parent)

    # Loading environment variables
    load_dotenv(project_root / ".env")

    # Creating logger
    log = create_logger(project_root, script_path)

    # Initializing run
    log.init_run()
    log.log_environment_setup()

    # ================================
    # Parameters configuration
    # ================================

    # Site de datasets: https://www.openml.org/

    dataset_name = "credit-g"
    datetime_enrichment = True
    datetime_start = "2020-01-01"
    datetime_end = "2026-01-01"
    output_path = project_root / "data" / "raw" / dataset_name

    try:
        # ================================
        # Pipeline steps
        # ================================
        log.title("Pipeline steps")

        # Download dataset from OpenML
        log.info(f"Downloading dataset {dataset_name} from OpenML")
        dataset = fetch_openml(name=dataset_name, as_frame=True)

        # Create unique id column
        dataset.frame["id"] = dataset.frame.index

        # Create timestamp column for timing purposes
        if datetime_enrichment:
            dataset.frame["dt_reference"] = np.random.choice(
                pd.date_range(start=datetime_start, end=datetime_end)
            )

        # Data description
        log.info(f"Total samples: {dataset.frame.shape[0]}")
        log.info(f"Number of variables: {dataset.frame.shape[1]}")
        log.info(f"Variables list: {dataset.frame.columns.values.tolist()}")

        # Save dataset to parquet
        save_dataframe_to_parquet(dataset.frame, output_path, log=log, overwrite=True)
        log.info(
            f"Dataset {dataset_name} downloaded and saved to {output_path}.parquet"
        )

    finally:
        # ================================
        # Closing run
        # ================================
        log.close_run()


if __name__ == "__main__":
    main()
