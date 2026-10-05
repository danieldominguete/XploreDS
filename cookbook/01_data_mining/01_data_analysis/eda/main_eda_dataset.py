"""
Xplore DS :: Data Analysis - Exploratory Data Analysis (EDA)

Author: daniel.dominguet@gmail.com
"""

from pathlib import Path

from dotenv import load_dotenv

from xploreds.log.xds_log import XploreDSLogging, create_logger
from xploreds.utils.xds_environment import resolve_project_root


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
    dataset_file_path = "data/raw/credit-g.parquet"

    try:
        # ================================
        # Pipeline steps
        # ================================
        log.title("Exploratory Data Analysis (EDA)")

        from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Relatório de Crédito", explorative=True)
profile.to_file("relatorio_eda.html")

    finally:
        # ================================
        # Closing run
        # ================================
        log.close_run()


if __name__ == "__main__":
    main()
