"""
Xplore DS :: Cookbook Script Template

Standard pipeline template for scripts under ``cookbook/``.

Author: daniel.dominguet@gmail.com
"""

# %%
from pathlib import Path

from dotenv import load_dotenv

from xploreds.log.xds_log import create_logger
from xploreds.utils.xds_environment import resolve_project_root

# -----------------------------------------------------------------------------
# PARAMETERS CONFIGURATION
# -----------------------------------------------------------------------------
var_example = "example"

# -----------------------------------------------------------------------------
# SCRIPT EXECUTION
# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Getting project root and script path
script_path = Path(__file__).resolve()
project_root = resolve_project_root(script_path.parent)

# Loading environment variables
load_dotenv(project_root / ".env")

# Creating logger
log = create_logger(project_root, script_path)

# Initializing run
log.init_run()
log.log_environment_setup()
# -----------------------------------------------------------------------------
# Pipeline execution
# -----------------------------------------------------------------------------
try:

    log.title("Pipeline steps")
    # Pipeline steps
    log.info(f"Variable example: {var_example}")
except Exception as exc:
    log.error("Pipeline execution failed: %s", exc, exc_info=True)
    raise
finally:
    log.close_run()

# %%
