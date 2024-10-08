"""
Xplore DS :: Data visualization with Plotly
"""

import plotly.express as px
from pathlib import Path
import sys
import os

# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))

from xplore_ds.data_handler.file import create_folder


def deploy_chart_in_navigator(fig: object) -> None:
    fig.show()


def save_chart_file(fig: object, path: str = None) -> None:
    create_folder(os.path.dirname(path))
    fig.write_image(path)


def plot_scatter_2d(
    x, y, view_chart: bool = True, save_chart: bool = False, file_path_image: str = None
):

    fig = px.scatter(x=x, y=y)

    if save_chart:
        save_chart_file(fig, file_path_image)

    if view_chart:
        deploy_chart_in_navigator(fig)
