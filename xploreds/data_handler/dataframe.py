"""
Xplore DS :: Dataframe Tools Package
"""

import pandas as pd


def describe_dataframe(data: pd, log=None) -> None:

    if log is not None:
        log.title("Dataframe description")
        log.info(data.info(verbose=True, memory_usage=True, show_counts=True))

    # description = dataframe.describe()

    # for row in description.iterrows():
    #     logging.info(row[0])
    #     for id in row[1].index:
    #         logging.info("Var: " + id + " {a:.3f}".format(a=(row[1][id])))
