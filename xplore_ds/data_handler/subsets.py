"""
Xplore DS :: Dataset Tools Package
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_subsets(
    data: pd,
    proportion_test_samples: float = 0.1,
    random_state: int = None,
    shuffle: bool = False,
    log: object = None,
) -> pd:

    # Splitting the dataset into the Training set and Test set
    # random_state: integer number maintain reproducible output
    # shuffle: mix samples before split
    data_train, data_test = train_test_split(
        data,
        test_size=proportion_test_samples,
        shuffle=shuffle,
        random_state=random_state,
    )

    log.info("Train and test subsets with shuffle = " + str(shuffle))
    log.info("Train samples: " + str(data_train.shape))
    log.info("Test samples: " + str(data_test.shape))

    return data_train, data_test
