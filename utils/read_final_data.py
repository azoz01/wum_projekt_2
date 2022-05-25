from typing import Tuple
from pipelines import pipelines_config
import pickle as pkl
import numpy as np
from typing import Tuple


def read_train_test_data() -> Tuple[np.ndarray, np.ndarray]:
    with open(pipelines_config.train_data_enriched_path, "rb") as f:
        X_train = pkl.load(f)
    with open(pipelines_config.test_data_enriched_path, "rb") as f:
        X_test = pkl.load(f)

    return X_train, X_test
