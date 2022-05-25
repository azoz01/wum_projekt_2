from pipelines_config import *
import pickle as pkl
import pandas as pd
import numpy as np


def main():
    logger.info("STARTED CONCATENATING DATA WITH METADATA")
    logger.info(
        f"Started reading train and test data from {(train_dim_red_data_path, test_dim_red_data_path)}"
    )
    with open(train_dim_red_data_path, "rb") as f:
        X_train = pkl.load(f)
    with open(test_dim_red_data_path, "rb") as f:
        X_test = pkl.load(f)

    logger.info(
        f"Started reading metadata from {(train_data_meta_path, test_data_meta_path)}"
    )
    train_meta = pd.read_pickle(train_data_meta_path)
    test_meta = pd.read_pickle(test_data_meta_path)

    logger.info("Started concatenating data with metadata")
    X_train = np.concatenate([X_train, train_meta.values], axis=1)
    X_test = np.concatenate([X_test, test_meta.values], axis=1)

    logger.info(f"Started saving train data to {train_data_enriched_path}")
    with open(train_data_enriched_path, "wb") as f:
        pkl.dump(X_train, f)
    logger.info(f"Started saving test data to {test_data_enriched_path}")
    with open(test_data_enriched_path, "wb") as f:
        pkl.dump(X_test, f)


if __name__ == "__main__":
    main()
