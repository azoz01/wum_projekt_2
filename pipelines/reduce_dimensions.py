from sklearn.decomposition import TruncatedSVD
from pipelines_config import *
import pandas as pd
import pickle as pkl


def main():
    """
    Projects train and test samples to dimensions specified in
    pipelines_config (now it's 3000 - explained var. ratio ~0.85)
    using truncated SVD algorithm
    """
    logger.info("STARTED DIMENSIONALITY REDUCTION PIPELINE")
    logger.info(f"Started reading train data from {train_data_tfidf_path}")
    X_train = pd.read_pickle(train_data_tfidf_path)
    logger.info(f"Started reading test data from {test_data_tfidf_path}")
    X_test = pd.read_pickle(test_data_tfidf_path)

    logger.info("Started fitting truncated SVD estimator")
    svd = TruncatedSVD(n_components=final_dimensions_count)
    svd.fit(X_train)

    logger.info("Started transforming train data")
    X_train = svd.transform(X_train)
    logger.info("Started transforming test data")
    X_test = svd.transform(X_test)

    logger.info(f"Started saving train data to {train_dim_red_data_path}")
    with open(train_dim_red_data_path, "wb") as f:
        pkl.dump(X_train, f)
    logger.info(f"Started saving test data to {test_dim_red_data_path}")
    with open(test_dim_red_data_path, "wb") as f:
        pkl.dump(X_test, f)


if __name__ == "__main__":
    main()
