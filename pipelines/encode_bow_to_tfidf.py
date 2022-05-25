import sys

sys.path.append("models")

from pipelines_config import *
from encoding import TfidfFromDictionariesVectorizer
import pandas as pd
import pickle as pkl


def main():
    """
    Converts dataframe with bag-of-words dictionaries, to dataframe with
    tf-idf sparse vectors
    """
    logger.info("STRARTED ENCODING BOWS TO TF-IDF PIPELINE")
    logger.info(f"Started reading train data from {train_data_path}")
    train_data = pd.read_pickle(train_data_path)
    logger.info(f"Started reading test data from {test_data_path}")
    test_data = pd.read_pickle(test_data_path)

    tfidf = TfidfFromDictionariesVectorizer()
    logger.info("Started fitting TF-IDF")
    tfidf.fit(train_data)

    logger.info("Started tranfrorming train data")
    train_data = tfidf.transform(train_data)
    logger.info("Started tranfrorming test data")
    test_data = tfidf.transform(test_data)

    logger.info(f"Started saving train data to {train_data_tfidf_path}")
    with open(train_data_tfidf_path, "wb") as f:
        pkl.dump(train_data, f)
    logger.info(f"Started saving test data to {test_data_tfidf_path}")
    with open(test_data_tfidf_path, "wb") as f:
        pkl.dump(test_data, f)


if __name__ == "__main__":
    main()
