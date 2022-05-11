import pandas as pd
import numpy as np
import swifter

from funcy import select_keys
from pipelines_config import *
from sklearn.feature_extraction.text import TfidfVectorizer


def dict_to_string(bow_dict: dict) -> str:
    """Converts bow dictionary to string, where each word occurs
    as many times as count in dict

    Args:
        bow_dict (dict)

    Returns:
        str
    """
    items = [[item[0]] * item[1] for item in bow_dict.items()]
    return " ".join([str(item) for l in items for item in l if item])


def filter_too_rare(bow_dict: dict, too_rare_tokens: np.ndarray) -> dict:
    """From bow_dict removes tokens stored in too_rare_tokens

    Args:
        bow_dict (dict)
        too_rare_tokens (np.ndarray)

    Returns:
        dict
    """
    return select_keys(lambda key: key not in too_rare_tokens, bow_dict)


def main():
    """
    From train and test dataframes deletes all tokens which
    occur in less than doc_freq_min_threshold documents.
    Token list is evaluated based only on train data. However,
    both frames are transformed
    """
    logger.info("STARTED FILTERING TOO RARE TOKENS PIPELINE")
    logger.info(
        f"Started Reading train and test data from {(train_data_path, test_data_path)}"
    )
    df_train = pd.read_pickle(train_data_path)
    df_test = pd.read_pickle(test_data_path)

    logger.info("Started conversion from bow dictionaries to strings")
    df_train["string"] = df_train["bow_dict"].apply(dict_to_string)
    df_test["string"] = df_test["bow_dict"].apply(dict_to_string)

    logger.info("Started fitting TF-IDF vectorizer")
    tfidf = TfidfVectorizer(use_idf=False, norm=None)
    logger.info("Started searching for too rare tokens")
    transformed = tfidf.fit_transform(df_train["string"])

    inv_vocab = {item[1]: item[0] for item in tfidf.vocabulary_.items()}

    too_rare_mask = np.array((transformed > 0).sum(axis=0) <= doc_freq_min_threshold)[0]
    too_rare_indices = np.where(too_rare_mask)
    too_rare_tokens = np.vectorize(inv_vocab.get)(too_rare_indices)
    logger.info("Started filtering too rare tokens")
    df_train["bow_dict"] = df_train["bow_dict"].swifter.apply(
        filter_too_rare, too_rare_tokens=too_rare_tokens
    )
    df_test["bow_dict"] = df_test["bow_dict"].swifter.apply(
        filter_too_rare, too_rare_tokens=too_rare_tokens
    )

    logger.info(
        f"Started saving train and test data to {train_data_path, test_data_path}"
    )
    df_train = df_train.drop(columns=["string"])
    df_test = df_test.drop(columns=["string"])
    df_train.to_pickle(train_data_path)
    df_test.to_pickle(test_data_path)


if __name__ == "__main__":
    main()
