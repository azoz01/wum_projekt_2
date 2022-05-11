import pandas as pd
from tqdm.auto import tqdm
from functools import partial
from pandas.core.groupby import DataFrameGroupBy

from pipelines_config import *

tqdm.pandas()


def flatten_group(group: DataFrameGroupBy, id_to_word_mapping: dict) -> dict:
    """Converts group of entries corresponding to one document
    to bag of words dictionary.

    Args:
        group (DataFrameGroupBy)
        id_to_word_mapping (dict): mapping from ids is group to words

    Returns:
        dict: Bag of words dictionary constructed from group
    """
    ids = group["word_id"]
    words = ids.map(id_to_word_mapping)
    counts = group["count"].values
    return dict(zip(words, counts))


def get_id_to_word_mapping_from_csv(vocab_path: str) -> dict:
    """Reads csv_path with vocabulary, reindexes and returns
    as dictionary in form id: word

    Args:
        vocab_path (str)

    Returns:
        dict
    """
    logger.info(f"Started reading dataframe from {vocab_path}")
    df = pd.read_csv(vocab_path, header=None, sep=" ")
    df.index = df.index + 1
    logger.info(f"Started conversion from dataframe to dictionary")
    return dict(zip(df.index, df[0]))


def convert_raw_data_to_bow_dictionaries_df(
    docword_path: str, dataset_name: str, id_to_word_mapping: dict
) -> pd.DataFrame:
    """Reads docword_path file and converts into bag of word dictionary

    Args:
        docword_path (str)
        dataset_name (str)
        id_to_word_mapping (dict): mapping from ids in frame to words

    Returns:
        pd.DataFrame
    """
    logger.info(f"Started reading input csv from {docword_path}")
    df = pd.read_csv(docword_path)

    logger.info("Started grouping by doc_id and converting raw data to dictionary")
    df = (
        df.groupby("doc_id")
        .progress_apply(partial(flatten_group, id_to_word_mapping=id_to_word_mapping))
        .reset_index(drop=False)
    )
    logger.info("Started renaming columns and adding column with dataset label")
    df.columns = ["doc_id", "bow_dict"]
    df["label"] = dataset_name
    return df


def main():
    """
    Converts raw data into bag of words dictionaries format and saves to
    docword.dict.{name}.pkl files.
    """
    logger.info("STARTED CONVERSION TO DICTIONARIES PIPELINE")
    for name, docword_path, vocab_path, output_path in zip(
        dataset_names, docwords_sampled_paths, vocab_paths, docword_dicts_paths
    ):
        logger.info(f"Started processing {name}")
        logger.info(f"Started getting mapping from ids to words")
        id_to_word_mapping = get_id_to_word_mapping_from_csv(vocab_path)
        logger.info("Started conversion from raw file to bow dictionary")
        df = convert_raw_data_to_bow_dictionaries_df(
            docword_path, name, id_to_word_mapping
        )
        logger.info("Head of output")
        print(df.head(5))
        logger.info(f"Started saving output df to {output_path}")
        df.to_pickle(output_path)


if __name__ == "__main__":
    main()
