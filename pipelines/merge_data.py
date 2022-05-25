import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

from pipelines_config import *

# Merges all docword.dict.{name}.pkl files into one frame splits and saves
# to pickle


def main():
    """
    Reads all dataframes from docword.dict.{name}.pkl files
    and saves result into data_all.pkl file
    """
    logger.info("STARTED MERGING DATA PIPELINE")
    logger.info(f"Started reading input dataframes from {docword_dicts_paths}")
    dfs = list(map(pd.read_pickle, docword_dicts_paths))
    logger.info("Started merging dataframes")
    concatenated = reduce(lambda df1, df2: pd.concat([df1, df2], axis="index"), dfs)
    logger.info(f"Head of output dataframe")
    print(concatenated.head(5))
<<<<<<< HEAD
    logger.info(f"Saving output to {concatenated_data_path}")
    concatenated.to_pickle(concatenated_data_path)
=======
    df_train, df_test = train_test_split(concatenated, test_size=test_size)
    logger.info(f"Started saving train data to {train_data_path}")
    df_train.reset_index(drop=True).to_pickle(train_data_path)
    logger.info(f"Started saving test data to {test_data_path}")
    df_test.reset_index(drop=True).to_pickle(test_data_path)
>>>>>>> origin/main


if __name__ == "__main__":
    main()
