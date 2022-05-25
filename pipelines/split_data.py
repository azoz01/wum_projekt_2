from pipelines_config import *
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    logger.info("STARTED TRAIN TEST SPLIT DATA PIPELINE")
    logger.info(f"Started reading concatenated data from {concatenated_data_path}")
    concatenated = pd.read_pickle(concatenated_data_path)
    logger.info("Started splitting data to train and test samples")
    df_train, df_test = train_test_split(concatenated, test_size=test_size)
    logger.info(f"Started saving train data to {train_data_path}")
    df_train.reset_index(drop=True).to_pickle(train_data_path)
    logger.info(f"Started saving test data to {test_data_path}")
    df_test.reset_index(drop=True).to_pickle(test_data_path)


if __name__ == "__main__":
    main()
