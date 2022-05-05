import pandas as pd
import pickle as pkl
from scipy import sparse
from functools import reduce

from paths import *


# Merges all converted.* files into one frame and saves
# to pickle in format (X, y) where X is sparse matrix
# with bag of words vectors and y is label of source file
# of corresponding vector
if __name__ == "__main__":
    dfs = list(map(pd.read_pickle, converted_docwords_paths))
    concatenated = reduce(lambda df1, df2: pd.concat([df1, df2], axis="index"), dfs)

    matrix = sparse.vstack(concatenated["bow_vector"].values)
    X, y = matrix, concatenated["label"]

    with open(data_all_path, "wb") as f:
        pkl.dump((X, y), f)
