import pandas as pd
import pickle as pkl
from scipy import sparse
from functools import reduce

from paths import *


# Merges all converted.* files into one frame and saves
# to pickle
if __name__ == "__main__":
    dfs = list(map(pd.read_pickle, converted_docwords_paths))
    concatenated = reduce(lambda df1, df2: pd.concat([df1, df2], axis="index"), dfs)
    concatenated.to_pickle(data_all_path)
