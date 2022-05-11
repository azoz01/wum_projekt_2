import json
import os
import pandas as pd
import pickle as pkl
from functools import reduce
from paths import vocab_paths, dataset_names, vocab_unified_path


def rename_vocab_df(df):
    df["index"] = df.index
    df.rename(columns={0: "vocab"}, inplace=True)
    return df


def update_docword(df: pd.DataFrame, mapping: dict):
    df.rename(columns={0: "article_id", 1: "old index", 2: "count"}, inplace=True)
    df["vocab_index"] = df[df.columns[1]].map(mapping)
    return df


def unify_vocab(
    vocab_paths,
    names,
    vocab_out=vocab_unified_path,
):
    # read and preapre dfs
    dfs = [pd.read_csv(df, header=None, sep=" ") for df in vocab_paths]
    dfs = list(map(rename_vocab_df, dfs))

    # create and preapre full vocab df
    df_concat = reduce(
        lambda df1, df2: pd.concat([df1[["vocab"]], df2[["vocab"]]]), dfs
    )
    df_concat.drop_duplicates(inplace=True)
    df_concat["index"] = df_concat.index

    df_concat[["vocab"]].to_csv(
        path_or_buf=vocab_out,
        header=False,
        index=False,
    )

    # create mapping for indexes
    merged_dfs = list(
        map(
            lambda df: pd.merge(
                df_concat,
                df,
                how="inner",
                on="vocab",
                suffixes=("_merged", "_original"),
            )[["index_merged", "index_original"]],
            dfs,
        )
    )

    # NOTE: adding 1 to index is necessary to match numeration from 0 (pandas) and 1 (source files)
    index_mapping = [
        {
            (row[1] + 1): row[0]
            for row in df[["index_merged", "index_original"]].to_numpy()
        }
        for df in merged_dfs
    ]

    # save mappings
    for (mapping, name) in zip(index_mapping, names):
        with open(os.path.join("resources", "data", f"{name}_map.pkl"), "bw") as f:
            pkl.dump(mapping, f)


# NOTE: work only if docword and names paths are passed in the same order

if __name__ == "__main__":

    unify_vocab(vocab_paths, dataset_names)
