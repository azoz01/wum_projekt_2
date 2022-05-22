from pipelines_config import *
from bow_dict_wrapper import BowDicstWrapper

import pandas as pd
from textstat import syllable_count


from functools import reduce


def enrich_dataset(path: str, overwrite: bool = True) -> pd.DataFrame:
    df = pd.read_pickle(path)

    add_length(df)
    add_mean_word_length(df)
    add_unique_words(df)
    add_sentiments(df)
    add_pollysylabels(df)

    if overwrite:
        df.to_pickle(train_data_path)


def add_length(df: pd.DataFrame) -> None:
    df["article_len"] = df.apply(lambda x: sum(x["bow_dict"].values()), axis=1)


def add_mean_word_length(df: pd.DataFrame) -> None:
    def _reduce_dict(d: dict) -> float:
        total_letters = sum(
            map(
                lambda entity: (len(entity[0]) * entity[1])
                if type(entity[0]) is str
                else 0,
                d.items(),
            )
        )
        total_words = reduce(lambda s, t: s + t, d.values(), 0)
        return total_letters / total_words

    df["mean_word_length"] = df.apply(lambda x: _reduce_dict(x["bow_dict"]), axis=1)


def add_unique_words(df: pd.DataFrame) -> None:
    df["total_unique_words"] = df.apply(lambda x: len(x["bow_dict"]), axis=1)


def add_sentiments(df: pd.DataFrame) -> None:
    """
    add column that describe number of words with
    absolute value of sentument greater than 0
    """

    wrapper = BowDicstWrapper(df["bow_dict"], labels=df["label"])

    tmp_df = pd.DataFrame(wrapper.tokens)
    tmp_df = tmp_df.rename(columns={0: "token"})
    tmp_df = tmp_df[["token"]]
    tmp_df["word"] = tmp_df["token"].apply(lambda x: x.text)
    tmp_df["polarity"] = tmp_df["token"].apply(lambda x: x._.blob.polarity)
    sentimental_words = set(tmp_df.loc[tmp_df["polarity"] != 0]["word"])

    def _reduce_to_sentimental(d: dict) -> int:
        return sum(
            list(
                map(
                    lambda x: x[1] if x[0] in sentimental_words else 0,
                    d.items(),
                )
            )
        )

    df["sentimental_words"] = df.apply(
        lambda row: _reduce_to_sentimental(row["bow_dict"]), axis=1
    )


def add_pollysylabels(df: pd.DataFrame) -> None:
    def _count_pollysylables(old_dict):
        count = 0
        for key in old_dict.keys():
            if type(key) is str and syllable_count(key) > 2:
                count += old_dict[key]
        return count

    df["polysyllabels"] = df["bow_dict"].apply(_count_pollysylables)


def main():
    enrich_dataset(train_data_path, overwrite=True)
    enrich_dataset(test_data_path, overwrite=True)


if __name__ == "__main__":
    main()
