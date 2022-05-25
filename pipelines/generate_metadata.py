from pipelines_config import *
from bow_dict_wrapper import BowDicstWrapper
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore

import swifter
import os
import shutil
import pickle as pkl
import pandas as pd
import numpy as np
from textstat import syllable_count


from functools import reduce


def enrich_dataset(path: str, metadata_path: str, **kwargs) -> pd.DataFrame:
    logger.info(f"Started reading data from {path}")
    df = pd.read_pickle(path)

    logger.info("Started adding article length")
    add_length(df)
    logger.info("Started mean word length")
    add_mean_word_length(df)
    logger.info("Started adding unique words count")
    add_unique_words(df)
    logger.info("Started adding counts of words with significant sentiment")
    add_sentiments(df)
    logger.info("Started adding pollysylabels rate")
    add_pollysylabels(df)

    metacols = df[
        [
            "article_len",
            "mean_word_length",
            "total_unique_words",
            "sentimental_words",
            "polysyllabels",
        ]
    ]

    logger.info("Started generating LDA topics")
    topics_df = get_lda_topics(df, kwargs["train_lda"])

    logger.info("Started merging topics to metadata")
    metadata = pd.concat([metacols, topics_df], axis=1)

    logger.info(f"Started saving to {metadata_path}")
    metadata.to_pickle(metadata_path)


def get_lda_topics(df: pd.DataFrame, train_lda=False) -> pd.DataFrame:
    def bow_to_list(bow):
        l = [[item[0]] * item[1] for item in bow.items()]
        l = [str(item) for sublist in l for item in sublist]
        return l

    token_lists = df["bow_dict"].swifter.apply(bow_to_list)

    # Train or load pretrained from temp
    if train_lda:
        dictionary = Dictionary(token_lists)
        bows = token_lists.apply(dictionary.doc2bow)
        lda = LdaMulticore(bows, num_topics=lda_topics_num)
        with open("temp/lda.pkl", "wb") as f:
            pkl.dump((lda, dictionary), f)
    else:
        with open("temp/lda.pkl", "rb") as f:
            lda, dictionary = pkl.load(f)
            bows = token_lists.apply(dictionary.doc2bow)

    pred = np.array([max(lda[bow], key=lambda topic: topic[1])[0] for bow in bows])
    oh = np.zeros((pred.shape[0], lda_topics_num))
    oh[list(range(oh.shape[0])), pred] = 1
    feature_names = [f"topic_{i}" for i in range(lda_topics_num)]
    return pd.DataFrame(data=oh, columns=feature_names)


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
    logger.info("STARTED GENERATING METADATA PIPELINE")
    # Directory for pretrained LDA model
    os.mkdir("temp")

    enrich_dataset(train_data_path, train_data_meta_path, train_lda=True)
    enrich_dataset(test_data_path, test_data_meta_path, train_lda=False)
    # Clear temp
    shutil.rmtree("temp")


if __name__ == "__main__":
    main()
