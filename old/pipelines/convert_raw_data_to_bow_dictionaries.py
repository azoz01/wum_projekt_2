import os
import shutil

import numpy as np
import pandas as pd
import pickle as pkl

from scipy.sparse import csr_array

from functools import partial
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import monotonically_increasing_id, col, max

from paths import *

spark = SparkSession.builder.appName("spark").getOrCreate()

docword_schema = StructType(
    [
        StructField("doc_id", IntegerType(), True),
        StructField("word_id", IntegerType(), True),
        StructField("count", IntegerType(), True),
    ]
)

max_word_id = pd.read_csv(vocab_unified_path).shape[0] - 1

# Converts all records with word counts of given doc to
# bag of words dictionary
def flatten_group(group, id_to_word_df):
    ids = group["word_id"]
    words = ids.apply(lambda id: id_to_word_df.loc[id][0]).values
    counts = group["count"].values
    return dict(zip(words, counts))


def convert_raw_data_to_bow_vectors(
    docword_path, dataset_name, vocab_mapping_path, output_path, vocab_unified_path
):
    df = (
        spark.read.format("csv")
        .option("delimiter", True)
        .option("header", False)
        .option("delimiter", " ")
        .schema(docword_schema)
        .load(docword_path)
    )
    # Drop first three rows
    df = df.withColumn("id", monotonically_increasing_id())
    df = df.where(df.id > 2)
    df = df.drop("id")

    # NOTE: Partition big csv into csvs with 10000 docs each
    n = df.select("doc_id").distinct().count() // 10000
    df = df.withColumn("part_col", col("doc_id") % n)

    # Write partitioned csvs into temporary directory
    temp_path = "temp"
    df.write.format("csv").partitionBy("part_col").save(temp_path)

    # Read and process little csvs
    # out_df = pd.DataFrame(columns=["doc_id", "bow_vector"])
    out_df = pd.DataFrame(columns=["doc_id", "bow_dict"])
    with open(vocab_mapping_path, "rb") as f:
        vocab_mapping = pkl.load(f)

    id_to_word_df = pd.read_csv(vocab_unified_path, header=None)

    for d in tqdm(os.listdir(temp_path)):

        if not d.startswith("part_col="):
            continue
        path = os.path.join(temp_path, d)
        csv_path = list(filter(lambda dir: dir.endswith(".csv"), os.listdir(path)))[0]

        temp_df = pd.read_csv(
            os.path.join(path, csv_path),
            header=None,
        )
        temp_df.columns = ["doc_id", "word_id", "count"]
        temp_df["word_id"] = temp_df["word_id"].map(vocab_mapping)

        temp_df = (
            temp_df.groupby("doc_id")
            .apply(partial(flatten_group, id_to_word_df=id_to_word_df))
            .reset_index(drop=False)
        )
        temp_df.columns = ["doc_id", "bow_dict"]
        out_df = pd.concat([out_df, temp_df]).reset_index(drop=True)
        # temp_df is needed no more so can be removed from memory
        del [temp_df]
    out_df["label"] = dataset_name
    out_df.to_pickle(output_path)
    # Clean after processing
    shutil.rmtree(temp_path)


if __name__ == "__main__":
    for name, docword_path, vocab_mapping_path, output_path in zip(
        dataset_names, docword_paths, vocab_mapping_paths, converted_docwords_paths
    ):
        print(f"Processing {name}")
        convert_raw_data_to_bow_vectors(
            docword_path, name, vocab_mapping_path, output_path, vocab_unified_path
        )
