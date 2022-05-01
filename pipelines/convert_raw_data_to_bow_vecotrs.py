import os
import dask.dataframe as dd
import numpy as np
from scipy.sparse import csr_array

def convert_raw_data_to_bow_vectors(name, docword_path):
    df = dd.read_csv(docword_path, compression="gzip", skiprows=3, header=None, sep=" ", blocksize=None)
    df.columns = ["doc_id", "word_id", "count"]
    max_id = df.word_id.max().compute()
    print(max_id)

    def flatten_group(group):
        ids = group["word_id"].values
        counts = group["count"].values
        bow_vector = np.zeros(max_id)
        bow_vector[ids - 1] = counts
        bow_vector = csr_array(bow_vector)
        return bow_vector

    df = (
        df.groupby("doc_id")
        .apply(flatten_group)
        .reset_index(drop=False)
        .rename(columns={0: "bow_vector"})
    )

    df["target"] = name
    df = df.compute()
    df.to_pickle(os.path.join("resources", "data", f"joined.{name}.pkl"))
    del [df]

if __name__ == "__main__":
    dataset_names = ["enron", "kos", "nytimes", "nips", "pubmed"]
    docword_paths = [
        os.path.join("resources", "data", f"docword.{name}.txt.gz")
        for name in dataset_names
    ]
    vocab_paths = [
        os.path.join("resources", "data", f"vocab.{name}.txt") for name in dataset_names
    ]

    for name, docword_path in zip(dataset_names, docword_paths):
        print(f"Processing {name}")
        convert_raw_data_to_bow_vectors(name, docword_path)