import os

dataset_names = ["kos", "nips", "nytimes", "enron"]

docword_paths = [
    os.path.join("resources", "data", f"docword.{name}.txt.gz")
    for name in dataset_names
]

vocab_paths = [
    os.path.join("resources", "data", f"vocab.{name}.txt") for name in dataset_names
]

vocab_mapping_paths = [
    os.path.join("resources", "data", f"{name}_map.pkl") for name in dataset_names
]

vocab_unified_path = os.path.join("resources", "data", "vocab_unified.csv")

converted_docwords_paths = [
    f"resources/data/converted.{name}.pkl" for name in dataset_names
]

data_all_path = f"resources/data/data_all.pkl"
