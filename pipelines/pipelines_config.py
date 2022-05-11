import logging
import os

# Logger setup - logs everything to console
logger = logging.Logger("log")
consoleHandler = logging.StreamHandler()
logFormatter = logging.Formatter("%(asctime)s %(message)s")
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# Variables used during pipelines
sample_size = 10000
test_size = 0.3
doc_freq_min_threshold = 2

# Paths
dataset_names = ["kos", "nips", "nytimes", "enron", "pubmed"]

root_data_path = os.path.join("resources", "data")

docword_paths = [
    os.path.join(root_data_path, f"docword.{name}.txt.gz") for name in dataset_names
]

docwords_sampled_paths = [
    os.path.join(root_data_path, f"docword.{name}.sample.csv") for name in dataset_names
]

vocab_paths = [
    os.path.join(root_data_path, f"vocab.{name}.txt") for name in dataset_names
]

docword_dicts_paths = [
    os.path.join(root_data_path, f"docword.dict.{name}.pkl") for name in dataset_names
]

train_data_path = os.path.join(root_data_path, "train_data.pkl")
test_data_path = os.path.join(root_data_path, "test_data.pkl")
