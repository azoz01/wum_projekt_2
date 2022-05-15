from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from model_logging import logger
import swifter
import numpy as np
import pandas as pd
import gensim.downloader


class TfidfFromDictionariesVectorizer(TransformerMixin):
    def _generate_text_from_bow_dict(self, bow_dict: dict) -> str:
        """
        converts bow dictionary to text. Each word in bow_dict
        is repeated according to their counts. Separated by " "
        Args:
            bow_dict (dict): bag of words dictionary

        Returns:
            str: generated text
        """
        nested_lists = [[str(item[0])] * item[1] for item in bow_dict.items()]
        flattened_list = [el for sublist in nested_lists for el in sublist]
        return " ".join(flattened_list)

    def __init__(self):
        logger.info("TfidfFromDictionariesVectorizer.__init__ - create tf-idf")
        self.tfidf = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=1000)

    def fit(self, X: pd.Series, y=None, **kwargs):
        logger.info(
            "TfidfFromDictionariesVectorizer.fit - started converting dictionaries to texts"
        )
        texts = X.swifter.apply(self._generate_text_from_bow_dict)
        logger.info("TfidfFromDictionariesVectorizer.fit - started fitting tf-idf")
        self.tfidf.fit(texts)
        logger.info("TfidfFromDictionariesVectorizer.fit - started fitting pca")
        transformed = self.tfidf.transform(texts)
        self.svd.fit(transformed)
        return self

    def transform(self, X: pd.Series, y=None, **kwargs) -> pd.Series:
        """
        Converts pd.series of bag of words to vectors using tf-idf
        Args:
            X (pd.Series)
            y (_type_, optional)

        Returns:
            pd.Series: pd.Series of encoded vectors (np.ndarray)
        """
        logger.info(
            "TfidfFromDictionariesVectorizer.transform - started converting dictionaries to texts"
        )
        texts = X.swifter.apply(self._generate_text_from_bow_dict)
        logger.info(
            "TfidfFromDictionariesVectorizer.transform - started transforming using tf-idf"
        )
        transformed = self.tfidf.transform(texts)
        transformed = self.svd.transform(transformed)
        transformed = pd.Series(data=[transformed[i] for i in range(transformed.shape[0])])
        return transformed


class WordEmbeddingDocVectorizer(TransformerMixin):
    def __init__(self):
        """
        Loads embedding model
        """
        logger.info("WordEmbeddingDocVectorizer.__init__ - load embedding model")
        self.wv = gensim.downloader.load("glove-wiki-gigaword-50")

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: pd.Series, y=None, **kwargs) -> pd.Series:
        """
        Converts pd.series of bag of words to vectors according to self.wv
        Args:
            X (pd.Series): Series of bag of words dictionaries
            y (_type_, optional):

        Returns:
            pd.Series: Series of documents embeddings
        """
        logger.info("WordEmbeddingDocVectorizer.transform - started")
        return X.map(self._avg_embedding_from_bow_dict)

    def _avg_embedding_from_bow_dict(self, bow_dict: dict) -> np.ndarray:
        """
        Converts bag of words dictionary to embedding. Output vector is weighted
        mean of word embeddings of words inside.
        Args:
            bow_dict (dict)

        Returns:
            np.ndarray: Output embedding
        """
        items = list(bow_dict.items())
        scaled_vectors = [
            self._get_item_or_zeros(item[0], self.wv) * item[1] for item in items
        ]
        if len(scaled_vectors) == 0:
            return self._get_zero_vector_of_embedding(self.wv)

        vectors = (
            np.stack(
                scaled_vectors,
                axis=0,
            )
        )
        words_count = np.sum(list(bow_dict.values()))
        return vectors.sum(axis=0) / words_count

    def _get_zero_vector_of_embedding(self, embedding: dict) -> np.ndarray:
        """
        Returns zero vector shaped similarly to embedding vectors
        Args:
            embedding (dict)
        Returns:
            np.ndarray: zero vector with proper shape
        """
        return np.zeros(embedding["be"].shape)

    def _get_item_or_zeros(self, word: str, embedding: dict) -> np.ndarray:
        """
        Return embedding of word if present in embedding else returns zeros
        Args:
            word (str)
            embedding (dict)

        Returns:
            np.ndarray: Output embedding or zeros
        """
        if word in embedding:
            return embedding[word]
        else:
            return self._get_zero_vector_of_embedding(embedding)


# List of all possible encodings. All must be sklearn.base.TransformerMixin
encodings_labels = ["Tf-idf", "Word embedding"]
encodings = [TfidfFromDictionariesVectorizer(), WordEmbeddingDocVectorizer()]
# encodings_labels = ["Tf-idf"]
# encodings = [TfidfFromDictionariesVectorizer()]
logger.info("Loaded encodings: " + str(encodings_labels))
