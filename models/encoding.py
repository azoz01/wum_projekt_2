from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from model_logging import logger
import swifter
import pandas as pd


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

    def fit(self, X: pd.Series, y=None, **kwargs):
        logger.info(
            "TfidfFromDictionariesVectorizer.fit - started converting dictionaries to texts"
        )
        texts = X["bow_dict"].swifter.apply(self._generate_text_from_bow_dict)
        logger.info("TfidfFromDictionariesVectorizer.fit - started fitting tf-idf")
        self.tfidf.fit(texts)
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
        texts = X["bow_dict"].swifter.apply(self._generate_text_from_bow_dict)
        logger.info(
            "TfidfFromDictionariesVectorizer.transform - started transforming using tf-idf"
        )
        transformed = self.tfidf.transform(texts)
        return transformed


# List of all possible encodings. All must be sklearn.base.TransformerMixin
encodings_labels = ["Tf-idf"]
encodings = [TfidfFromDictionariesVectorizer()]
logger.info("Loaded encodings: " + str(encodings_labels))
