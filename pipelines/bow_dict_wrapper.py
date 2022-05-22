# temporary location

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy_syllables import SpacySyllables

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("spacytextblob")
nlp.add_pipe("syllables", after="tagger")


class BowDicstWrapper:
    def __init__(self, bow_dicts, labels=None, init_dicts=True):

        self.bow_dicts = bow_dicts
        self.labels = labels

        self.labelwise_wordcount = None
        self.wordcount = None
        self.labelwise_word_frequency = None
        self.word_frequency = None
        self.vocab = None
        self.tokens = None

        if init_dicts:
            self.create_dicts()

    def create_dicts(self):
        if self.labels is not None:
            self._create_all_dicts()
        else:
            self._simple_dicts()

    def _create_vocab(self):
        self.vocab = [x for x in self.wordcount.keys() if type(x) is str]

    def _create_tokens(self):
        self.tokens = list(nlp.pipe(self.vocab))

    def _create_all_dicts(self):

        self.labelwise_wordcount = {
            "enron": {},
            "nytimes": {},
            "nips": {},
            "pubmed": {},
            "kos": {},
        }

        self.labelwise_word_frequency = {
            "enron": {},
            "nytimes": {},
            "nips": {},
            "pubmed": {},
            "kos": {},
        }

        self.wordcount = {}
        self.word_frequency = {}

        for words, label in zip(self.bow_dicts, self.labels):
            for word in words:
                if word in self.labelwise_word_frequency[label]:
                    self.labelwise_word_frequency[label][word] += 1
                    self.labelwise_wordcount[label][word] += words[word]
                else:
                    self.labelwise_word_frequency[label][word] = 1
                    self.labelwise_wordcount[label][word] = words[word]
                if word in self.word_frequency:
                    self.word_frequency[word] += 1
                    self.wordcount[word] += words[word]
                else:
                    self.word_frequency[word] = 1
                    self.wordcount[word] = words[word]

        self._create_vocab()
        self._create_tokens()

    def _simple_dicts(self):
        raise NotImplementedError("guess who didn't implement that method yet lol")
