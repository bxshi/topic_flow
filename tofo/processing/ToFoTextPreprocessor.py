import re

import nltk
from nltk.corpus import stopwords as sws
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


class ToFoTextPreprocessor(object):
    def __init__(self):
        self.__sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.__stopwords = sws.words('english')
        self.__tokenizer = WordPunctTokenizer()
        self.__lemmatizer = WordNetLemmatizer()

        self.__regex_shrink = re.compile(r'[ \t]+')
        self.__regex_symbol = re.compile(r'\W+')
        self.__regex_url = re.compile(r'\(http[^\)]+\)')
        self.__regex_preprocess = re.compile(r'[^a-zA-Z0-9\.\,\-\_\%\!]')
        np_grammar = r"""
  NP: {<PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
        self.__np_regex = nltk.RegexpParser(np_grammar)

    def preprocessing(self, text: str):
        """
        Remove urls (but keep hover text) and continous spaces
        :return:
        """
        return self.__regex_shrink.sub(' ', self.__regex_preprocess.sub(" ", self.__regex_url.sub(' ', text)))

    def doc_to_sentence(self, text: str):
        """
        Split input into sentences, and remove all non-alphanumerical characters
        :param text:
        :return:
        """
        a = self.__sentence_detector.tokenize(text)
        return a  # [self.__regex_symbol.sub(' ', x) for x in a]

    def doc_to_words(self, text: str):
        """
        Break input text into words, remove all stop words
        :param text:
        :return:
        """
        return [self.__regex_shrink.sub(" ", self.__regex_symbol.sub(" ", x)).lower() for x in self.__tokenizer.tokenize(text)
                if
                x not in self.__stopwords and self.__regex_shrink.sub(" ", self.__regex_symbol.sub(" ", x)) != ' ']

    def doc_to_phrases(self, text: str):
        """
        Extract noun phrases from input text, remove all noun phrases that are stop words
        :param text:
        :return:
        """
        np_tree = [x for x in self.__np_regex.parse(nltk.pos_tag(self.__tokenizer.tokenize(text))) if
                   isinstance(x, nltk.tree.Tree)]
        nphrases = [" ".join([x[0] for x in y]).lower() for y in np_tree]

        return [x for x in nphrases if x not in self.__stopwords]
