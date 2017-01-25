import collections
import itertools
import re
import operator

import nltk
import numpy as np
from nltk.corpus import stopwords as sws
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


class ToFoConceptProcessor(object):
    def __init__(self):
        self.__sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.__stopwords = sws.words('english')
        self.__tokenizer = WordPunctTokenizer()
        self.__lemmatizer = WordNetLemmatizer()

        self.__regex_shrink = re.compile(r'[ \t]+')
        self.__regex_symbol = re.compile(r'\W+')
        self.__regex_url = re.compile(r'\(http[^\)]+\)')
        self.__np_regex = nltk.RegexpParser("NP: {<DT>?<JJ|JJS|JJR>*<NN|NNS|NNP>+}")

    def lemma(self, tup):
        """

        :param tup:
        :return:
        """
        return self.__regex_shrink.sub(" ", self.__lemmatizer.lemmatize(tup[0],
                                                                        'a' if tup[1] in ['JJ', 'JJR', 'JJS'] else (
                                                                            'v' if tup[1].startswith(
                                                                                'V') else 'n'))).strip()

    def np_chunking(self, sentences, no_stopwords=False):
        np_tree_sentences = [self.__np_regex.parse(nltk.pos_tag(self.__tokenizer.tokenize(x))) for x in sentences]
        results = list()
        for sentence in np_tree_sentences:
            result = [x for x in sentence if isinstance(x, nltk.tree.Tree)]
            results.append(
                [" ".join([y[0] for y in x if (not no_stopwords or y[0] not in self.__stopwords)]) for x in result])
            results[-1] = [x for x in results[-1] if not x == ""]

        return results

    def tokenize(self, sentences, no_stopwords=False):
        return [[y for y in self.__tokenizer.tokenize(x) if (not no_stopwords or y[0] not in self.__stopwords)] for x in
                sentences]

    def remove_sub_phrases(self, phrases):
        # get all noun phrases with at least two words
        np_list = [x for x in itertools.chain(*phrases) if ' ' in x]
        # remove all phrases if there is a super set of it.
        # For example, `card` will be removed if `green card` exists. but `discard` is not a superset of card.
        new_phrases = map(
            lambda x: None if sum([y.endswith(x) and (len(y.split(' ')) > len(x.split(' '))) for y in np_list]) else x,
            itertools.chain(*phrases))

        # remove words that contains symbols only and single character words
        return [x for x in new_phrases if (x is not None and self.__regex_symbol.match(x) is None and len(x) > 1)]

    def process(self, text, preprocess_rules=None, process_rules=None):

        # Step 1. Text preprocessing
        if preprocess_rules is not None:
            if 'no_urls' in preprocess_rules:
                text = self.__regex_url.sub(' ', text)
            # all other future preprocessing rules
            if 'no_spaces' in preprocess_rules:
                text = self.__regex_shrink.sub(' ', text)

        # Step 2. Break input text into sentences
        sentences = self.__sentence_detector.tokenize(text)

        if preprocess_rules is not None and 'no_symbols' in preprocess_rules:
            sentences = [self.__regex_symbol.sub(' ', x) for x in sentences]

        # Step 3. Break each sentences into words/phrases
        if 'phrase' in process_rules:
            word_sentences = self.np_chunking(sentences,
                                              (preprocess_rules is not None and 'no_stopwords' in preprocess_rules))
        elif 'word' in process_rules:
            word_sentences = self.tokenize(sentences,
                                           (preprocess_rules is not None and 'no_stopwords' in preprocess_rules))
        else:
            raise NotImplementedError("%s is not implemented." % process_rules)

        if preprocess_rules is not None:
            if 'no_subphrase' in preprocess_rules:
                word_sentences = self.remove_sub_phrases(word_sentences)
            if 'lower' in preprocess_rules:
                word_sentences = [[y.lower() for y in x] for x in word_sentences]
            if 'lemmatization' in preprocess_rules:
                word_sentences = [[self.__lemmatizer.lemmatize(y) for y in x] for x in word_sentences]

        if 'doc' in process_rules:
            return list(ToFoConceptProcessor.__flatten(word_sentences))
        elif 'sentence' in process_rules:
            return word_sentences
        else:
            raise NotImplementedError("process_rules should have either `doc` or `sentence`.")

    @staticmethod
    def __flatten(elems):
        """
        Flatten list of lists into a single list of elements
        :param elems:
        :return:
        """
        for elem in elems:
            if isinstance(elem, collections.Iterable) and not isinstance(elem, (str, bytes, np.ndarray)):
                for el in ToFoConceptProcessor.__flatten(elem):
                    yield el
            else:
                yield elem

    @staticmethod
    def __flatten_word_list(elems):
        """
        Flatten list of lists into list of str lists
        :param elems:
        :return:
        """
        for elem in elems:
            if isinstance(elem, collections.Iterable) and len(elem) > 0 and not isinstance(elem[0], str):
                for el in ToFoConceptProcessor.__flatten_word_list(elem):
                    yield el
            else:
                yield elem

    @staticmethod
    def __bag_of_words(sentences):
        bag_of_words = dict()
        for x in ToFoConceptProcessor.__flatten(sentences):
            if x not in bag_of_words:
                bag_of_words[x] = len(bag_of_words)

        return bag_of_words

    @staticmethod
    def tfidf(data):

        # a list of str lists, can be either sentences or docs
        doc_data = list(ToFoConceptProcessor.__flatten_word_list([x['phrases'] for x in data]))

        bow = ToFoConceptProcessor.__bag_of_words(doc_data)

        tf = np.zeros((len(doc_data), len(bow)), dtype=np.float32)
        for i in range(len(doc_data)):
            for w in doc_data[i]:
                tf[i, bow[w]] += 1

        idf_vals = np.log(float(tf.shape[0]) / np.sum(np.minimum(tf, 1), 0))

        tfidf = np.multiply(tf, idf_vals)

        idx = 0
        for i in range(len(data)):
            # sentence level
            if len(data[i]['phrases']) > 0 and isinstance(data[i]['phrases'][0], list):
                data[i]['tfidf'] = tfidf[idx:len(data[i]['phrases']), :]
                idx += len(data[i]['phrases'])
            else:
                data[i]['tfidf'] = tfidf[idx, :]
                idx += 1

        bow_list = [x[0] for x in sorted(bow.items(), key=operator.itemgetter(1))]

        return bow_list, data

    @staticmethod
    def top_k_tfidf(data, bow, topk):

        for i in range(len(data)):
            if len(data[i]['tfidf'].shape) > 1:
                raise NotImplementedError("Do not support sentence-level tfidf")

            data[i]['tfidf_topk'] = [bow[x] for x in data[i]['tfidf'].argsort()[::-1][:topk].tolist()]

        return data
