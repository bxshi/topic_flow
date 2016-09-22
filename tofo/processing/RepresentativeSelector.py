from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tofo.CommentDocument import CommentDocument


class AbstractRepresentativeSelector(object):
    @staticmethod
    def select(document: CommentDocument, boo: dict):
        pass


class BinaryRepresentativeSelector:
    @staticmethod
    def rank_words(clusters, sentences, boo):
        cluster_words = list()
        for cluster in clusters:
            tmp = np.add.reduce(sentences[cluster, :])
            cluster_words.append([boo[x] for x in np.argsort(tmp)[::-1] if tmp[x] > 0])

        return cluster_words

    @staticmethod
    def select(document: CommentDocument, boo: dict, feature='bow'):
        if feature == 'bow':
            document.wtop_by_sentence = BinaryRepresentativeSelector.rank_words(document.wcluster_by_sentence,
                                                                                document.wvec_by_sentence, boo)
        elif feature == 'bop':
            document.ptop_by_sentence = BinaryRepresentativeSelector.rank_words(document.pcluster_by_sentence,
                                                                                document.pvec_by_sentence, boo)
        else:
            raise NotImplementedError("does not support %s" % feature)
        for child in document.children:
            BinaryRepresentativeSelector.select(child, boo, feature)


class BOWRepresentativeSelector(AbstractRepresentativeSelector):
    @staticmethod
    def select(document: CommentDocument, boo: dict):
        return BinaryRepresentativeSelector.select(document, boo, 'bow')


class BOPRepresentativeSelector(AbstractRepresentativeSelector):
    @staticmethod
    def select(document: CommentDocument, boo: dict):
        return BinaryRepresentativeSelector.select(document, boo, 'bop')
