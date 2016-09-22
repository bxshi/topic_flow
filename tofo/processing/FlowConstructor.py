from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tofo.CommentDocument import CommentDocument
from tofo.utils import rogers_tanmoto_similarity


class AbstractFlowConstructor(object):
    @staticmethod
    def connect(document: CommentDocument):
        raise NotImplementedError("This is an abstract class")


class BinaryFlowConstructor(object):
    @staticmethod
    def calculate_connection(parents, children):
        connections = [0 for x in range(children.shape[0])]
        for cid in range(children.shape[0]):
            best_parent = -1
            best_sim = -1
            for pid in range(parents.shape[0]):
                sim = rogers_tanmoto_similarity(parents[pid, :], children[cid, :])
                if best_sim < sim:
                    best_sim = sim
                    best_parent = pid
            connections.append(best_parent)
        return connections

    @staticmethod
    def calculate_centroids(clusters, sentences):
        centroids = np.zeros((len(clusters), sentences.shape[1]), dtype=np.int8)
        for i in range(centroids.shape[0]):
            centroids[i, :] = np.logical_or.reduce(sentences[clusters[i], :])
        return centroids

    @staticmethod
    def connect(document: CommentDocument, feature='bow'):

        if feature == 'bow':
            centroids = BinaryFlowConstructor.calculate_centroids(document.wcluster_by_sentence,
                                                                  document.wvec_by_sentence)
        elif feature == 'bop':
            centroids = BinaryFlowConstructor.calculate_centroids(document.pcluster_by_sentence,
                                                                  document.pvec_by_sentence)
        else:
            raise NotImplementedError("%s does not support" % feature)

        if document.children is not None and len(document.children) > 0:
            for child in document.children:
                if feature == 'bow':
                    child_centroids = BinaryFlowConstructor.calculate_centroids(child.wcluster_by_sentence,
                                                                                child.wvec_by_sentence)
                    child.wcluster_by_sentence_connection = BinaryFlowConstructor.calculate_connection(centroids,
                                                                                                       child_centroids)
                elif feature == 'bop':
                    child_centroids = BinaryFlowConstructor.calculate_centroids(child.pcluster_by_sentence,
                                                                                child.pvec_by_sentence)
                    child.pcluster_by_sentence_connection = BinaryFlowConstructor.calculate_connection(centroids,
                                                                                                       child_centroids)
                else:
                    raise NotImplementedError("%s does not support" % feature)
                BinaryFlowConstructor.connect(child, feature)


class BOWFlowConstructor(AbstractFlowConstructor):
    @staticmethod
    def connect(document: CommentDocument):
        return BinaryFlowConstructor.connect(document, 'bow')


class BOPFlowConstructor(AbstractFlowConstructor):
    @staticmethod
    def connect(document: CommentDocument):
        return BinaryFlowConstructor.connect(document, 'bop')
