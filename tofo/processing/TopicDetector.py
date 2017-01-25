from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tofo.CommentDocument import CommentDocument
from tofo.utils import rogers_tanmoto_dissimilarity, rogers_tanmoto_similarity
from tofo.utils.NClusters import poly_fit_method


class AbstractTopicDetector(object):
    @staticmethod
    def fit(document: CommentDocument):
        raise NotImplementedError("Abstract method")


class BinaryTopicDetector(object):
    @staticmethod
    def __boo_helper(document: CommentDocument, boo: dict, feature):
        boo_source = document.bow_by_sentence if feature == 'bow' else document.bop_by_sentence
        boo_source = set().union(*boo_source)
        for elem in boo_source:
            if elem not in boo:
                boo[elem] = len(boo)

        for child in document.children:
            BinaryTopicDetector.__boo_helper(child, boo, feature)

    @staticmethod
    def __boo_vector(document: CommentDocument, boo: dict, feature):
        if feature == 'bow':
            v = document.words_by_sentence
        elif feature == 'bop':
            v = document.phrases_by_sentence
        else:
            raise NotImplementedError("does not support feature type %s" % feature)

        arr = np.zeros((len(v), len(boo)), dtype=np.int8)
        for i, sent in enumerate(v):
            for w in sent:
                arr[i, boo[w]] = 1

        if feature == 'bow':
            document.wvec_by_sentence = arr
        elif feature == 'bop':
            document.pvec_by_sentence = arr
        else:
            raise NotImplementedError("does not support feature type %s" % feature)

        for child in document.children:
            BinaryTopicDetector.__boo_vector(child, boo, feature)

    @staticmethod
    def __similarity(lhs, rhs):
        """
        Rogers-Tanmoto Similarity
        :param lhs:
        :param rhs:
        :return:
        """
        return rogers_tanmoto_similarity(lhs, rhs)

    @staticmethod
    def __dissimilarity(lhs, rhs):
        """
        Rogers-Tanmoto disSimilarity
        :param lhs:
        :param rhs:
        :return:
        """
        return rogers_tanmoto_dissimilarity(lhs, rhs)

    @staticmethod
    def __local_dissimilarity(lhs, rhs, locality):
        """
        Rogers-Tanmoto disSimilarity with locality
        :param lhs:
        :param rhs:
        :return:
        """
        nominator = float(2 * np.sum(np.logical_xor(lhs, rhs)))
        denominator = float(np.sum(np.logical_and(lhs, rhs)) +
                            np.sum(np.logical_and(np.logical_not(np.logical_and(lhs, rhs)), locality)) +
                            nominator)
        if denominator == 0:
            print("bingo")
        return nominator / denominator

    @staticmethod
    def __dist_matrix(m):
        """
        Calculate distance matrix with `__dissimilarity`
        :param m:
        :return: minimum position (dist, i, j) and distance matrix
        """
        dist_mat = np.zeros((m.shape[0], m.shape[0]), dtype=np.float32)
        min_dist = (2, -1, -1)
        locality = np.logical_and.reduce(m)
        avg = 0
        for i in range(m.shape[0]):
            for j in range(i + 1, m.shape[0]):
                dist_mat[i, j] = BinaryTopicDetector.__local_dissimilarity(m[i, :], m[j, :], locality)
                avg += dist_mat[i, j]
                if min_dist[0] > dist_mat[i, j]:
                    min_dist = (dist_mat[i, j], i, j)

        return min_dist, dist_mat, avg / (m.shape[0] ** 2) * 2

    @staticmethod
    def __cluster_helper(m):
        # initialize cluster ids for each sentence
        clusters = [[x] for x in range(m.shape[0])]
        clusters_history = [clusters]
        metric_history = list()
        elems = m

        while len(elems) > 1:
            (mindist, i, j), distmat, metric = BinaryTopicDetector.__dist_matrix(elems)
            # print("current cluster state", clusters)
            # print("current number of elems", elems.shape)
            # print("combine %d,%d, distance %.2f" % (i, j, mindist))
            next_elems = np.zeros((elems.shape[0] - 1, elems.shape[1]), dtype=np.int8)
            next_clusters = list()
            next_pos = 0
            while next_pos < i:
                next_elems[next_pos, :] = elems[next_pos, :]
                next_clusters.append(clusters[next_pos])
                next_pos += 1

            # create new centroid
            next_elems[next_pos, :] = np.logical_or(elems[i, :], elems[j, :])
            next_clusters.append(([clusters[i]] if isinstance(clusters[i], int) else clusters[i]) +
                                 ([clusters[j]] if isinstance(clusters[j], int) else clusters[j]))
            next_pos += 1

            for p in range(i + 1, j):
                next_elems[next_pos, :] = elems[p, :]
                next_clusters.append(clusters[p])
                next_pos += 1

            j += 1
            while j < elems.shape[0]:
                next_elems[next_pos, :] = elems[j, :]
                next_clusters.append(clusters[j])
                next_pos += 1
                j += 1

            elems = next_elems
            clusters = next_clusters
            clusters_history.append(clusters)
            metric_history.append(metric)

        metric_history.append(0)

        best_k = poly_fit_method(metric_history, clusters_history)

        return clusters_history[-best_k]

    @staticmethod
    def __boo_clustering(document: CommentDocument, feature):
        if feature == 'bow':
            m = document.wvec_by_sentence
            document.wcluster_by_sentence = BinaryTopicDetector.__cluster_helper(m)
        elif feature == 'bop':
            m = document.pvec_by_sentence
            document.pcluster_by_sentence = BinaryTopicDetector.__cluster_helper(m)
        else:
            raise NotImplementedError("does not support feature type %s" % feature)

        for child in document.children:
            BinaryTopicDetector.__boo_clustering(child, feature)

    @staticmethod
    def fit(document: CommentDocument, feature="bow"):
        # create bag of words/phrases
        bag_of_obj = dict()
        BinaryTopicDetector.__boo_helper(document, bag_of_obj, feature)
        # create boo vector
        BinaryTopicDetector.__boo_vector(document, bag_of_obj, feature)
        # run clustering algorithm
        BinaryTopicDetector.__boo_clustering(document, feature)

        return bag_of_obj


class BOWTopicDetector(AbstractTopicDetector):
    @staticmethod
    def fit(document: CommentDocument):
        return BinaryTopicDetector.fit(document, 'bow')


class BOPTopicDetector(AbstractTopicDetector):
    @staticmethod
    def fit(document: CommentDocument):
        return BinaryTopicDetector.fit(document, 'bop')
