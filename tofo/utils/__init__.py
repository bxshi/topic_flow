from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def rogers_tanmoto_similarity(lhs, rhs):
    """
            Rogers-Tanmoto Similarity
            :param lhs:
            :param rhs:
            :return:
            """
    nominator = float(np.sum(np.logical_and(lhs, rhs)) +
                      np.sum(np.logical_not(np.logical_and(lhs, rhs))))
    denominator = float(2 * np.sum(np.logical_xor(lhs, rhs)) + nominator)

    return nominator / denominator

def rogers_tanmoto_dissimilarity(lhs, rhs):
    """
            Rogers-Tanmoto disSimilarity
            :param lhs:
            :param rhs:
            :return:
            """
    nominator = float(2 * np.sum(np.logical_xor(lhs, rhs)))
    denominator = float(np.sum(np.logical_and(lhs, rhs)) +
                        np.sum(np.logical_not(np.logical_and(lhs, rhs))) +
                        nominator)
    return nominator / denominator