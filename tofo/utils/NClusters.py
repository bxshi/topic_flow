import numpy as np


def poly_fit_method(metrics, clusters):
    """

    :param data: N-by-M numpy array, where N is the number of samples, and M is the number of features.
    :param clusters: A history of clusters
    :param binary: If the input data is binary
    :return: best cluster size k
    """

    if len(metrics) == 1:
        return 1

    kclusters = [len(x) for x in clusters[:-1]]
    metrics = metrics[:-1]
    line = np.poly1d(np.polyfit(kclusters, metrics, 1))

    best_k = 1
    best_diff = 0
    for (i, k) in enumerate(kclusters):
        diff = abs(line(k) - metrics[i])
        if best_diff < diff:
            best_diff = diff
            best_k = k

    return best_k
