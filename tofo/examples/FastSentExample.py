from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import numpy as np

from tofo.processing import ToFoTextPreprocessor, BOPTopicDetector, BOPFlowConstructor, BOPRepresentativeSelector, \
    BOPPlotter
from tofo.CommentDocument import comment_document_parser
from tofo.utils.Graph import TopicGraph

if __name__ == '__main__':
    data_path = sys.argv[1]
    with open(data_path, encoding='utf-8') as f:
        json_data = json.loads("\n".join(f.readlines()), encoding='utf-8')

    preprocessor = ToFoTextPreprocessor()

    data = comment_document_parser(json_data, preprocessor)

    vocab = dict()
    with open('/data/bshi/SentenceRepresentation/FastSent/FastSent_no_autoencoding_300_10_0.model.txt',
              encoding='utf8') as f:
        n_vocab, vec_dim = [int(x) for x in f.readline().strip().split()]
        word_vec = np.zeros((n_vocab, vec_dim), dtype=np.float32)

        for line in f:
            tmp = line.strip().split(' ')
            word_vec[len(vocab)] = [float(x) for x in tmp[1:]]
            vocab[tmp[0]] = len(vocab)

            if len(vocab) % 50000 == 0:
                print("loaded %d / %d" % (len(vocab), n_vocab), end='\r')
    print("Vocab loaded, size %d", len(vocab))
    print("Word vector loaded, shape", word_vec.shape)

    bag_of_words = BOPTopicDetector.fit(data)
    bag_of_words = dict((v, k) for k, v in bag_of_words.items())

    BOPFlowConstructor.connect(data)

    BOPRepresentativeSelector.select(data, bag_of_words)

    plot_data = BOPPlotter.convert(data)

    dot_graph = TopicGraph.to_dot(plot_data)

    print(dot_graph)