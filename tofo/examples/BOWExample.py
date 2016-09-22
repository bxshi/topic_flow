from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys

from tofo.processing import ToFoTextPreprocessor, BOWTopicDetector, BOWFlowConstructor, BOWRepresentativeSelector, \
    BOWPlotter
from tofo.CommentDocument import comment_document_parser
from tofo.utils.Graph import TopicGraph

if __name__ == '__main__':
    data_path = sys.argv[1]
    with open(data_path, encoding='utf-8') as f:
        json_data = json.loads("\n".join(f.readlines()), encoding='utf-8')

    preprocessor = ToFoTextPreprocessor()

    data = comment_document_parser(json_data, preprocessor)

    bag_of_words = BOWTopicDetector.fit(data)
    bag_of_words = dict((v, k) for k, v in bag_of_words.items())

    BOWFlowConstructor.connect(data)

    BOWRepresentativeSelector.select(data, bag_of_words)

    plot_data = BOWPlotter.convert(data)

    dot_graph = TopicGraph.to_dot(plot_data)

    print(dot_graph)