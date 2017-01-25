from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
from gensim.corpora import Dictionary

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

    with open('./cmv_sentences.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps([x for x in data.get_sentences()]))
    bag_of_words = BOPTopicDetector.fit(data)
    bag_of_words = dict((v, k) for k, v in bag_of_words.items())

    BOPFlowConstructor.connect(data)

    BOPRepresentativeSelector.select(data, bag_of_words)

    plot_data = BOPPlotter.convert(data)

    dot_graph = TopicGraph.to_dot(plot_data)
    with open('./BOPExample.dot', 'w', encoding='utf8') as f:
        f.write(dot_graph.__str__())
    print(dot_graph)