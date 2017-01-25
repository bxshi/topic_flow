from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys

from tofo.processing import ToFoConceptProcessor
from tofo.utils.Graph import ConceptGraph

data_input = sys.argv[1]
encoding = 'utf8'

with open(data_input, 'r', encoding=encoding) as f:
    jdata = json.loads("\n".join(f.readlines()), encoding=encoding)

jdata = [x for x in jdata if "your comment has been removed" not in x['text']]

processor = ToFoConceptProcessor()

for i in range(len(jdata)):
    jdata[i]['phrases'] = processor.process(jdata[i]['text'],
                                            preprocess_rules=['no_urls', 'no_spaces', 'no_symbols',
                                                              'no_stopwords', 'no_subphrases', 'lemmatization'],
                                            process_rules=["phrase", "sentence"])
bow_list, jdata = processor.tfidf(jdata)

jdata = processor.top_k_tfidf(jdata, bow_list, 10)

graph = ConceptGraph()

res = graph.to_dot(jdata, text_field='tfidf_topk')
print(res)