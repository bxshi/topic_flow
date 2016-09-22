from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tofo.CommentDocument import CommentDocument


class AbstractPlotter(object):
    @staticmethod
    def convert(document: CommentDocument):
        raise NotImplementedError("Abstract method")


class BinaryPlotter:
    @staticmethod
    def convert(document: CommentDocument,
                output: list,
                top_k_words=5,
                feature='bow'):
        # parent_id         parent discussion id
        # doc_id            discussion id
        # cluster_id        cluster id inside a discussion
        # parent_cluster_id
        # text              text in each discussion
        if feature == 'bow':
            for i in range(len(document.wcluster_by_sentence)):
                output.append({'parent_id': document.parent_id,
                               'doc_id': document.doc_id,
                               'cluster_id': i,
                               'parent_cluster_id': document.wcluster_by_sentence_connection[
                                   i] if document.wcluster_by_sentence_connection is not None else None,
                               'text': "\n".join(document.wtop_by_sentence[i][:top_k_words])})
        elif feature == 'bop':
            for i in range(len(document.pcluster_by_sentence)):
                output.append({'parent_id': document.parent_id,
                               'doc_id': document.doc_id,
                               'cluster_id': i,
                               'parent_cluster_id': document.pcluster_by_sentence_connection[
                                   i] if document.pcluster_by_sentence_connection is not None else None,
                               'text': "\n".join(document.ptop_by_sentence[i][:top_k_words])})
        else:
            raise NotImplementedError("does not support %s" % feature)

        for child in document.children:
            BinaryPlotter.convert(child, output, top_k_words, feature)


class BOWPlotter(AbstractPlotter):
    @staticmethod
    def convert(document: CommentDocument):
        output = list()
        BinaryPlotter.convert(document, output, top_k_words=5, feature='bow')
        return output


class BOPPlotter(AbstractPlotter):
    @staticmethod
    def convert(document: CommentDocument):
        output = list()
        BinaryPlotter.convert(document, output, top_k_words=5, feature='bop')
        return output
