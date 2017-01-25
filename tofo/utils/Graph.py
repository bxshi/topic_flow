from graphviz import Digraph


class ConceptGraph(object):
    def __init__(self):
        pass

    @staticmethod
    def to_dot(data, id_field='id', parent_field='parent', text_field='text'):
        dot_graph = Digraph(encoding='utf-8', engine='dot', format='svg')

        for x in data:
            dot_graph.node(x[id_field],
                           x[text_field] if isinstance(x[text_field], str) else ",\n".join(x[text_field]),
                           fillcolor="#F0F8FF",
                           styled='filled',
                           id=x[id_field],
                           shape="box")

        for x in data:
            if x[parent_field] != 0:
                dot_graph.edge(x[parent_field],
                               x[id_field],
                               id="-".join([x[parent_field], x[id_field]]),
                               dir="back",
                               arrowtail="normal")

        return dot_graph


class TopicGraph:
    @staticmethod
    def cluster_id(node_id, cluster_id):
        return "_".join([str(x) for x in [node_id, cluster_id]])

    @staticmethod
    def doc_id(doc_id):
        return "cluster_" + str(doc_id)

    @staticmethod
    def to_dot(data):
        dot_graph = Digraph(encoding='utf-8', engine='dot', format='svg')

        # Draw topics for each reply/discussion
        doc_ids = [x['doc_id'] for x in data]
        for doc_id in set(doc_ids):
            graph_name = TopicGraph.doc_id(doc_id)
            clusters = [x for x in data if x['doc_id'] == doc_id]
            doc_graph = Digraph(name=graph_name,
                                graph_attr={'compound': 'true',
                                            'label': graph_name,
                                            'bgcolor': 'white' if clusters[0]['parent_id'] is not None else 'blue'})

            for cl in clusters:
                node_id = TopicGraph.cluster_id(cl['doc_id'], cl['cluster_id'])
                doc_graph.node(node_id,
                               str(cl['text']),
                               id=node_id,
                               shape='box')

            dot_graph.subgraph(doc_graph)

        # Draw reply edges
        # reply_edges = set([(x['parent_id'], x['doc_id']) for x in data if x['parent_id'] is not None])
        # for pid, did in reply_edges:
        #     dot_graph.edge(TopicGraph.doc_id(pid), TopicGraph.doc_id(did), color='black', penwidth='2.0')

        # Draw cluster edges
        for elem in data:
            if elem['parent_cluster_id'] is None:
                continue
            parent_cluster = TopicGraph.cluster_id(elem['parent_id'], elem['parent_cluster_id'])
            current_cluster = TopicGraph.cluster_id(elem['doc_id'], elem['cluster_id'])

            dot_graph.edge(parent_cluster, current_cluster, color='green')

        return dot_graph