import csv
import itertools
import json
import re

import numpy as np
import tensorflow as tf

embed_path = './glove.42B.300d.npy'
dict_path = './glove.42B.300d.dict.json'

sentence_path = './sentences.csv'
code_path = './cmp_codes.csv'

punct_regex = re.compile(r'[^a-zA-Z0-9\-\_\']+')
symbol_regex = re.compile(r'\W+')

batch_size = 32
capacity = 5

N_NEG_SAMPLE = 2
MAX_LEN = 50


def load_glove(embed_path, dict_path):
    """
    Load Glove embedding and word-id map
    """
    glove_embedding = np.load(embed_path)
    with open(dict_path, encoding='utf8') as f:
        glove_word_map = json.loads("\n".join(f.readlines()))

    # Add a PLACEHOLDER word
    glove_word_map['__PLACEHOLDER__'] = len(glove_embedding)
    glove_embedding = np.append(glove_embedding,
                                np.zeros((1, glove_embedding.shape[1]),
                                         dtype=np.float32),
                                axis=0)
    # Add an UNKNOWN word
    glove_word_map['__UNKNOWN__'] = len(glove_embedding)
    glove_embedding = np.append(glove_embedding,
                                np.mean(glove_embedding, axis=0, keepdims=True),
                                axis=0)

    return glove_embedding, glove_word_map


def word_to_id(w, dic):
    """
    word to GloVe id, if there is no match,
    try 1) remove all symbols but hyper and underscore
    2) if still no match, remove hyper and underscore
    """
    res = list()
    try:
        res.append(dic[w])
    except KeyError:
        w = punct_regex.sub(' ', w).split()
        for x in w:
            try:
                res.append(dic[x])
            except KeyError:
                x = symbol_regex.sub(' ', x).split()
                for y in x:
                    try:
                        res.append(dic[y])
                    except KeyError:
                        res.append(dic['__UNKNOWN__'])
    return res


def load_data(sentence_path, code_path, embed_dict, train_ratio=0.9):
    sentences = list()
    targets = list()

    cmp_code = dict()
    cmp_desc = dict()
    cmp_id = dict()  # internal use only

    with open(code_path, encoding='utf8') as f:
        code_reader = csv.DictReader(f)
        for row in code_reader:
            cmp_id[row['cmp_code']] = len(cmp_code)
            cmp_code[len(cmp_code)] = row['cmp_code']
            cmp_desc[len(cmp_desc)] = row['meaning']

    with open(sentence_path, encoding='utf8') as f:
        sentence_reader = csv.DictReader(f)
        for row in sentence_reader:
            nested_sent = [word_to_id(x, embed_dict) for x in
                           row['content'].lower().split()]
            sentence = list(itertools.chain(*nested_sent))
            if len(sentence) == 0 or len(sentence) > MAX_LEN:
                continue
            padded_sentence = sentence + (
                [embed_dict['__PLACEHOLDER__']] * (MAX_LEN - len(sentence)))
            sentences.append(np.asarray(padded_sentence, dtype=np.int32))
            targets.append(cmp_id[row['cmp_code']])

    partition = np.random.choice([0, 1], len(sentences), replace=True,
                                 p=[1 - train_ratio, train_ratio])

    return np.asarray(sentences, dtype=np.int32), \
           np.asarray(targets, dtype=np.int32), \
           np.concatenate(np.argwhere(partition == 1)), \
           np.concatenate(np.argwhere(partition == 0)), cmp_code, cmp_desc


def sentence_by_topics(sent_ids, topics):
    topic_sent_map = dict()
    for sent_id in sent_ids:
        topic = int(topics[sent_id])
        if topic not in topic_sent_map:
            topic_sent_map[topic] = list()
        topic_sent_map[topic].append(sent_id)
    return topic_sent_map


def generate_training_data(sents, sent_topics, topic_sent_map,
                           pos_sample_number, neg_sample_number,
                           minibatch_size):
    np.random.shuffle(sents)
    start = 0
    while start < len(sents):
        end = min(start + minibatch_size, len(sents))
        d = np.zeros((end - start, 1 + pos_sample_number + neg_sample_number),
                     dtype=np.int32)
        d[:, 0] = sents[start:end]
        for i in range(d.shape[0]):
            d[i, 1:(pos_sample_number + 1)] = np.random.choice(
                topic_sent_map[sent_topics[d[i, 0]]],
                pos_sample_number, replace=True)
            negs = np.random.choice(sents, neg_sample_number, replace=True)
            for j, neg in enumerate(negs):
                while neg in topic_sent_map[sent_topics[d[i, 0]]]:
                    neg = np.random.choice(sents, 1)
                d[i, (pos_sample_number + 1 + j)] = neg
        yield d
        start = end


embed_vec, embed_dict = load_glove(embed_path, dict_path)

sentences, sentence_topics, \
training_id, testing_id, \
topics, topic_desc = load_data(sentence_path, code_path, embed_dict)

# data for training
training_topic_sentence_map = sentence_by_topics(training_id, sentence_topics)


def last_output(out, out_len):
    batch_size = tf.shape(out)[0]
    max_length = int(out.get_shape()[1])
    out_size = int(out.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (out_len - 1)
    flat = tf.reshape(out, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def avg_output(out, out_len):
    return tf.reduce_sum(out, reduction_indices=1) / \
           tf.cast(tf.reshape(out_len, (-1, 1)), np.float32)


with tf.device('/cpu'):
    # Constant word embedding
    word_embedding = tf.get_variable("word_embedding", shape=embed_vec.shape,
                                     dtype=np.float32, trainable=False)

    ph_word_embedding = tf.placeholder(np.float32, shape=embed_vec.shape)

    sentence_word_ids = tf.get_variable("sentence_words",
                                        initializer=tf.zeros_initializer(
                                            sentences.shape, dtype=np.int32),
                                        dtype=np.int32, trainable=False)

    ph_sentence_word_ids = tf.placeholder(np.int32, shape=sentences.shape)

    sentence_len = tf.reshape(tf.cast(tf.reduce_sum(tf.cast(
        tf.not_equal(sentence_word_ids, embed_dict['__PLACEHOLDER__']),
        tf.int32), reduction_indices=1), tf.int32), (-1, 1))

    sentence_topic_ids = tf.get_variable("sentence_topics",
                                         initializer=tf.zeros_initializer(
                                             sentence_topics.shape,
                                             dtype=np.int32),
                                         dtype=np.int32, trainable=False)

    ph_sentence_topic_ids = tf.placeholder(np.int32,
                                           shape=sentence_topics.shape)

    # One instance, one positive instance, and a set of negative instances
    sentence_input = tf.placeholder(tf.int32, [None, 2 + N_NEG_SAMPLE])

    sentence_eval_input = tf.placeholder(tf.int32, [None])

    init_word_embedding = word_embedding.assign(ph_word_embedding)
    init_sentence_word_ids = sentence_word_ids.assign(ph_sentence_word_ids)
    init_sentence_topic_ids = sentence_topic_ids.assign(ph_sentence_topic_ids)

    outputs = list()
    lens = list()
    states = list()
    sent_words_list = list()

    # all different rnn reusing the same cell
    rnn_cell = tf.nn.rnn_cell.LSTMCell(embed_vec.shape[1], state_is_tuple=True,
                                       use_peepholes=True)
    dropout_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                                     output_keep_prob=0.5,
                                                     seed=955)


    def l2_val(t, reduction_indices=2, keep_dims=False):
        l2_t = tf.reduce_sum(tf.square(t), reduction_indices=reduction_indices,
                             keep_dims=keep_dims)
        return tf.sqrt(tf.maximum(1e-12, l2_t))


    with tf.variable_scope("RNN_Lists"):
        for i in range(1 + N_NEG_SAMPLE):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # [None, 1, 50]
            sent_words = tf.nn.embedding_lookup(sentence_word_ids,
                                                sentence_input[:, i])
            sent_words_list.append(sent_words)
            sent_lens = tf.cast(
                tf.reduce_sum(tf.cast(tf.not_equal(sent_words, embed_dict['__PLACEHOLDER__']), tf.int32),
                              reduction_indices=1), tf.int32)
            sent_embedding = tf.nn.embedding_lookup(word_embedding, sent_words)  # [None, 50, 300]

            rnn_output, rnn_state = tf.nn.dynamic_rnn(dropout_rnn_cell, sent_embedding, dtype=tf.float32,
                                                      sequence_length=tf.reshape(sent_lens, (-1,)))
            # rnn_output -> [#None, max_length, #feature]
            outputs.append(rnn_output)
            lens.append(sent_lens)
            states.append(rnn_state)

        # eval
        tf.get_variable_scope().reuse_variables()
        eval_sent_words = tf.nn.embedding_lookup(sentence_word_ids, sentence_eval_input)  # [None, 50]
        eval_sent_lens = tf.cast(
            tf.reduce_sum(tf.cast(tf.not_equal(eval_sent_words, embed_dict['__PLACEHOLDER__']), tf.int32),
                          reduction_indices=1), tf.int32)
        eval_sent_embedding = tf.nn.embedding_lookup(word_embedding, eval_sent_words)  # [None, 50, 300]
        eval_rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, eval_sent_embedding, dtype=tf.float32,
                                               sequence_length=tf.reshape(eval_sent_lens, (-1,)))


        def eval_top_one_result(eval_output):
            eval_norm = l2_val(eval_output, 1, keep_dims=True)  # [None, 1]
            eval_sim_denominator = tf.matmul(eval_norm, eval_norm, transpose_b=True,
                                             name="eval_sim_denominator")  # [None, None]
            eval_sim = tf.matmul(eval_output, eval_output, transpose_b=True,
                                 name="eval_sim_matrix") / eval_sim_denominator  # [None, None]

            accuracy_list = list()

            for k in [2, 6, 11]:
                val, idx = tf.nn.top_k(eval_sim, k=k, sorted=True, name="eval_sim_topk")  # use top-2 to avoid self
                top_k_sentence_ids = tf.nn.embedding_lookup(sentence_eval_input, idx[:, 1:])
                top_k_sentence_topics = tf.reshape(tf.nn.embedding_lookup(sentence_topic_ids, top_k_sentence_ids),
                                                   (-1, k - 1))
                original_sentence_topics = tf.reshape(tf.nn.embedding_lookup(sentence_topic_ids, sentence_eval_input),
                                                      (-1, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(top_k_sentence_topics, original_sentence_topics), np.float32))
                accuracy_list.append(accuracy)
            return accuracy_list


        eval_avg_output = avg_output(eval_rnn_output, eval_sent_lens)  # [None, 300]
        eval_avg_accuracy = eval_top_one_result(eval_avg_output)

        eval_last_output = last_output(eval_rnn_output, eval_sent_lens)  # [None, 300]
        eval_last_accuracy = eval_top_one_result(eval_last_output)


    def cross_entropy_cost(output_list):
        lhs = tf.expand_dims(output_list[0], 1, name="expanded_lhs_output")  # [None, 1, #features]
        lhs_norm = tf.sqrt(tf.reduce_sum(tf.square(lhs), reduction_indices=2))  # [None, 1]
        rhs = tf.pack(output_list[1:], axis=1, name='packed_rhs_target')  # [None, NEG_SAMPLE + 1, #features]
        rhs_norm = tf.sqrt(tf.reduce_sum(tf.square(rhs), reduction_indices=2))  # [None, NEG_SAMPLE + 1]
        dot_product_sim = tf.reduce_sum(lhs * rhs, reduction_indices=2, name='dot_sim') / (
        lhs_norm * rhs_norm)  # [None, NEG_SAMPLE + 1]
        softmax_sim = tf.nn.softmax(dot_product_sim, name='softmax_dot_sim')
        cost = tf.reduce_mean(-tf.log(softmax_sim[:, 0]))
        return cost


    optimizer = tf.train.AdamOptimizer()

    # Averaged output
    averged_outputs = list()
    for output, rnn_len in zip(outputs, lens):
        averged_outputs.append(avg_output(output, rnn_len))
    averged_output_cost = cross_entropy_cost(averged_outputs)
    averged_output_train_op = optimizer.minimize(averged_output_cost)

    # Last output
    last_outputs = list()
    for output, rnn_len in zip(outputs, lens):
        last_outputs.append(last_output(output, rnn_len))
    last_output_cost = cross_entropy_cost(last_outputs)
    last_output_train_op = optimizer.minimize(last_output_cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    sess.run([init_word_embedding, init_sentence_word_ids, init_sentence_topic_ids],
             {ph_word_embedding: embed_vec, ph_sentence_word_ids: sentences, ph_sentence_topic_ids: sentence_topics})

    eval_accuracy = [0.] * 3
    max_minibatches = len(training_id) / batch_size


    def progress_bar(curr, max_n, nchars):
        progress = min(nchars, int(curr * nchars / max_n))
        return ("*" * progress) + ("-" * (nchars - progress))


    for epoch in range(100):
        for i, minibatch_data in enumerate(generate_training_data(
                training_id, sentence_topics,
                training_topic_sentence_map, 1, N_NEG_SAMPLE, batch_size)):
            print(minibatch_data)
            if i % 50 == 0 or i == max_minibatches - 1:
                eval_accuracy = sess.run(eval_avg_accuracy, {sentence_eval_input: testing_id})
            cost, _ = sess.run([averged_output_cost, averged_output_train_op], {sentence_input: minibatch_data})
            print("Epoch %d\tMinibatch %d\t[%s]\tloss %.4f\taccuracy@1 %.4f @5 %.4f @10 %.4f" % (
            epoch, i, progress_bar(i, max_minibatches, 20), cost, eval_accuracy[0], eval_accuracy[1], eval_accuracy[2]),
                  end='\r')
        print("")
