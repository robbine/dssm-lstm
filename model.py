import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from cells import SimpleLSTMCell
import math
PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class LSTMDSSM(object):
    """
    The LSTM-DSSM model refering to the paper: Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
    papaer available at: https://arxiv.org/abs/1502.06922
    """

    def __init__(self,
                 num_lstm_units,
                 embed,
                 neg_num=4,
                 gradient_clip_threshold=5.0):
        self.queries = tf.placeholder(dtype=tf.string, shape=[None, None], name='queries')  # shape: batch*len
        self.queries_length = tf.placeholder(dtype=tf.int32, shape=[None], name='queries_length')  # shape: batch
        self.docs = tf.placeholder(dtype=tf.string, shape=[neg_num + 1, None, None], name='docs')  # shape: (neg_num + 1)*batch*len
        self.docs_length = tf.placeholder(dtype=tf.int32, shape=[neg_num + 1, None], name='docs_length')  # shape: (neg_num + 1)*batch

        self.word2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True
        )

        self.learning_rate = tf.Variable(0.0001, trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.momentum = tf.Variable(0.9, trainable=False, dtype=tf.float32)

        self.index_queries = self.word2index.lookup(self.queries)  # batch*len
        self.index_docs = [self.word2index.lookup(doc) for doc in tf.unstack(self.docs)]

        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        self.embed_queries = tf.nn.embedding_lookup(self.embed, self.index_queries)
        self.embed_docs = [tf.nn.embedding_lookup(self.embed, index_doc) for index_doc in self.index_docs]

        cell_fw = tf.contrib.rnn.GRUCell(num_lstm_units)
        cell_bw = tf.contrib.rnn.GRUCell(num_lstm_units)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.embed_queries, self.queries_length, dtype=tf.float32)
        self.states_q = tf.concat([state_fw, state_bw], axis=1)
        self.states_d = []
        for i in range(neg_num + 1):
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.embed_docs[i], self.docs_length[i], dtype=tf.float32)
            self.states_d.append(tf.concat([state_fw, state_bw], axis=1))
        # with tf.variable_scope('query_lstm'):
        #     self.cell_q = SimpleLSTMCell(num_lstm_units)
        # with tf.variable_scope('doc_lstm'):
        #     self.cell_d = SimpleLSTMCell(num_lstm_units)
        #
        # self.state_q = dynamic_rnn(self.cell_q, self.embed_query, self.query_length, dtype=tf.float32,
        #                                  scope="simple_lstm_cell_query")[1][1]  # shape: 1*num_units
        # self.states_q = dynamic_rnn(self.cell_q, self.embed_queries, self.queries_length, dtype=tf.float32,
        #                                  scope="simple_lstm_cell_query")[1][1]  # shape: batch*num_units
        # self.states_d = [dynamic_rnn(self.cell_d, self.embed_docs[i], self.docs_length[i], dtype=tf.float32,
        #                                     scope="simple_lstm_cell_doc")[1][1] for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch*num_units


        self.queries_norm = tf.sqrt(tf.reduce_sum(tf.square(self.states_q), axis=1))
        self.docs_norm = [tf.sqrt(tf.reduce_sum(tf.square(self.states_d[i]), axis=1)) for i in range(neg_num + 1)]

        self.prods = [tf.reduce_sum(tf.multiply(self.states_q, self.states_d[i]), axis=1) for i in range(neg_num + 1)]
        self.sims = [(self.prods[i] / (self.queries_norm * self.docs_norm[i])) for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch
        self.sims = tf.convert_to_tensor(self.sims, name='sims')
        self.gamma = tf.Variable(initial_value=1.0, expected_shape=[], dtype=tf.float32)  # scaling factor according to the paper
        self.sims = tf.multiply(self.sims, self.gamma)
        self.prob = tf.nn.softmax(self.sims, dim=0)  # shape: (neg_num + 1)*batch
        self.hit_prob = tf.transpose(self.prob[0])

        self.loss = -tf.reduce_mean(tf.log(self.hit_prob))

        self.params = tf.trainable_variables()
        self.all_params = tf.global_variables()
        #opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)  # use Nesterov's method, according to the paper
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, gradient_clip_threshold)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def print_all_parameters(self):
        for item in self.all_params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length']}

        output_feed = [self.loss, self.update, self.states_q, self.states_d, self.queries_norm, self.docs_norm, self.prods, self.sims, self.prob, self.hit_prob]
        return session.run(output_feed, input_feed)

    def validate_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length']}
        output_feed = [self.loss]
        return session.run(output_feed, input_feed)

    def predict_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length']}
        output_feed = [self.sims]
        return session.run(output_feed, input_feed)

    def test_step(self, session, queries, docs, ground_truths):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length']}
        output_feed = [self.sims]
        scores = (session.run(output_feed, input_feed)[0][0] + 1) / 2
        l = len(ground_truths)
        loss = 0
        for i in range(l):
            predict = scores[i]
            ground_truth = ground_truths[i]
            predict = max([min([predict, 1 - 1e-15]), 1e-15])
            if ground_truth == 0:
                loss += math.log(1 - predict)
            else:
                loss += math.log(predict)
        return -loss / l




