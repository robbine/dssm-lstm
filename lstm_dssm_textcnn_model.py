import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from cells import SimpleLSTMCell
import math
PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class LSTMDSSMTEXTCNN(object):
    """
    The LSTM-DSSM model refering to the paper: Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
    papaer available at: https://arxiv.org/abs/1502.06922
    """

    def __init__(self,
                 num_lstm_units,
                 embed,
                 drop_out=0.5,
                 neg_num=4,
                 gradient_clip_threshold=5.0,
                 filter_sizes=[2],
                 num_filters=128,
                 sequence_length=30):
        self.queries = tf.placeholder(dtype=tf.string, shape=[None, None])  # shape: batch*len
        self.queries_length = tf.placeholder(dtype=tf.int32, shape=[None])  # shape: batch
        self.docs = tf.placeholder(dtype=tf.string, shape=[neg_num + 1, None, None])  # shape: (neg_num + 1)*batch*len
        self.docs_length = tf.placeholder(dtype=tf.int32, shape=[neg_num + 1, None])  # shape: batch*(neg_num + 1)
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.embed_size=embed.shape[1]
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.num_filters = num_filters
        self.sequence_length=sequence_length
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.
        self.drop_out = drop_out

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

        with tf.variable_scope('query_lstm'):
            self.cell_q = SimpleLSTMCell(num_lstm_units)
        with tf.variable_scope('doc_lstm'):
            self.cell_d = SimpleLSTMCell(num_lstm_units)

        self.states_q = dynamic_rnn(self.cell_q, self.embed_queries, self.queries_length, dtype=tf.float32,
                                         scope="simple_lstm_cell_query")[1][1]  # shape: batch*num_units
        self.states_d = [dynamic_rnn(self.cell_d, self.embed_docs[i], self.docs_length[i], dtype=tf.float32,
                                            scope="simple_lstm_cell_doc")[1][1] for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch*num_units
        self.prob, self.hit_prob = self.cosine_similarity(self.states_q, self.states_d)
        self.loss = -tf.reduce_mean(tf.log(self.hit_prob))


        self.embed_queries_expanded=tf.expand_dims(self.embed_queries, -1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.embed_docs_expanded=[tf.expand_dims(embed_doc, -1) for embed_doc in self.embed_docs] #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        #2.1 get features of sentence1
        conv_queries=self.conv_relu_pool_dropout(self.embed_queries_expanded, name_scope_prefix="s1") #[None,num_filters_total]
        #2.2 get features of sentence2
        conv_docs =[self.conv_relu_pool_dropout(embed_doc_expanded, name_scope_prefix="s2") for embed_doc_expanded in self.embed_docs_expanded]  # [None,num_filters_total]
        self.cnn_prob, self.cnn_hit_prob = self.cosine_similarity(conv_queries, conv_docs)
        self.cnn_loss = -tf.reduce_mean(tf.log(self.cnn_hit_prob))

        self.params = tf.trainable_variables()
        #opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)  # use Nesterov's method, according to the paper
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(self.loss + self.cnn_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, gradient_clip_threshold)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def cosine_similarity(self, query, docs):
        queries_norm = tf.sqrt(tf.reduce_sum(tf.square(query), axis=1))
        docs_norm = [tf.sqrt(tf.reduce_sum(tf.square(doc), axis=1)) for doc in docs]
        prods = [tf.reduce_sum(tf.multiply(query, doc), axis=1) for doc in docs]
        sims = tf.convert_to_tensor([(prod / (queries_norm * doc_norm)) for prod,doc_norm in zip(prods,docs_norm)]) # shape: (neg_num + 1)*batch
        gamma = tf.Variable(initial_value=1.0, expected_shape=[], dtype=tf.float32)  # scaling factor according to the paper
        sims = sims * gamma
        prob = tf.nn.softmax(sims, dim=0)  # shape: (neg_num + 1)*batch
        hit_prob = tf.transpose(prob[0])
        return prob, hit_prob

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def conv_relu_pool_dropout(self,sentence_embeddings_expanded, name_scope_prefix=None):
        # 1.loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            #with tf.name_scope(name_scope_prefix + "convolution-pooling-%s" % filter_size):
            with tf.variable_scope(name_scope_prefix + "convolution-pooling-%s" % filter_size, reuse=tf.AUTO_REUSE):
                # ====>a.create filter
                filter = tf.get_variable(name_scope_prefix+"filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer, )
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                b = tf.get_variable(name_scope_prefix+"b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 2.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        h_pool = tf.concat(pooled_outputs,
                           3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        h_pool_flat = tf.reshape(h_pool, [-1,
                                          self.num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        # 3.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        return h_drop

    def train_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length'],
                      self.dropout_keep_prob: self.drop_out}

        output_feed = [self.loss, self.update, self.states_q, self.states_d, self.prob, self.hit_prob]
        return session.run(output_feed, input_feed)

    def validate_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self.docs: docs['texts'],
                      self.docs_length: docs['texts_length'],
                      self.dropout_keep_prob: 1.0}
        output_feed = [self.loss, self.prob]
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




