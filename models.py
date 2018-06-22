#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 16:08
# @Author  : Hanwei Zhu
# @File    : models.py


import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle
from nltk import word_tokenize
import util
from abc import abstractmethod


class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt

        # load vocabulary list
        with open("/mnt/new_vocabulary.pickle", "rb") as input_file:
            self.voc = pickle.load(input_file)
        # load word embedding matrix
        emb_mat = np.load(self.opt["embedding_matrix_dir"])

        with tf.variable_scope("inputs"):
            # shape [batch_size, length of words]
            self._text_input = tf.placeholder(tf.float32,
                                              shape=[None, None],
                                              name='input_x')
            self._label = tf.placeholder(tf.float32,
                                         shape=[None, 1],
                                         name='input_y')
            self._dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                     shape=[],
                                                     name='dropout_keep_prob')

        with tf.variable_scope("embedding"):
            self.emb_mat = tf.Variable(emb_mat, trainable=False, dtype=tf.float32)
            self.sent = tf.nn.embedding_lookup(params=self.emb_mat, ids=self._text_input)

        self.saver = None
        self.sess = None
        self.writer = None
        self.output = None
        self.train_step = None
        self.loss = None

        self._build_model()
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()

    @abstractmethod
    def _build_model(self):
        pass

    def start(self, version=0, is_warm=True):
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if not is_warm or version < 0:
            self.sess.run(self.init_op)
            self.writer = tf.summary.FileWriter(self.opt["tfboard_dir"])
        else:
            self.saver.restore(self.sess, self.opt["model_dir"] + str(version))

    def predict(self, text):
        words = word_tokenize(text)
        words_emb = list(map(lambda x: util.convert_word_to_embedding_index(x, self.voc), words))
        return self.sess.run(self.output, feed_dict={self._text_input: words_emb,
                                                     self._dropout_keep_prob: 1.0})

    def fit(self, X, y, batch_size=128, epochs=100, drop_out_rate=0.6):
        for e in range(epochs):
            # shuffle training data set
            train_X, labels = shuffle(X, y, random_state=0)

            # TODO: translate words to indexes

            i = 0
            while i < train_X.shape[0]:
                start = i
                end = i + batch_size
                batch_X = train_X[start:end, :]
                batch_y = labels[start:end, :]
                i = end

                _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self._text_input: batch_X,
                                                                                 self._label: batch_y,
                                                                                 self._dropout_keep_prob: drop_out_rate
                                                                                 })
                print("epoch: {}/{}, loss: {}".format(e, epochs, loss))

    def partial_fit(self, batch_X, batch_y, drop_out_rate=0.6):
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self._text_input: batch_X,
                                                                         self._label: batch_y,
                                                                         self._dropout_keep_prob: drop_out_rate
                                                                         })

    def save_model(self, curr_version):
        """
        Save Tensorflow model
        """
        dir_path = self.opt["MODEL_DIR"] + str(type(self).__name__) + '_' + curr_version
        save_path = self.saver.save(self.sess, dir_path)
        print("Model saved in path: %s" % save_path)


class RNNClassifier(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

    def _build_model(self):
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('rnn_block'):
            gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["rnn_node_num"], activation=tf.nn.relu)
                 for _ in range(self.opt["rnn_layer_num"])])
            gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["rnn_node_num"], activation=tf.nn.relu)
                 for _ in range(self.opt["rnn_layer_num"])])

            _, h_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                         cell_bw=gru_cell_bw,
                                                         inputs=self.sent,
                                                         dtype=tf.float32)

        with tf.variable_scope('fully_connected_block'):
            fully_connected_layer_1 = tf.layers.dense(h_state,
                                                      self.opt["fully_connected_layer_1_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)
            fully_connected_layer_2 = tf.layers.dense(fully_connected_layer_1,
                                                      self.opt["fully_connected_layer_2_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)
            fully_connected_layer_3 = tf.layers.dense(fully_connected_layer_2,
                                                      self.opt["fully_connected_layer_3_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)

        with tf.name_scope('output'):
            output_layer = tf.layers.dense(fully_connected_layer_3,
                                           self.opt["label_dim"],
                                           kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer,
                                           activation=tf.nn.relu)
            self.output = tf.arg_max(output_layer, 1)

        with tf.name_scope('loss'):
            self.loss = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._label, logits=output_layer))

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
