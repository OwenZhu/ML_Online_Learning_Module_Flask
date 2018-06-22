#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 16:08
# @Author  : Hanwei Zhu
# @File    : models.py


import tensorflow as tf


class BaseModel(object):
    def __init__(self, opt, model_path='model/'):
        self.opt = opt
        self._X = tf.placeholder(tf.float32,
                                 shape=[None, ],
                                 name='input_x')
        self._y = tf.placeholder(tf.float32,
                                 shape=[None, ],
                                 name='input_y')
        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                 shape=[],
                                                 name='dropout_keep_prob')
        self._build_model()
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(model_path)

    def _build_model(self):
        pass

    def predict(self, ticket):
        return self.sess.run(self.output, feed_dict={self._X: ticket,
                                                     self._dropout_keep_prob: 1.0})

    def fit(self):
        pass

    def save_model(self, curr_version):
        """
        Save Tensorflow model
        """
        dir_path = self.opt["MODEL_DIR"] + str(type(self).__name__) + '_' + curr_version
        save_path = self.saver.save(self.sess, dir_path)
        print("Model saved in path: %s" % save_path)


class RNNClassifier(BaseModel):
    def __init__(self, model_path='model/'):
        super().__init__(model_path)

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
                                                         inputs=self._X,
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
            self.output = tf.arg_max(fully_connected_layer_3, 1)

        with tf.name_scope('loss'):
            self.loss = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=fully_connected_layer_3))

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
