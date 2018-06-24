#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 16:08
# @Author  : Hanwei Zhu
# @File    : models.py


import os
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from nltk import word_tokenize
import util
from abc import abstractmethod
from layers import Layers


class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt

        # load vocabulary list
        with open(self.opt["vocaburary_list_dir"], "rb") as input_file:
            self.voc = pickle.load(input_file)
        # load word embedding matrix
        emb_mat = np.load(self.opt["embedding_matrix_dir"])

        with tf.variable_scope("inputs"):
            # shape [batch_size, length of words]
            self._text_input = tf.placeholder(tf.int32,
                                              shape=[None, None],
                                              name='input_x')
            self._label = tf.placeholder(tf.float32,
                                         shape=[None, self.opt["label_dim"]],
                                         name='input_y')
            self._dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                     shape=[],
                                                     name='dropout_keep_prob')

        with tf.variable_scope("embedding"):
            self.emb_mat = tf.Variable(emb_mat, trainable=False, dtype=tf.float32)
            self.emb_sent = tf.nn.embedding_lookup(params=self.emb_mat, ids=self._text_input)

        self.saver = None
        self.sess = None
        self.writer = None
        self.output = None
        self.train_step = None
        self.loss = None

        self.global_step = 0

        self._build_model()
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()

    @abstractmethod
    def _build_model(self):
        pass

    def start(self, version=0, is_warm_start=False):
        """
        start ML model
        """
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        model_save_path = self.opt["model_dir"] + str(type(self).__name__) + '_' + str(version) + '/' + \
                          str(type(self).__name__)

        if is_warm_start:
            self.saver.restore(self.sess, model_save_path)
        else:
            self.sess.run(self.init_op)

        tb_save_path = self.opt["tb_dir"] + str(type(self).__name__) + str(version) + '/'
        self.writer = tf.summary.FileWriter(tb_save_path)

    def save_model(self, curr_version=0):
        """
        Save Tensorflow model
        """
        dir_path = self.opt["model_dir"] + str(type(self).__name__) + '_' + str(curr_version) + '/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = self.saver.save(self.sess, dir_path + str(type(self).__name__))
        print("Model saved in path: %s" % save_path)

    def predict(self, text_list):
        emd_word_list = []
        for index, text in enumerate(text_list):
            words = word_tokenize(text)
            emd_word_list.append(
                list(map(lambda x: util.convert_word_to_embedding_index(x, self.voc), words)))

        # padding to a matrix by '0'
        max_w = util.find_max_length(emd_word_list)
        for ws in emd_word_list:
            ws.extend([0] * (max_w - len(ws)))

        emd_text_batch = np.asarray(emd_word_list)
        pred_list = list(self.sess.run(self.output, feed_dict={self._text_input: emd_text_batch,
                                                               self._dropout_keep_prob: 1.0}))
        return pred_list

    def fit(self, X, y, batch_size=128, epochs=5, drop_out_rate=0.6):
        self.global_step = 0
        
        for e in range(epochs):
            # shuffle training set
            train_X, labels = shuffle(X, y, random_state=0)
            
            i = 0
            while i < train_X.shape[0]:
                start = i
                end = i + batch_size
                batch_X = train_X[start:end]
                batch_y = labels[start:end]
                i = end

                emd_word_list = []
                for index, text in enumerate(batch_X):
                    words = word_tokenize(text)
                    emd_word_list.append(
                        list(map(lambda x: util.convert_word_to_embedding_index(x, self.voc), words)))

                # padding to a matrix by '0'
                max_w = util.find_max_length(emd_word_list)
                for ws in emd_word_list:
                    ws.extend([0] * (max_w - len(ws)))

                emd_text_batch = np.asarray(emd_word_list)

                _, loss, summaries = self.sess.run([self.train_step, self.loss, self.merged],
                                                   feed_dict={self._text_input: emd_text_batch,
                                                              self._label: batch_y,
                                                              self._dropout_keep_prob: drop_out_rate
                                                              })
                self.writer.add_summary(summaries, self.global_step)
                self.global_step += 1

    def score(self, X, y_true):
        y_pred = self.predict(X)
        score = accuracy_score(list(np.argmax(y_true.values, axis=1)), y_pred)
        return score
                
    def partial_fit(self, batch_X, batch_y, drop_out_rate=0.6):
        self.global_step = 0
        
        emd_word_list = []
        for index, text in enumerate(batch_X):
            words = word_tokenize(text)
            emd_word_list.append(
                list(map(lambda x: util.convert_word_to_embedding_index(x, self.voc), words)))

        # padding to a matrix by '0'
        max_w = util.find_max_length(emd_word_list)
        for ws in emd_word_list:
            ws.extend([0] * (max_w - len(ws)))

        emd_text_batch = np.asarray(emd_word_list)

        _, loss, summaries = self.sess.run([self.train_step, self.loss, self.merged],
                                           feed_dict={self._text_input: emd_text_batch,
                                                      self._label: batch_y,
                                                      self._dropout_keep_prob: drop_out_rate
                                                      })
        self.writer.add_summary(summaries, self.global_step)
        self.global_step += 1


class RNNClassifier(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

    def _build_model(self):
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('rnn_block'):
            gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [Layers.dropout_wrapped_gru_cell(self._dropout_keep_prob,
                                                 num_units=self.opt["rnn_node_num"],
                                                 activation=tf.nn.relu)
                 for _ in range(self.opt["rnn_layer_num"])])
            gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [Layers.dropout_wrapped_gru_cell(self._dropout_keep_prob,
                                                 num_units=self.opt["rnn_node_num"],
                                                 activation=tf.nn.relu)
                 for _ in range(self.opt["rnn_layer_num"])])

            _, h_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell_fw,
                                                         cell_bw=gru_cell_bw,
                                                         inputs=self.emb_sent,
                                                         dtype=tf.float32)
            rnn_output = tf.concat([h_state[0][-1], h_state[1][-1]], axis=1)

        with tf.variable_scope('fully_connected_block'):
            fully_connected_layer_1 = tf.layers.dense(rnn_output,
                                                      self.opt["fully_connected_layer_1_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)
            fully_connected_layer_1 = tf.nn.dropout(fully_connected_layer_1, self._dropout_keep_prob)

            fully_connected_layer_2 = tf.layers.dense(fully_connected_layer_1,
                                                      self.opt["fully_connected_layer_2_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)
            fully_connected_layer_2 = tf.nn.dropout(fully_connected_layer_2, self._dropout_keep_prob)

            fully_connected_layer_3 = tf.layers.dense(fully_connected_layer_2,
                                                      self.opt["fully_connected_layer_3_node_num"],
                                                      kernel_initializer=w_initializer,
                                                      bias_initializer=b_initializer,
                                                      activation=tf.nn.relu)
            fully_connected_layer_3 = tf.nn.dropout(fully_connected_layer_3, self._dropout_keep_prob)

        with tf.name_scope('output'):
            output_layer = tf.layers.dense(fully_connected_layer_3,
                                           self.opt["label_dim"],
                                           kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer,
                                           activation=tf.nn.relu)
            tf.summary.histogram('logits', output_layer)
            self.output = tf.arg_max(output_layer, 1)

        with tf.name_scope('loss'):
            self.loss = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._label, logits=output_layer))
            tf.summary.scalar("training_loss", self.loss)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
