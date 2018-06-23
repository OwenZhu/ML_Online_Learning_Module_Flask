#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/23 11:02
# @Author  : Hanwei Zhu
# @File    : layers.py


import tensorflow as tf


class Layers:
    @staticmethod
    def dropout_wrapped_gru_cell(in_keep_prob, num_units=64, activation=tf.nn.relu):
        gru_cell = tf.contrib.rnn.GRUCell(num_units=num_units, activation=activation)
        rnn_layer = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=in_keep_prob)
        return rnn_layer
