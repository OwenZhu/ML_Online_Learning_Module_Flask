#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 16:00
# @Author  : Hanwei Zhu
# @File    : config.py


MODEL_OPTION = {
    "batch_size": 64,
    "model_dir": "model/",
    "tb_dir": "graph/",
    "embedding_matrix_dir": "/mnt/word_embedding_matrix.npy",
    "vocaburary_list_dir": "/mnt/vocabulary.pickle",
    "rnn_node_num": 64,
    "rnn_layer_num": 2,
    "fully_connected_layer_1_node_num": 64,
    "fully_connected_layer_2_node_num": 64,
    "fully_connected_layer_3_node_num": 64,
    "label_dim": 3,
    "emd_dim": 50
}
