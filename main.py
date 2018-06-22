#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request, jsonify
import models
import config
import pickle
from nltk import word_tokenize
import util

app = Flask(__name__)
app.config.from_object(config)

with open("/mnt/new_vocabulary.pickle", "rb") as input_file:
    vocabulary = pickle.load(input_file)

global_step = 0
clf = None


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    # TODO: what to do when client waiting the training result
    list(map(lambda x: util.convert_word_to_embedding_index(x, vocabulary), ))
    pass


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if clf is None:
            return jsonify({'status': False, 'response': 'Train a model first'})

        formed_data = {"problem_abstract": "This is a test"}
        # json_ = request.json
        words = word_tokenize(formed_data["problem_abstract"])
        words_emb = list(map(lambda x: util.convert_word_to_embedding_index(x, vocabulary), words))
        prediction = clf.predict(words_emb)
        return jsonify({"prediction": prediction})


@app.route('/rollback', methods=['POST'])
def rollback():
    global clf
    clf = models.RNNClassifier(app.config["MODEL_OPT"])
    pass


if __name__ == '__main__':
    clf = models.RNNClassifier(app.config["MODEL_OPT"])
    app.run()
