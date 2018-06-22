#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request, jsonify
import models
import config
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config.from_object(config)

training_data_path = "data/deepdive.csv"

global_step = 0
clf = None
curr_version = 0


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    if request.method == 'GET':
        train_data = pd.read_csv(training_data_path, encoding="ISO-8859-1")
        train_X = train_data['problem_abstract']
        train_y = train_data['Application_Status']
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=0)
        clf.fit(train_X, train_y)

        global curr_version
        clf.save_model(curr_version)
        curr_version += 1

        # TODO: what to do when client waits the training result?
        pass


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if clf is None:
            return jsonify({'status': False, 'response': 'Train a model first'})

        formed_data = {"problem_abstract": "This is a test"}
        # json_ = request.json

        prediction = clf.predict(formed_data["problem_abstract"])
        return jsonify({"prediction": prediction})


@app.route('/rollback', methods=['POST'])
def rollback():
    global clf
    clf = models.RNNClassifier(app.config["MODEL_OPT"])
    global curr_version
    curr_version -= 3
    clf.start(curr_version, is_warm=True)


if __name__ == '__main__':
    clf = models.RNNClassifier(app.config["MODEL_OPT"])
    app.run()
