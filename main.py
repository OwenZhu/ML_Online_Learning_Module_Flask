#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os
import models
import config

app = Flask(__name__)
app.config.from_object(config)

training_data_path = "data/training_data.csv"

clf = None
curr_version = None
training_buffer = None


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    if request.method == 'GET':
        # load training set
        train_data = pd.read_csv('data/training_data.csv', index_col='id', encoding="ISO-8859-1")
        train_X = train_data['problem_abstract']
        train_y = train_data[['Active', 'Planned', 'Retired']]
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=0)

        clf.fit(train_X, train_y)

        # TODO: validate the classifier, decide if save current version

        # save model after training
        global curr_version
        clf.save_model(str(curr_version))
        curr_version += 1

        # TODO: what to do when client waits the training result?
        return jsonify({"status": True, "response": "Finished training"})


@app.route('/partial_train', methods=['POST'])
def partial_train():
    if request.method == 'POST':
        if clf is None:
            return jsonify({'status': False, 'response': 'Train a model first'})

        json_ = request.json
        text_list = [t["problem_abstract"] for t in json_]
        label_list = [t["Application_Status"] for t in json_]
        clf.partial_fit(text_list, label_list)
        return jsonify({"status": True, "response": "Finished partial training"})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if clf is None:
            return jsonify({'status': False, 'response': 'Train a model first'})

        # formed_data = {"problem_abstract": "This is a test"}
        # text_list = [formed_data["problem_abstract"]]
        
        json_ = request.json
        text_list = [t["problem_abstract"] for t in json_]
        prediction = clf.predict(text_list)
        return jsonify({"status": True, "pred_list": prediction})


@app.route('/rollback', methods=['POST'])
def rollback():
    global curr_version
    if curr_version < 3:
        return jsonify({"status": False, "response": "Cannot rollback"})

    global clf
    clf = models.RNNClassifier(app.config["MODEL_OPTION"])
    curr_version -= 3
    try:
        clf.start(curr_version, is_warm_start=True)
    except Exception as e:
        print(e)
        clf.start()
        return jsonify({"status": False, "response": "Last version not found, rollback to initial version"})

    return jsonify({"status": True, "response": "Rollback to version " + str(curr_version) + "successfully"})


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree(app.config["MODEL_OPTION"]["model_dir"])
        os.makedirs(app.config["MODEL_OPTION"]["model_dir"])

        global curr_version, clf
        curr_version = 0
        clf = models.RNNClassifier(app.config["MODEL_OPTION"])
        clf.start()

        return jsonify({'status': True, 'response': 'Wipe successfully'})

    except Exception as e:
        print(e)
        return jsonify({'status': False, 'response': 'Could not remove and recreate the model directory'})


if __name__ == '__main__':
    curr_version = 0
    clf = models.RNNClassifier(app.config["MODEL_OPTION"])
    clf.start()

    # TODO: create a partial training buffer
    # columns = ['Active', 'Planned', 'Retired']
    # training_buffer = pd.DataFrame(columns=columns)

    app.run()
